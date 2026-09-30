"""Host tests of fail-closed acceptance logic, not GPU execution."""
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
import validate_router as v


def report(tmp_path,n=2,tag=None):
    suite=ET.Element('testsuite')
    for i in range(n):
        test=ET.SubElement(suite,'testcase',name='t'+str(i),classname='gpu')
        if tag and i==0:ET.SubElement(test,tag)
    path=tmp_path/'r.xml';ET.ElementTree(suite).write(path);return path


def test_exact_python_success(tmp_path):
    assert v.check_python_report(report(tmp_path),v.PY_MARKER,0,2)==2


@pytest.mark.parametrize('reason',['code','zero','count','marker','failure','error','skipped','duplicates'])
def test_bad_python_reports_fail(tmp_path,reason):
    file=report(tmp_path,tag=reason if reason in ('failure','error','skipped') else None)
    if reason=='duplicates':
        tree=ET.parse(file);cs=list(tree.getroot());cs[1].set('name',cs[0].get('name'));tree.write(file)
    with pytest.raises(ValueError):v.check_python_report(file,'' if reason=='marker' else v.PY_MARKER,
        1 if reason=='code' else 0,0 if reason=='zero' else (3 if reason=='count' else 2))


@pytest.mark.parametrize('text',[
    'test result: ok. 0 passed; 0 failed; 0 ignored;',
    'test result: ok. 12 passed; 0 failed; 1 ignored;',
    'test result: FAILED. 11 passed; 1 failed; 0 ignored;',
    'test result: ok. 12 passed; 0 failed; 0 ignored;',
])
def test_missing_rust_execution_not_accepted(text):
    with pytest.raises(ValueError):v.check_rust_report(text,0)


def test_rust_runtime_marker_and_exact_count():
    text=v.RUST_MARKER+'CudaRuntime\ntest result: ok. 12 passed; 0 failed; 0 ignored;'
    assert v.check_rust_report(text,0)==12


@pytest.mark.parametrize('text,tool',[
    ('','memcheck'),('ERROR SUMMARY: 1 errors','memcheck'),
    ('RACECHECK SUMMARY: 1 hazards displayed (0 errors, 1 warnings)','racecheck'),
    ('ERROR SUMMARY: 0 errors','racecheck'),
])
def test_sanitizer_error_or_missing_output_fails(text,tool):
    with pytest.raises(ValueError):v.check_sanitizer(text,tool)


def test_unique_test_collection_required():
    names='tests/test_router_gpu.py::test_a\ntests/test_router_gpu.py::test_b\n2 tests collected'
    assert len(v.collect_names(names))==2
    with pytest.raises(ValueError):v.collect_names(names+'\ntests/test_router_gpu.py::test_a')
    with pytest.raises(ValueError):v.collect_names('0 tests collected')


def test_v29_validator_test_target_exists_after_fix():
    path=Path(__file__).with_name('validate_v29_training.py')
    text=path.read_text();assert 'parents[1]' in text
    root=path.resolve().parents[1]
    assert (root/'python/tests/test_v29_gpu.py').is_file()
