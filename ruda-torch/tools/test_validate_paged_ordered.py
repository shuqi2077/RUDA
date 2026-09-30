"""Parser/runner tests. XML/log fixtures are not device execution evidence."""
import importlib.util
from pathlib import Path
import sys
import pytest
D=Path(__file__).parent
sys.path.insert(0,str(D))
import validate_paged_ordered as v
NAMES=[v.TEST+'::test_a',v.TEST+'::test_b[float32]']

def xml(tmp_path,variant='ok'):
    text='<testsuite><testcase classname="x" name="test_a"/><testcase classname="x" name="test_b[float32]"/></testsuite>'
    if variant in ('failure','error','skipped'):text=text.replace('name="test_a"/>',f'name="test_a"><{variant}/></testcase>')
    if variant=='duplicate':text=text.replace('test_b[float32]','test_a')
    if variant=='empty':text='<testsuite/>'
    if variant=='malformed':text='<testsuite'
    p=tmp_path/'result.xml';p.write_text(text);return p

def test_valid(tmp_path):assert v.check_result(xml(tmp_path),v.MARKER,0,NAMES)==2
@pytest.mark.parametrize('variant',['failure','error','skipped','duplicate','empty','malformed'])
def test_invalid_xml(tmp_path,variant):
    with pytest.raises(Exception):v.check_result(xml(tmp_path,variant),v.MARKER,0,NAMES)
@pytest.mark.parametrize('code',[1,2,124,-11])
def test_nonzero_exit(tmp_path,code):
    with pytest.raises(ValueError):v.check_result(xml(tmp_path),v.MARKER,code,NAMES)

def test_no_marker(tmp_path):
    with pytest.raises(ValueError,match='marker'):v.check_result(xml(tmp_path),'',0,NAMES)

def test_collection():assert v.collect_names('\n'.join(NAMES)+'\n2 tests collected')==NAMES
@pytest.mark.parametrize('text',['','\n'.join(NAMES+NAMES),'0 tests collected'])
def test_bad_collection(text):
    with pytest.raises(ValueError):v.collect_names(text)
@pytest.mark.parametrize('tool',v.TOOLS)
def test_missing_sanitizer_summary(tmp_path,tool):
    with pytest.raises(ValueError):v.check_result(xml(tmp_path),v.MARKER,0,NAMES,tool)
@pytest.mark.parametrize('summary',['ERROR SUMMARY: 1 errors','RACECHECK SUMMARY: 1 hazards displayed (0 errors, 1 warnings)'])
def test_sanitizer_error(tmp_path,summary):
    with pytest.raises(ValueError):v.check_result(xml(tmp_path),v.MARKER+'\n'+summary,0,NAMES,'racecheck' if 'RACECHECK' in summary else 'memcheck')

def test_preflight_failure_is_recorded(tmp_path,monkeypatch):
    monkeypatch.setattr(v.shutil,'which',lambda _:None)
    out=tmp_path/'out';assert v.main(['--build','--output',str(out)])==2
    import json
    result=json.loads((out/'summary.json').read_text());assert not result['gpu_validated'] and result['errors']

def test_result_reuse_refused(tmp_path):
    (tmp_path/'summary.json').write_text('{}')
    with pytest.raises(SystemExit):v.main(['--output',str(tmp_path)])
