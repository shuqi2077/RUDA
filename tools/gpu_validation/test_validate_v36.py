"""Host checks for strict acceptance logic; not GPU execution."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/gpu_validation'))
import validate_v36 as v

def sample_log():
    return '\n'.join(
        f'RUDA_V36_HISTORY_COMPACTION_BENCH mla={mla} queries={n} length={length} reserved={reserved} dim={d} sample={i} compact={c} seconds_with_readback=0.01'
        for mla,n,length,reserved,d in [('false',32,128,0,128),('false',32,128,240,128),('true',32,128,240,512)]
        for i in range(7) for c in ('false','true'))

def test_all_samples_required():
    assert len(v.benchmark_samples(sample_log()))==42

@pytest.mark.parametrize('mode',['missing','duplicate','zero','negative','nan','wrongdim','wrongsample'])
def test_bad_samples_refused(mode):
    text=sample_log()
    if mode=='missing':text='\n'.join(text.splitlines()[:-1])
    elif mode=='duplicate':text+='\n'+text.splitlines()[0]
    elif mode=='zero':text=text.replace('=0.01','=0',1)
    elif mode=='negative':text=text.replace('=0.01','=-1',1)
    elif mode=='nan':text=text.replace('=0.01','=NaN',1)
    elif mode=='wrongdim':text=text.replace('dim=128','dim=129',1)
    else:text=text.replace('sample=0','sample=9',1)
    with pytest.raises(ValueError):v.benchmark_samples(text)

def test_expected_native_test_names_exist():
    import re
    code=(ROOT/'ruDNN/src/paged_attention/tests_history_compaction.rs').read_text()
    actual=set(re.findall(r'#\[test\]\s*fn (v36_[a-z0-9_]+)',code))
    assert actual==v.NAMES and len(actual)==21

@pytest.mark.parametrize('code,text',[
    (1,'test result: ok. 21 passed; 0 failed; 0 ignored;'),
    (0,'test result: ok. 0 passed; 0 failed; 0 ignored;'),
    (0,'test result: ok. 20 passed; 0 failed; 1 ignored;'),
    (0,'test result: ok. 21 passed; 0 failed; 0 ignored;'),
])
def test_missing_marker_skip_zero_exit_not_accepted(code,text):
    with pytest.raises(ValueError):v.check_rust_report(text,code,21,v.MARKER)

def test_correct_marked_native_report_parses():
    assert v.check_rust_report(v.MARKER+'\ntest result: ok. 21 passed; 0 failed; 0 ignored;',0,21,v.MARKER)==21

@pytest.mark.parametrize('tool', ['memcheck','initcheck','synccheck','racecheck'])
def test_absent_sanitizer_summary_fails(tool):
    with pytest.raises(ValueError):v.check_rust_report(v.MARKER+'\ntest result: ok. 21 passed; 0 failed; 0 ignored;',0,21,v.MARKER,tool)

def test_missing_compiler_preflight_preserves_failure_evidence(monkeypatch,tmp_path):
    import json
    monkeypatch.setattr(v.shutil,'which',lambda _: None)
    assert v.main(['--output',str(tmp_path)])==2
    report=json.loads((tmp_path/'summary.json').read_text())
    assert report['rust_gpu_validated'] is False
    assert report['commands']==[] and report['errors']

def test_preexisting_summary_not_overwritten(tmp_path):
    p=tmp_path/'summary.json';p.write_text('keep')
    with pytest.raises(SystemExit):v.main(['--output',str(tmp_path)])
    assert p.read_text()=='keep'
