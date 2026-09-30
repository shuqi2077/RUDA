import json
import pytest
import validate_v31 as v

GOOD='test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 4 filtered out\nRUDA_V31_GROUPED_GPU_EXECUTED=CudaRuntime'

def test_accept_complete():assert v.check_rust_report(GOOD,0,13,'RUDA_V31_GROUPED_GPU_EXECUTED=')==13
@pytest.mark.parametrize('text,code,expected,marker',[
 (GOOD,1,13,None),(GOOD,0,0,None),(GOOD,0,12,None),
 (GOOD.replace('0 failed','1 failed'),0,13,None),
 (GOOD.replace('0 ignored','1 ignored'),0,13,None),
 (GOOD,0,13,'missing-marker'),('test result: ok. 0 passed; 0 failed; 0 ignored;',0,13,None),
 (GOOD+'\n'+GOOD,0,13,None)])
def test_reject_incomplete(text,code,expected,marker):
 with pytest.raises(ValueError):v.check_rust_report(text,code,expected,marker)
@pytest.mark.parametrize('tool',['memcheck','initcheck','synccheck'])
def test_sanitizer_success(tool):v.check_sanitizer('ERROR SUMMARY: 0 errors',tool)
@pytest.mark.parametrize('tool',['memcheck','initcheck','synccheck','racecheck'])
def test_no_sanitizer_summary_rejected(tool):
 with pytest.raises(ValueError):v.check_sanitizer('',tool)
@pytest.mark.parametrize('text',['ERROR SUMMARY: 1 errors',
 'RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)',
 'RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)'])
def test_sanitizer_problems(text):
 with pytest.raises(ValueError):v.check_sanitizer(text,'racecheck')
def test_racecheck_zero():v.check_sanitizer('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)','racecheck')
def test_collect_unique():assert v.test_names('a::b: test\na::c: test\n')==['a::b','a::c']
@pytest.mark.parametrize('text',['','a::b: test\na::b: test\n'])
def test_collect_fail(text):
 with pytest.raises(ValueError):v.test_names(text)
def test_select_compiled_binary():
 obj={'reason':'compiler-artifact','profile':{'test':True},'target':{'name':'rublas'},'executable':'/tmp/tests'}
 assert v.executable(json.dumps(obj),'rublas')=='/tmp/tests'
 with pytest.raises(ValueError):v.executable(json.dumps(obj),'other')
def test_actual_source_test_names_match_inventory():
 import re
 for package,target,feature,prefix,names,marker,bf16 in v.SCOPES:
  path=v.ROOT/('ruBLAS/src/tensor_grouped/tests_backward.rs' if package=='rublas' else 'ruDNN/src/moe/tests_v31.rs')
  actual=set(re.findall(r'fn (v31_\w+)\(',path.read_text()))
  assert actual==names and bf16 in path.read_text()

def test_preflight_missing_toolchain_is_failure(tmp_path,monkeypatch):
 monkeypatch.setattr(v.shutil,'which',lambda name:None)
 assert v.main(['--output',str(tmp_path)])==2
 report=json.loads((tmp_path/'summary.json').read_text())
 assert not report['gpu_validated'] and report['runs']==[]
 assert 'missing toolchain' in report['errors'][0]
