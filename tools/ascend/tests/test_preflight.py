"""Host-only failure handling; mocks never execute or replace an NPU kernel."""
import importlib.util, json, sys
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]/'validate.py'
spec=importlib.util.spec_from_file_location('ascend_validate',P)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def test_missing_tools_and_sdk_cannot_pass(monkeypatch):
    monkeypatch.setattr(v.shutil,'which',lambda _:None)
    def missing(_):raise OSError('deliberately absent test runtime')
    monkeypatch.setattr(v.ctypes,'CDLL',missing)
    problems=v.preflight(None)
    assert all(x in problems for x in ['cargo','rustc','g++','ASCEND_HOME_PATH/--toolkit'])
    assert any('AscendCL:' in x for x in problems)

def test_preflight_failed_summary_is_not_gpu_success(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',['validate.py','--output',str(tmp_path),'--preflight-only'])
    monkeypatch.setattr(v,'preflight',lambda _:['CANN unavailable'])
    assert v.main()==1
    report=json.loads((tmp_path/'summary.json').read_text())
    assert report['status']=='preflight_failed' and report['npu_passed'] is False
    assert report['stages']==[]

def test_preflight_success_is_explicitly_incomplete(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'argv',['validate.py','--output',str(tmp_path),'--preflight-only'])
    monkeypatch.setattr(v,'preflight',lambda _:[])
    assert v.main()==3
    report=json.loads((tmp_path/'summary.json').read_text())
    assert report['status']=='preflight_only_not_validated' and report['npu_passed'] is False

@pytest.mark.parametrize('code,stdout',[(1,'SDK error'),(0,'no native device marker')])
def test_failed_stage_or_missing_final_marker_never_passes(monkeypatch,tmp_path,code,stdout):
    from types import SimpleNamespace
    monkeypatch.setattr(sys,'argv',['validate.py','--output',str(tmp_path),'--toolkit',str(tmp_path/'sdk')])
    monkeypatch.setattr(v,'preflight',lambda _:[])
    def run(cmd,**kw):return SimpleNamespace(returncode=code,stdout=stdout)
    monkeypatch.setattr(v.subprocess,'run',run)
    assert v.main()==1
    report=json.loads((tmp_path/'summary.json').read_text())
    assert report['npu_passed'] is False and report['status']=='failed'
    assert not report['stages'][-1]['accepted']
    assert report['stages'][-1]['stage']==('sdk_abi' if code else 'npu_numeric')
