"""Host-only validation parser tests; synthesized XML is never GPU evidence."""
import importlib.util
from pathlib import Path
import pytest
P=Path(__file__).with_name('validate_training.py');S=importlib.util.spec_from_file_location('vt',P);M=importlib.util.module_from_spec(S);S.loader.exec_module(M)

def xml(tmp_path,count=2,tag='',same=False):
    p=tmp_path/'t.xml';p.write_text('<testsuites><testsuite>'+''.join(f'<testcase classname="gpu" name="t{0 if same else i}">{tag}</testcase>' for i in range(count))+'</testsuite></testsuites>');return p

def test_valid_plain(tmp_path):assert M.check_report(xml(tmp_path),M.MARKER,0,2)==2
@pytest.mark.parametrize('tag',['<skipped/>','<failure/>','<error/>'])
def test_reject_failed_or_skipped(tmp_path,tag):
    with pytest.raises(ValueError):M.check_report(xml(tmp_path,tag=tag),M.MARKER,0,2)
@pytest.mark.parametrize('count',[0,1,3])
def test_reject_missing_tests(tmp_path,count):
    with pytest.raises(ValueError):M.check_report(xml(tmp_path,count),M.MARKER,0,2)

def test_reject_duplicate(tmp_path):
    with pytest.raises(ValueError):M.check_report(xml(tmp_path,same=True),M.MARKER,0,2)

def test_reject_missing_marker(tmp_path):
    with pytest.raises(ValueError):M.check_report(xml(tmp_path),'host reference only',0,2)
@pytest.mark.parametrize('code',[1,2,86,137,-9])
def test_bad_exit(tmp_path,code):
    with pytest.raises(ValueError):M.check_report(xml(tmp_path),M.MARKER,code,2)
@pytest.mark.parametrize('tool',['memcheck','initcheck','synccheck'])
def test_sanitizer_summary_required(tmp_path,tool):
    p=xml(tmp_path)
    with pytest.raises(ValueError):M.check_report(p,M.MARKER,0,2,tool)
    with pytest.raises(ValueError):M.check_report(p,M.MARKER+' ERROR SUMMARY: 1 errors',0,2,tool)
    assert M.check_report(p,M.MARKER+' ERROR SUMMARY: 0 errors',0,2,tool)==2

def test_race_summary(tmp_path):
    p=xml(tmp_path);good=' RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
    assert M.check_report(p,M.MARKER+good,0,2,'racecheck')==2
    with pytest.raises(ValueError):M.check_report(p,M.MARKER+good.replace('0 warnings','1 warnings'),0,2,'racecheck')

def test_failed_resume_does_not_overwrite_prior_evidence(tmp_path, monkeypatch):
    import json
    prior={'identity':{'version':'old'},'runs':{'0-plain':{'validated':True}}}
    target=tmp_path/'summary.json';original=json.dumps(prior);target.write_text(original)
    monkeypatch.delenv('RUDA_PTX_VERSION',raising=False)
    assert M.main(['--output',str(tmp_path),'--resume'])==2
    assert target.read_text()==original
    attempts=list(tmp_path.glob('resume-attempt-*.json'))
    assert len(attempts)==1
    assert json.loads(attempts[0].read_text())['prior_evidence_preserved']

def test_resume_identity_mismatch_keeps_original(tmp_path, monkeypatch):
    import json, types, os
    root=tmp_path/'repo';root.mkdir();(root/'Cargo.lock').write_text('lock')
    tests=root/'ruda-torch/python/tests';tests.mkdir(parents=True);(tests/'test_training_gpu.py').write_text('test');(tests/'test_optimizer_fused_gpu.py').write_text('fused');(tests/'test_gradient_stats_hierarchy_gpu.py').write_text('hierarchy')
    lib=root/'native.so';lib.write_bytes(b'native');cpp=root/'cpp.so';cpp.write_bytes(b'cpp')
    out=tmp_path/'results';out.mkdir();target=out/'summary.json';original='{"identity":{"old":true},"runs":{}}';target.write_text(original)
    monkeypatch.setattr(M,'ROOT',root);monkeypatch.setattr(M,'probe_gpu',lambda:{'compute_capability':[7,5]})
    monkeypatch.setenv('RUDA_CUDA_COMPILER','ptx');monkeypatch.setenv('RUDA_PTX_VERSION','8.0');monkeypatch.setenv('RUDA_TORCH_LIBRARY',str(lib))
    def run(cmd,**kwargs):
        kwargs['stdout'].write(json.dumps({'torch':'test','cpp':str(cpp)})+'\n')
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(M.subprocess,'run',run)
    assert M.main(['--output',str(out),'--resume'])==2
    assert target.read_text()==original
    assert len(list(out.glob('resume-attempt-*.json')))==1
