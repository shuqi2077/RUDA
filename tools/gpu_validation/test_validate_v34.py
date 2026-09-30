"""Acceptance parser tests, not device execution."""
import sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import validate_v34 as v

def samples():
    return '\n'.join(f'RUDA_V34_PRUNING_BENCH queries={n} length={l} mode={m} sample={s} prune={p} seconds_with_readback=0.001'
        for n,l,m in [(32,128,0),(256,256,0),(256,256,3)] for s in range(7) for p in ['true','false'])
def test_full_samples():assert len(v.benchmark_samples(samples()))==42
@pytest.mark.parametrize('mode',['empty','missing','duplicate','zero','nan','unknown'])
def test_invalid_benchmark(mode):
    text=samples()
    if mode=='empty':text=''
    if mode=='missing':text='\n'.join(text.splitlines()[1:])
    if mode=='duplicate':text+='\n'+text.splitlines()[0]
    if mode=='zero':text=text.replace('0.001','0.000',1)
    if mode=='nan':text=text.replace('0.001','nan',1)
    if mode=='unknown':text=text.replace('queries=32','queries=33',1)
    with pytest.raises(ValueError):v.benchmark_samples(text)
def test_preflight_fail_closed(tmp_path,monkeypatch):
    monkeypatch.setattr(v.shutil,'which',lambda _:None)
    assert v.main(['--build','--output',str(tmp_path)])==2
    r=json.loads((tmp_path/'summary.json').read_text());assert r['errors'] and not r['rust_gpu_validated'] and not r['pytorch_gpu_validated']
def test_preserve_existing_evidence(tmp_path):
    (tmp_path/'summary.json').write_text('{}')
    with pytest.raises(SystemExit):v.main(['--output',str(tmp_path)])
