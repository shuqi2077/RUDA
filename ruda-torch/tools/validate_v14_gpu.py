#!/usr/bin/env python3
"""Strict hardware gate. Builds real Rust/C++, runs PTX tests in sync/async
processes, and writes evidence; NEVER changes async default or claims skipped
hardware tests passed. No install is performed unless --build is explicit.
"""
import argparse,ctypes,json,os,platform,shutil,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');p.add_argument('--output',type=Path,default=Path('v14-hardware-validation'));a=p.parse_args()
root=Path(__file__).resolve().parents[2];a.output.mkdir(parents=True,exist_ok=True)
report={'rust_compiled':False,'gpu_validated':False,'default_async_changed':False,'platform':platform.platform(),'commands':[]}
def finish(code,reason=None):
    if reason:report['error']=reason
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(code)
if not shutil.which('cargo'):finish(2,'Rust/Cargo is required and was not found')
if not os.environ.get('RUDA_PTX_VERSION'):finish(2,'Set RUDA_PTX_VERSION to a version supported by your driver')
try:ctypes.CDLL('libcuda.so.1')
except OSError as e:finish(2,f'NVIDIA driver unavailable: {e}')
env=dict(os.environ,RUDA_REQUIRE_GPU='1',RUDA_CUDA_COMPILER='ptx')
env.setdefault('RUDA_TORCH_LIBRARY',str(root/'target/release/libruda_torch_native.so'))
def run(cmd,**extra):
    current=dict(env,**extra);idx=len(report['commands']);log=a.output/f'{idx:02d}.log'
    with log.open('w') as f:r=subprocess.run(cmd,cwd=root,env=current,stdout=f,stderr=subprocess.STDOUT)
    report['commands'].append({'cmd':cmd,'returncode':r.returncode,'log':str(log),'env':extra})
    if r.returncode:finish(1,'A required native build/hardware test failed; see command log')
if a.build:
    run(['cargo','build','-p','ruda-torch-native','--release']);report['rust_compiled']=True
    run([sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','-e','ruda-torch/python'])
run(['cargo','test','-p','ruda-torch-native','--release','v14_gpu_','--','--ignored','--test-threads=1'])
report['rust_compiled']=True
for mode in ('0','1'):
    run([sys.executable,'-m','pytest','-q','ruda-torch/python/tests/test_v14_gpu.py','--maxfail=1'],RUDA_TORCH_ASYNC=mode)
report['gpu_validated']=True;report['scope']='single GPU on this host only; no full model or multi-GPU certification';finish(0)
