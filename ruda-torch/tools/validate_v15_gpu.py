#!/usr/bin/env python3
"""Strict single-device PTX acceptance. Missing GPU/build tools, zero-test runs
and skipped hardware tests are failures. Never changes the async default.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def cargo_passed(text: str, minimum: int) -> int:
    matches=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)
    valid=[int(p) for p,f,i in matches if int(f)==0 and int(i)==0 and int(p)>=minimum]
    if not valid:
        raise ValueError(f'expected at least {minimum} real Rust tests, none failed/ignored; zero tests is not success')
    return max(valid)


def pytest_passed(path: Path, minimum: int) -> int:
    root=ET.parse(path).getroot();cases=list(root.iter('testcase'))
    if len(cases)<minimum or any(c.find(tag) is not None for c in cases for tag in ('failure','error','skipped')):
        raise ValueError('missing, failed, or skipped required PyTorch GPU cases')
    return len(cases)


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',action='store_true')
    p.add_argument('--output',type=Path,default=Path('v15-hardware-validation'))
    args=p.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    report={'version':'v15','abi':9,'gpu_validated':False,'rust_compiled':False,
            'default_async_changed':False,'platform':platform.platform(),'commands':[]}
    def save():
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        if not shutil.which('cargo'):
            raise RuntimeError('Rust/Cargo is required and was not found')
        if not os.environ.get('RUDA_PTX_VERSION'):
            raise RuntimeError('Set RUDA_PTX_VERSION for the actual driver')
        driver=ctypes.CDLL('libcuda.so.1')
        driver.cuInit.argtypes=[ctypes.c_uint];driver.cuInit.restype=ctypes.c_int
        driver.cuDeviceGetCount.argtypes=[ctypes.POINTER(ctypes.c_int)];driver.cuDeviceGetCount.restype=ctypes.c_int
        status=driver.cuInit(0);count=ctypes.c_int()
        if status!=0 or driver.cuDeviceGetCount(ctypes.byref(count))!=0 or count.value<1:
            raise RuntimeError(f'no usable NVIDIA device (cuInit status={status})')
        report['device_count']=count.value
        env=dict(os.environ,RUDA_REQUIRE_GPU='1',RUDA_CUDA_COMPILER='ptx')
        env.setdefault('RUDA_TORCH_LIBRARY',str(root/'target/release/libruda_torch_native.so'))
        def run(cmd,extra=None):
            log=output/f'{len(report["commands"]):02d}.log'
            with log.open('w') as f:
                result=subprocess.run(cmd,cwd=root,env=dict(env,**(extra or {})),stdout=f,stderr=subprocess.STDOUT)
            report['commands'].append({'command':cmd,'returncode':result.returncode,'log':log.name,'env':extra or {}})
            save()
            if result.returncode:raise RuntimeError(f'required command failed: {log.name}')
            return log.read_text(errors='replace')
        if args.build:
            run(['cargo','build','-p','ruda-torch-native','--release']);report['rust_compiled']=True
            run([sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','-e','ruda-torch/python'])
        for prefix,count in [('v14_gpu_',4),('v15_gpu_',6)]:
            text=run(['cargo','test','-p','ruda-torch-native','--release',prefix,'--','--ignored','--test-threads=1'])
            report[prefix+'passed']=cargo_passed(text,count);report['rust_compiled']=True
        for mode in ('0','1'):
            junit=output/f'gpu-async-{mode}.xml'
            run([sys.executable,'-m','pytest','-q','ruda-torch/python/tests/test_v14_gpu.py',
                 'ruda-torch/python/tests/test_v15_gpu.py','--maxfail=1',f'--junitxml={junit}'],{'RUDA_TORCH_ASYNC':mode})
            report[f'python_async_{mode}_passed']=pytest_passed(junit,61)
        report['gpu_validated']=True
        report['scope']='single NVIDIA device with these binaries/driver only; not full-model or multi-GPU certification'
        save();print(json.dumps(report,indent=2));return 0
    except (OSError,RuntimeError,ValueError,ET.ParseError) as error:
        report['error']=str(error);save();print(json.dumps(report,indent=2));return 2

if __name__=='__main__':
    raise SystemExit(main())
