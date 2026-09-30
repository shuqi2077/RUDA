#!/usr/bin/env python3
"""Strict v30 router/MoE acceptance. Runs actual Rust binaries and native PyTorch.

Builds are never wrapped in a sanitizer. Missing hardware, zero executed tests,
missing runtime markers, ignored/skipped tests and sanitizer errors all fail.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
TOOLS=('memcheck','racecheck','initcheck','synccheck')
PY_MARKER='RUDA_V30_ROUTER_GPU_EXECUTED'
RUST_MARKER='RUDA_V30_MOE_GPU_EXECUTED='


def check_sanitizer(text, tool):
    if tool is None:return
    errors=re.findall(r'ERROR SUMMARY:\s*(\d+) errors',text)
    races=re.findall(r'RACECHECK SUMMARY:\s*(\d+) hazards? displayed\s*\((\d+) errors?,\s*(\d+) warnings?\)',text)
    if any(int(x) for x in errors) or any(any(int(x) for x in row) for row in races):
        raise ValueError('sanitizer errors/warnings/hazards')
    if tool=='racecheck' and not races:raise ValueError('missing racecheck summary')
    if tool!='racecheck' and not errors:raise ValueError('missing sanitizer summary')


def check_python_report(xml, text, code, expected, tool=None):
    if code:raise ValueError(f'Python GPU process exited {code}')
    tests=list(ET.parse(xml).getroot().iter('testcase'))
    keys={(t.get('classname'),t.get('name')) for t in tests}
    if expected<=0 or len(tests)!=expected or len(keys)!=expected:
        raise ValueError('wrong or duplicate Python GPU test count')
    if any(t.find(tag) is not None for t in tests for tag in ('failure','error','skipped')):
        raise ValueError('GPU tests failed/errored/skipped')
    if PY_MARKER not in text:raise ValueError('missing native router execution marker')
    check_sanitizer(text,tool)
    return expected


def check_rust_report(text,code,tool=None):
    if code:raise ValueError(f'Rust GPU process exited {code}')
    result=re.search(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)
    if not result or tuple(map(int,result.groups()))!=(12,0,0) or RUST_MARKER not in text:
        raise ValueError('expected 12 executed Rust GPU tests and the CUDA runtime marker')
    check_sanitizer(text,tool)
    return 12


def collect_names(text):
    names=[line.strip() for line in text.splitlines()
           if 'test_router_gpu.py::test_' in line and ' ' not in line.strip()]
    if not names or len(names)!=len(set(names)):
        raise ValueError('missing or duplicate collected GPU test names')
    return names


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--build',action='store_true',help='rebuild Rust library and native C++ extension')
    parser.add_argument('--dtypes',default='float32,float16')
    parser.add_argument('--sanitizers',default='none',help='none, all, or comma-separated tools')
    parser.add_argument('--timeout',type=int,default=3600)
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args(argv)
    names=args.dtypes.split(',')
    tools=list(TOOLS) if args.sanitizers=='all' else ([] if args.sanitizers=='none' else args.sanitizers.split(','))
    if len(names)!=len(set(names)) or any(n not in ('float32','float16','bfloat16') for n in names):parser.error('invalid dtypes')
    if len(tools)!=len(set(tools)) or any(t not in TOOLS for t in tools) or args.timeout<=0:parser.error('invalid tools/timeout')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    summary=out/'summary.json'
    if summary.exists():parser.error('choose a fresh output directory; do not overwrite prior evidence')
    report={'version':'v30','gpu_validated':False,'dtype_scope':names,'commands':[],'runs':[],
            'cpu_reference_is_gpu_evidence':False,'errors':[]}
    env=dict(os.environ,RUDA_REQUIRE_GPU='1',RUDA_ROUTER_DTYPES=args.dtypes)
    env['PYTHONPATH']=str(ROOT/'ruda-torch/python')+os.pathsep+env.get('PYTHONPATH','')
    def save():summary.write_text(json.dumps(report,indent=2)+'\n')
    def run(label,cmd,overrides=None,cwd=ROOT):
        log=out/(label+'.log');e=dict(env,**(overrides or {}))
        try:
            with log.open('w') as stream:
                result=subprocess.run(cmd,cwd=cwd,env=e,stdout=stream,stderr=subprocess.STDOUT,timeout=args.timeout)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
        report['commands'].append({'command':cmd,'cwd':str(cwd),'exit_code':code,'log':log.name,
                                   'sha256':hashlib.sha256(log.read_bytes()).hexdigest()});save()
        return code,log.read_text(errors='replace')
    try:
        missing=[tool for tool in ('cargo','rustc') if not shutil.which(tool)]
        if missing:raise RuntimeError('missing toolchain: '+','.join(missing))
        if env.get('RUDA_CUDA_COMPILER')!='ptx' or not env.get('RUDA_PTX_VERSION'):
            raise RuntimeError('set RUDA_CUDA_COMPILER=ptx and a driver-supported RUDA_PTX_VERSION')
        from validate_training import probe_gpu
        report['hardware']=probe_gpu()
        if 'bfloat16' in names and report['hardware']['compute_capability'][0]<8:
            raise RuntimeError('BF16 requested on pre-Ampere hardware; no successful skip')
        if tools and not shutil.which('compute-sanitizer'):raise RuntimeError('compute-sanitizer unavailable')
        if args.build:
            code,_=run('rust-build',['cargo','build','--locked','--release','-p','ruda-torch-native'])
            if code:raise RuntimeError('native Rust build failed')
            target=Path(env.get('CARGO_TARGET_DIR',str(ROOT/'target')))
            if not target.is_absolute():target=ROOT/target
            env['RUDA_TORCH_LIBRARY']=str(target/'release/libruda_torch_native.so')
            code,_=run('cpp-build',[sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','-e','.'],cwd=ROOT/'ruda-torch/python')
            if code:raise RuntimeError('C++ extension build failed')
        library=Path(env.get('RUDA_TORCH_LIBRARY','__missing__'))
        if not library.is_file():raise RuntimeError('set RUDA_TORCH_LIBRARY or pass --build')
        report['native_library_sha256']=hashlib.sha256(library.read_bytes()).hexdigest()
        report['source_sha256']={}
        for folder in ('ruDNN/src/moe','ruda-torch/src','ruda-torch/python/ruda_torch'):
            for path in sorted((ROOT/folder).rglob('*')):
                if path.is_file() and path.suffix in ('.rs','.cpp','.inc','.py'):
                    report['source_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        code,text=run('rust-device-test-build',['cargo','test','--locked','--release','-p','ruDNN','--lib',
            '--features','tensor-moe,ruda-test-runtime/cuda','--no-run','--message-format=json'])
        if code:raise RuntimeError('Rust MoE test build failed')
        binaries=[]
        for line in text.splitlines():
            try:obj=json.loads(line)
            except json.JSONDecodeError:continue
            if obj.get('reason')=='compiler-artifact' and obj.get('profile',{}).get('test') and obj.get('executable'):
                if obj.get('target',{}).get('name')=='rudnn':binaries.append(obj['executable'])
        if len(set(binaries))!=1:raise RuntimeError('cannot identify exactly one ruDNN unit-test binary')
        binary=binaries[0];report['rust_test_binary_sha256']=hashlib.sha256(Path(binary).read_bytes()).hexdigest()
        for tool in [None,*tools]:
            prefix=[] if tool is None else ['compute-sanitizer','--tool',tool,'--error-exitcode','97']
            cmd=prefix+[binary,'moe::tests::v30_','--test-threads=1','--nocapture']
            code,text=run('rust-'+(tool or 'plain'),cmd)
            count=check_rust_report(text,code,tool);report['runs'].append({'scope':'rust','tool':tool,'passed':count});save()
        file='tests/test_router_gpu.py';cwd=ROOT/'ruda-torch/python'
        code,text=run('python-collect',[sys.executable,'-m','pytest','--collect-only','-q',file],cwd=cwd)
        if code:raise RuntimeError('GPU test collection failed')
        expected=len(collect_names(text));report['python_tests_per_mode']=expected
        for mode in ('0','1'):
            for tool in [None,*tools]:
                label='python-'+mode+'-'+(tool or 'plain');xml=out/(label+'.xml')
                prefix=[] if tool is None else ['compute-sanitizer','--tool',tool,'--target-processes','all','--error-exitcode','97']
                cmd=prefix+[sys.executable,'-m','pytest','-q','-s','--junitxml='+str(xml),file]
                code,text=run(label,cmd,{'RUDA_TORCH_ASYNC':mode},cwd)
                count=check_python_report(xml,text,code,expected,tool)
                report['runs'].append({'scope':'python','async':mode,'tool':tool,'passed':count});save()
        if args.benchmark:
            code,_=run('benchmark',[sys.executable,str(ROOT/'ruda-torch/python/examples/benchmark_router.py'),
                                   '--output',str(out/'benchmark.json')])
            if code:raise RuntimeError('router benchmark failed')
        report['gpu_validated']=True;save();return 0
    except Exception as error:
        report['errors'].append(str(error));save();print('REFUSED/FAILED:',error,file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
