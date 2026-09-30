#!/usr/bin/env python3
"""Strict, resumable native training acceptance. CPU references never count as GPU passes."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
MARKER='RUDA_V28_TRAINING_GPU_EXECUTED'
TOOLS=('memcheck','racecheck','initcheck','synccheck')


def check_report(path,text,code,expected,tool=None):
    if code:raise ValueError(f'test process exited {code}')
    cases=list(ET.parse(path).getroot().iter('testcase'))
    keys={(c.get('classname'),c.get('name')) for c in cases}
    if len(cases)!=expected or len(keys)!=expected:
        raise ValueError(f'expected {expected} unique tests, got {len(cases)}')
    if any(c.find(tag) is not None for c in cases for tag in ('failure','error','skipped')):
        raise ValueError('failed, errored or skipped GPU tests')
    if MARKER not in text:raise ValueError('missing real GPU memory/execution marker')
    if tool:
        errors=re.findall(r'ERROR SUMMARY:\s*(\d+) errors',text)
        races=re.findall(r'RACECHECK SUMMARY:\s*(\d+) hazards? displayed\s*\((\d+) errors?,\s*(\d+) warnings?\)',text)
        if any(int(e) for e in errors) or any(any(int(x) for x in r) for r in races):
            raise ValueError('sanitizer reported errors or hazards')
        if tool=='racecheck' and not races:raise ValueError('missing racecheck summary')
        if tool!='racecheck' and not errors:raise ValueError('missing sanitizer summary')
    return len(cases)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_gpu():
    d=ctypes.CDLL('libcuda.so.1')
    def call(fn,*args):
        code=getattr(d,fn)(*args)
        if code:raise RuntimeError(f'{fn} returned {code}')
    call('cuInit',0)
    device=ctypes.c_int();call('cuDeviceGet',ctypes.byref(device),0)
    major=ctypes.c_int();minor=ctypes.c_int();version=ctypes.c_int()
    call('cuDeviceGetAttribute',ctypes.byref(major),75,device)
    call('cuDeviceGetAttribute',ctypes.byref(minor),76,device)
    call('cuDriverGetVersion',ctypes.byref(version))
    uuid=(ctypes.c_ubyte*16)();call('cuDeviceGetUuid',ctypes.byref(uuid),device)
    return {'uuid':bytes(uuid).hex(),'compute_capability':[major.value,minor.value],'driver_api':version.value}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--build',action='store_true')
    p.add_argument('--dtypes',default='float32,float16');p.add_argument('--modes',choices=['both','0','1'],default='both')
    p.add_argument('--sanitizers',default='none');p.add_argument('--resume',action='store_true')
    p.add_argument('--timeout',type=int,default=1800);p.add_argument('--max-runs',type=int,default=0)
    a=p.parse_args(argv);names=a.dtypes.split(',')
    tools=list(TOOLS) if a.sanitizers=='all' else ([] if a.sanitizers=='none' else a.sanitizers.split(','))
    if not names or len(names)!=len(set(names)) or any(n not in ('float32','float16','bfloat16') for n in names):p.error('invalid dtypes')
    if len(set(tools))!=len(tools) or any(t not in TOOLS for t in tools) or a.timeout<=0 or a.max_runs<0:p.error('invalid tools/timeout/max-runs')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);target=out/'summary.json'
    if target.exists() and not a.resume:p.error('use --resume or a fresh output directory')
    report={'version':'v28','gpu_validated':False,'runs':{},'commands':[],'errors':[],
            'dtype_scope':names,'cpu_reference_is_gpu_evidence':False}
    prior=json.loads(target.read_text()) if target.exists() else None
    canonical=target
    # Build/import logs must not overwrite prior evidence before identity is checked.
    if prior is not None:
        target=out/f'resume-attempt-{time.time_ns()}.json'
    env=dict(os.environ,RUDA_CUDA_COMPILER='ptx',RUDA_REQUIRE_GPU='1',RUDA_TRAINING_DTYPES=a.dtypes)
    env['PYTHONPATH']=str(ROOT/'ruda-torch/python')+os.pathsep+env.get('PYTHONPATH','')
    def save():
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(target)
    def run(name,cmd,e=env,cwd=ROOT):
        stamp=str(time.time_ns());log=out/f'{name}-{stamp}.log'
        with log.open('w') as f:result=subprocess.run(cmd,cwd=cwd,env=e,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout)
        report['commands'].append({'command':cmd,'exit_code':result.returncode,'log':log.name});save()
        return result.returncode,log
    try:
        if os.environ.get('RUDA_CUDA_COMPILER')!='ptx' or not env.get('RUDA_PTX_VERSION'):
            raise RuntimeError('set RUDA_CUDA_COMPILER=ptx and a driver-supported RUDA_PTX_VERSION')
        if a.build and (not shutil.which('cargo') or not shutil.which('rustc')):
            raise RuntimeError('Rust/Cargo toolchain unavailable')
        hardware=probe_gpu()
        if 'bfloat16' in names and hardware['compute_capability'][0]<8:
            raise RuntimeError('BF16 was explicitly requested on a pre-Ampere device; no successful skips')
        if tools and not shutil.which('compute-sanitizer'):raise RuntimeError('compute-sanitizer unavailable')
        if a.build:
            env.setdefault('CARGO_BUILD_JOBS','1')
            code,_=run('rust-build',['cargo','build','--locked','--release','-p','ruda-torch-native'])
            if code:raise RuntimeError('Rust build failed')
            code,_=run('cpp-build',[sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','-e','.'],cwd=ROOT/'ruda-torch/python')
            if code:raise RuntimeError('C++ extension build failed')
            target_dir=Path(env.get('CARGO_TARGET_DIR',ROOT/'target'))
            if not target_dir.is_absolute():target_dir=ROOT/target_dir
            env['RUDA_TORCH_LIBRARY']=str(target_dir/'release/libruda_torch_native.so')
        library=Path(env.get('RUDA_TORCH_LIBRARY','__missing__'))
        if not library.is_file():raise RuntimeError('set RUDA_TORCH_LIBRARY or use --build')
        source={}
        for folder in ('ruda-torch/src','ruda-torch/python/ruda_torch','ruda-optim/src/fused_adamw'):
            for file in sorted((ROOT/folder).rglob('*')):
                if file.is_file() and file.suffix in ('.rs','.py','.cpp','.inc'):
                    source[str(file.relative_to(ROOT))]=digest(file)
        source['Cargo.lock']=digest(ROOT/'Cargo.lock')
        source['tests']=digest(ROOT/'ruda-torch/python/tests/test_training_gpu.py')
        source['fused_tests']=digest(ROOT/'ruda-torch/python/tests/test_optimizer_fused_gpu.py')
        source['hierarchy_tests']=digest(ROOT/'ruda-torch/python/tests/test_gradient_stats_hierarchy_gpu.py')
        source['validator']=digest(Path(__file__))
        code,log=run('native-import',[sys.executable,'-c',
            "import json,ruda_torch as r,torch;assert r._training_available and r._C.training_api_version==4;print(json.dumps({'torch':torch.__version__,'cpp':r._C.__file__}))"])
        if code:raise RuntimeError('native training API import failed')
        imported=json.loads(log.read_text().strip().splitlines()[-1])
        identity={'hardware':hardware,'source':source,'rust_library':digest(library),'cpp_library':digest(imported['cpp']),
                  'torch':imported['torch'],'ptx':env['RUDA_PTX_VERSION'],'dtypes':names,'modes':a.modes,'tools':tools}
        report['identity']=identity
        if prior is not None:
            if prior.get('identity')!=identity:raise RuntimeError('resume identity differs: do not mix source, device or binary results')
            report['runs']=prior.get('runs',{});report['commands']=prior.get('commands',[])+report['commands']
            target=canonical
        modes=['0','1'] if a.modes=='both' else [a.modes];expected=47*len(names)+13;executed=0
        tasks=[(mode,tool) for mode in modes for tool in [None,*tools]]
        for mode,tool in tasks:
            key=mode+'-'+(tool or 'plain');old=report['runs'].get(key)
            if old and old.get('validated'):
                xml=out/old['xml'];textfile=out/old['log']
                try:
                    if digest(xml)!=old['xml_sha256'] or digest(textfile)!=old['log_sha256']:raise ValueError('changed logs')
                    check_report(xml,textfile.read_text(errors='replace'),old['exit_code'],expected,tool)
                    continue
                except (ValueError,OSError,ET.ParseError):pass
            if a.max_runs and executed>=a.max_runs:
                report['incomplete']=True;save();return 3
            stamp=str(time.time_ns());xml=out/f'{key}-{stamp}.xml'
            cmd=[sys.executable,'-m','pytest','-q','-s',str(ROOT/'ruda-torch/python/tests/test_training_gpu.py'),str(ROOT/'ruda-torch/python/tests/test_optimizer_fused_gpu.py'),str(ROOT/'ruda-torch/python/tests/test_gradient_stats_hierarchy_gpu.py'),f'--junitxml={xml}']
            if tool:cmd=['compute-sanitizer','--tool',tool,'--target-processes','all','--error-exitcode','86',*cmd]
            code,log=run(key,cmd,dict(env,RUDA_TORCH_ASYNC=mode));executed+=1
            item={'validated':False,'exit_code':code,'xml':xml.name,'log':log.name}
            report['runs'][key]=item;save()
            count=check_report(xml,log.read_text(errors='replace'),code,expected,tool)
            item.update(validated=True,passed=count,xml_sha256=digest(xml),log_sha256=digest(log));save()
        report['gpu_validated']=True;report['incomplete']=False
        report['passed_test_executions']=sum(v['passed'] for v in report['runs'].values() if v.get('validated'))
        save();return 0
    except (OSError,RuntimeError,ValueError,ET.ParseError,subprocess.TimeoutExpired) as e:
        # Preserve older successful evidence if a resume probe itself failed.
        if prior is not None and target != canonical:report['prior_evidence_preserved']=True
        report['errors'].append(str(e));save();print(str(e),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
