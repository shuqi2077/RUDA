#!/usr/bin/env python3
"""Build and execute v31 production grouped/expert backward tests on CUDA/PTX.

No CPU substitute, ignored test or missing marker can satisfy GPU acceptance.
Compilers are outside sanitizer; each sanitizer runs the real test executable.
"""
from __future__ import annotations
import argparse,ctypes,hashlib,json,math,os,re,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
TOOLS=('memcheck','racecheck','initcheck','synccheck')
GROUPED_NAMES={
 'v31_runtime_marker','v31_scalar_reference_fp32','v31_cooperative_one_element',
 'v31_cooperative_aligned','v31_cooperative_all_tails','v31_cooperative_empty_experts',
 'v31_cooperative_all_empty','v31_cooperative_unbalanced','v31_cooperative_noncontiguous',
 'v31_auto_fp32_uses_scalar','v31_auto_fp16_matches_forced','v31_cooperative_repeatable',
 'v31_forced_fp32_rejected'}
EXPERT_NAMES={'v31_runtime_marker','v31_swiglu_fp32','v31_swiglu_fp16_rounding',
              'v31_swiglu_empty','v31_training_uses_selected_backward_strategy',
              'v31_forward_new_output_matches_legacy_inplace'}
SCOPES=[('rublas','rublas','tensor-grouped','tensor_grouped::tests_backward::v31_',GROUPED_NAMES,
         'RUDA_V31_GROUPED_GPU_EXECUTED=', 'bf16_grouped_v31'),
        ('ruDNN','rudnn','tensor-moe','moe::tests_v31::v31_',EXPERT_NAMES,
         'RUDA_V31_EXPERT_GPU_EXECUTED=', 'bf16_expert_v31')]

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def test_names(text):
    names=re.findall(r'^([\w:]+): test$',text,re.M)
    if not names or len(names)!=len(set(names)):raise ValueError('missing/duplicate test list')
    return names

def check_sanitizer(text,tool):
    if tool is None:return
    summaries=re.findall(r'ERROR SUMMARY:\s*(\d+) errors?',text)
    races=re.findall(r'RACECHECK SUMMARY:\s*(\d+) hazards? displayed\s*\((\d+) errors?,\s*(\d+) warnings?\)',text)
    if any(int(x) for x in summaries) or any(any(int(x) for x in r) for r in races):
        raise ValueError('sanitizer reported errors/hazards/warnings')
    if tool=='racecheck' and not races:raise ValueError('missing racecheck summary')
    if tool!='racecheck' and not summaries:raise ValueError('missing sanitizer summary')

def check_rust_report(text,code,expected,marker=None,tool=None):
    if code:raise ValueError(f'Rust test process exited {code}')
    matches=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)
    if expected<=0 or len(matches)!=1 or tuple(map(int,matches[0]))!=(expected,0,0):
        raise ValueError('wrong executed test count or ignored/failed cases')
    if marker is not None and marker not in text:raise ValueError('missing production CUDA marker')
    check_sanitizer(text,tool)
    return expected

def executable(text,target):
    found=[]
    for line in text.splitlines():
        try:d=json.loads(line)
        except json.JSONDecodeError:continue
        if d.get('reason')=='compiler-artifact' and d.get('profile',{}).get('test') and d.get('target',{}).get('name')==target and d.get('executable'):
            found.append(d['executable'])
    if len(set(found))!=1:raise ValueError('expected exactly one production test binary')
    return found[0]

def gpu_info():
    driver=ctypes.CDLL('libcuda.so.1')
    def call(name,*args):
        code=getattr(driver,name)(*args)
        if code:raise RuntimeError(f'{name} failed: {code}')
    call('cuInit',0);dev=ctypes.c_int();call('cuDeviceGet',ctypes.byref(dev),0)
    major=ctypes.c_int();minor=ctypes.c_int();version=ctypes.c_int()
    call('cuDeviceGetAttribute',ctypes.byref(major),75,dev)
    call('cuDeviceGetAttribute',ctypes.byref(minor),76,dev)
    call('cuDriverGetVersion',ctypes.byref(version))
    uuid=(ctypes.c_ubyte*16)();call('cuDeviceGetUuid',ctypes.byref(uuid),dev)
    return {'uuid':bytes(uuid).hex(),'compute_capability':[major.value,minor.value],'driver_api':version.value}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sanitizers',default='none');p.add_argument('--bf16',action='store_true')
    p.add_argument('--build-native',action='store_true',help='also compile full ruda-torch-native (no C++ change in v31)')
    p.add_argument('--benchmark',action='store_true');p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args(argv);tools=list(TOOLS) if a.sanitizers=='all' else ([] if a.sanitizers=='none' else a.sanitizers.split(','))
    if a.timeout<=0 or len(set(tools))!=len(tools) or any(t not in TOOLS for t in tools):p.error('invalid tools/timeout')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    target=out/'summary.json'
    if target.exists():p.error('choose a fresh output directory to preserve evidence')
    report={'version':'v31','gpu_validated':False,'rust_compiled':False,'bf16_requested':a.bf16,
            'benchmarked':False,'commands':[],'runs':[],'errors':[], 'scope':'public grouped/expert backward; not a complete model'}
    env=dict(os.environ,RUDA_REQUIRE_GPU='1');env.setdefault('CARGO_BUILD_JOBS','1')
    def save():target.write_text(json.dumps(report,indent=2)+'\n')
    def run(label,cmd):
        log=out/(label+'.log');code=2
        try:
            with log.open('w') as f:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout)
            code=r.returncode
        except subprocess.TimeoutExpired:code=124
        except OSError as exc:log.write_text(str(exc)+'\n')
        text=log.read_text(errors='replace')
        report['commands'].append({'command':cmd,'exit_code':code,'log':log.name,'sha256':digest(log)});save()
        return code,text
    try:
        missing=[t for t in ('rustc','cargo') if not shutil.which(t)]
        if missing:raise RuntimeError('missing toolchain: '+','.join(missing))
        if env.get('RUDA_CUDA_COMPILER')!='ptx' or not env.get('RUDA_PTX_VERSION'):
            raise RuntimeError('set RUDA_CUDA_COMPILER=ptx and a driver-supported RUDA_PTX_VERSION')
        report['hardware']=gpu_info()
        if a.bf16 and report['hardware']['compute_capability'][0]<8:raise RuntimeError('BF16 explicitly requested on unsupported device; no successful skip')
        if tools and not shutil.which('compute-sanitizer'):raise RuntimeError('compute-sanitizer unavailable')
        report['source_sha256']={}
        for folder in ['ruBLAS/src/tensor_grouped','ruDNN/src/moe']:
            for path in sorted((ROOT/folder).rglob('*.rs')):report['source_sha256'][str(path.relative_to(ROOT))]=digest(path)
        report['cargo_lock_sha256']=digest(ROOT/'Cargo.lock')
        for tool in ('rustc','cargo'):
            code,text=run(tool+'-version',[tool,'--version'])
            if code:raise RuntimeError(tool+' version failed')
        if a.build_native:
            code,_=run('native-build',['cargo','build','--locked','--release','-p','ruda-torch-native'])
            if code:raise RuntimeError('full native build failed')
        for package,name,feature,prefix,names,marker,bf16_name in SCOPES:
            code,text=run(name+'-build',['cargo','test','--locked','--release','-p',package,'--lib',
                '--features',feature+',ruda-test-runtime/cuda','--no-run','--message-format=json'])
            if code:raise RuntimeError(package+' test build failed')
            binary=executable(text,name)
            report.setdefault('binary_sha256',{})[name]=digest(binary)
            code,text=run(name+'-list',[binary,prefix,'--list'])
            if code or {n.rsplit('::',1)[-1] for n in test_names(text)}!=names:raise RuntimeError('unexpected '+package+' test list')
            for tool in [None,*tools]:
                launch=[] if tool is None else ['compute-sanitizer','--tool',tool,'--error-exitcode','97']
                code,text=run(name+'-'+(tool or 'plain'),launch+[binary,prefix,'--nocapture','--test-threads=1'])
                count=check_rust_report(text,code,len(names),marker,tool)
                report['runs'].append({'scope':name,'tool':tool,'passed':count,'bf16':False});save()
                if a.bf16:
                    code,text=run(name+'-bf16-'+(tool or 'plain'),launch+[binary,bf16_name,'--nocapture','--test-threads=1'])
                    count=check_rust_report(text,code,1,None,tool)
                    report['runs'].append({'scope':name,'tool':tool,'passed':count,'bf16':True});save()
            if a.benchmark and package=='rublas':
                code,text=run('benchmark',[binary,'benchmark_grouped_backward_v31','--ignored','--nocapture','--test-threads=1'])
                check_rust_report(text,code,1)
                samples=re.findall(r'RUDA_V31_BENCH .*?api_with_readback_us=([\d.]+)',text)
                if len(samples)!=42 or any(not math.isfinite(float(x)) or float(x)<=0 for x in samples):raise RuntimeError('incomplete paired benchmark')
                report['benchmarked']=True
        report['rust_compiled']=True;report['gpu_validated']=True;save();return 0
    except Exception as exc:
        report['errors'].append(str(exc));save();print('REFUSED/FAILED:',exc,file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
