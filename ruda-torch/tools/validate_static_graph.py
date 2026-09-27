#!/usr/bin/env python3
"""Build and execute the native PyTorch static graph tests in fresh processes.

No CPU fallback, successful skips, missing-runtime-marker acceptance, or changes
of the global async default. Existing v22 lower-level validator stays unchanged.
"""
from __future__ import annotations
import argparse,ctypes.util,hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]
MARKER='RUDA_V24_STATIC_GRAPH_RUNTIME abi=9 graph_api=2'
EXPECTED=77
TOOLS=('memcheck','racecheck','initcheck','synccheck')

def check_report(xml:Path,text:str,code:int,tool=None):
    if code: raise ValueError(f'process exited {code}')
    cases=list(ET.parse(xml).getroot().iter('testcase'))
    if len(cases)!=EXPECTED or len({c.get('name') for c in cases})!=EXPECTED or any(c.find(tag) is not None for c in cases for tag in ('failure','error','skipped')):
        raise ValueError('expected exactly 77 tests, zero failures/errors/skips')
    if MARKER not in text:raise ValueError('no verified native runtime marker')
    if tool:
        errors=re.findall(r'ERROR SUMMARY:\s*(\d+) errors',text)
        races=re.findall(r'RACECHECK SUMMARY:\s*(\d+) hazards? displayed\s*\((\d+) errors?,\s*(\d+) warnings?\)',text)
        if any(int(v)!=0 for v in errors) or any(any(int(v)!=0 for v in group) for group in races):
            raise ValueError('nonzero sanitizer summary')
        if tool=='racecheck':
            if not races:raise ValueError('missing racecheck summary')
        elif not errors:raise ValueError('missing sanitizer summary')
    return len(cases)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('v24-static-graph-validation'))
    p.add_argument('--build',action='store_true')
    p.add_argument('--async-modes',choices=('both','0','1'),default='both')
    p.add_argument('--sanitizers',default='none',help='none, all, or comma-separated tools')
    p.add_argument('--benchmark',action='store_true')
    p.add_argument('--timeout',type=int,default=1800)
    a=p.parse_args(argv)
    tools=list(TOOLS) if a.sanitizers=='all' else ([] if a.sanitizers=='none' else a.sanitizers.split(','))
    if len(set(tools))!=len(tools) or any(t not in TOOLS for t in tools) or a.timeout<=0:p.error('invalid tools/timeout')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():p.error('use a fresh result directory; prior evidence is not overwritten')
    report={'version':'v24','new_feature':'native-static-graph-fusion-and-workspace','gpu_validated':False,
            'build_requested':a.build,'commands':[],'passed_test_executions':0,'errors':[]}
    def save():
        tmp=out/'result.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'result.json')
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'ruda-torch/python')+os.pathsep+env.get('PYTHONPATH','')
    env['RUDA_REQUIRE_GPU']='1'
    def run(name,cmd,cwd=ROOT,run_env=None):
        log=out/(name+'.log')
        with log.open('w') as f:
            ret=subprocess.run(cmd,cwd=cwd,env=run_env or env,stdout=f,stderr=subprocess.STDOUT,timeout=a.timeout)
        report['commands'].append({'name':name,'command':cmd,'exit_code':ret.returncode,'log':log.name});save()
        return ret.returncode,log.read_text(errors='replace')
    try:
        if env.get('RUDA_CUDA_COMPILER')!='ptx' or not env.get('RUDA_PTX_VERSION'):
            raise RuntimeError('set RUDA_CUDA_COMPILER=ptx and device-supported RUDA_PTX_VERSION')
        if not ctypes.util.find_library('cuda'):raise RuntimeError('NVIDIA driver library unavailable')
        if tools and not shutil.which('compute-sanitizer'):raise RuntimeError('compute-sanitizer unavailable')
        if a.build:
            if not shutil.which('cargo') or not shutil.which('rustc'):raise RuntimeError('Rust toolchain unavailable')
            code,_=run('rust-build',['cargo','build','--locked','--release','-p','ruda-torch-native'])
            if code:raise RuntimeError('native Rust build failed')
            code,_=run('cpp-build',[sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','-e','.'],ROOT/'ruda-torch/python')
            if code:raise RuntimeError('C++ bridge build failed')
            filename='ruda_torch_native.dll' if os.name=='nt' else 'libruda_torch_native.so'
            env['RUDA_TORCH_LIBRARY']=str(ROOT/'target/release'/filename)
        library=Path(env.get('RUDA_TORCH_LIBRARY','__missing__')).resolve()
        if not library.is_file():raise RuntimeError('set RUDA_TORCH_LIBRARY to the rebuilt native library, or use --build')
        report['native_library']={'path':str(library),'sha256':hashlib.sha256(library.read_bytes()).hexdigest()}
        report['environment']={k:env.get(k) for k in ('RUDA_CUDA_COMPILER','RUDA_PTX_VERSION','RUDA_TORCH_LIBRARY')}
        paths=['ruda-torch/src/static_graph.rs','ruda-torch/src/graph_contract.rs',
            'ruda-torch/python/ruda_torch/csrc/backend.cpp','ruda-torch/python/ruda_torch/csrc/static_graph.inc',
            'ruda-torch/python/ruda_torch/_graph.py','ruda-torch/python/ruda_torch/_graph_spec.py',
            'ruda-torch/python/ruda_torch/_graph_opt.py','ruda-torch/src/kernels.rs']
        report['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
        if shutil.which('nvidia-smi'):
            code,hardware=run('hardware',['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'])
            report['hardware']=hardware.strip()
            if code:raise RuntimeError('GPU probe failed')
        modes=('0','1') if a.async_modes=='both' else (a.async_modes,)
        for mode in modes:
            e=dict(env,RUDA_TORCH_ASYNC=mode)
            for tool in (None,*tools):
                name=f'async-{mode}-'+(tool or 'plain');xml=out/(name+'.xml')
                cmd=[sys.executable,'-m','pytest','-q','-s',str(ROOT/'ruda-torch/python/tests/test_static_graph_gpu.py'),f'--junitxml={xml}']
                if tool:cmd=['compute-sanitizer','--tool',tool,'--target-processes','all','--error-exitcode','86',*cmd]
                code,text=run(name,cmd,run_env=e)
                report['passed_test_executions']+=check_report(xml,text,code,tool);save()
            if a.benchmark:
                code,_=run(f'benchmark-async-{mode}',[sys.executable,str(ROOT/'ruda-torch/tools/benchmark_static_graph.py'),
                    '--output',str(out/f'benchmark-async-{mode}.json')],run_env=e)
                if code:raise RuntimeError('native graph benchmark failed')
                code,_=run(f'optimizer-benchmark-async-{mode}',
                    [sys.executable,str(ROOT/'ruda-torch/tools/benchmark_static_graph_optimized.py'),
                     '--output',str(out/f'optimizer-benchmark-async-{mode}.json')],run_env=e)
                if code:raise RuntimeError('optimized graph benchmark failed')
        report['gpu_validated']=True;save();return 0
    except (RuntimeError,OSError,ValueError,subprocess.TimeoutExpired,ET.ParseError) as exc:
        report['errors'].append(str(exc));save();print(str(exc),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
