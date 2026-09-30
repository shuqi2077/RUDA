#!/usr/bin/env python3
"""Strict v32 native paged training acceptance; failure/skip is never success.

No GPU emulation or CPU implementation fallback. The reference is CPU only.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import xml.etree.ElementTree as ET
from validate_router import check_sanitizer
ROOT=Path(__file__).resolve().parents[2]
MARKER='RUDA_V32_PAGED_GPU_EXECUTED'
TEST='tests/test_paged_selected_gpu.py'
TOOLS=('memcheck','racecheck','initcheck','synccheck')

def collect_names(text):
    names=[x.strip() for x in text.splitlines() if x.strip().startswith(TEST+'::test_')]
    if not names or len(names)!=len(set(names)):raise ValueError('missing/duplicate GPU test collection')
    return names

def check_result(xml,text,code,names,tool=None):
    if code:raise ValueError('GPU process exited '+str(code))
    tests=list(ET.parse(xml).getroot().iter('testcase'))
    identifiers=[(x.get('classname'),x.get('name')) for x in tests]
    expected={x.split('::')[-1] for x in names}
    if not names or len(tests)!=len(names) or len(set(identifiers))!=len(names) or {x.get('name') for x in tests}!=expected:
        raise ValueError('wrong/duplicate test results')
    if any(x.find(k) is not None for x in tests for k in ('failure','error','skipped')):
        raise ValueError('failed, errored or skipped GPU case')
    if MARKER not in text:raise ValueError('missing real native GPU execution marker')
    check_sanitizer(text,tool)
    return len(tests)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',action='store_true')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dtypes',default='float32,float16')
    p.add_argument('--sanitizers',default='none');p.add_argument('--timeout',type=int,default=3600)
    args=p.parse_args(argv);dtypes=args.dtypes.split(',')
    tools=list(TOOLS) if args.sanitizers=='all' else ([] if args.sanitizers=='none' else args.sanitizers.split(','))
    if len(dtypes)!=len(set(dtypes)) or any(x not in ('float32','float16','bfloat16') for x in dtypes):p.error('invalid dtypes')
    if not dtypes or len(tools)!=len(set(tools)) or any(x not in TOOLS for x in tools) or args.timeout<=0:p.error('invalid options')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True);dest=output/'summary.json'
    if dest.exists():p.error('use a fresh result directory; do not overwrite evidence')
    report={'version':'v32','gpu_validated':False,'commands':[],'runs':[],'errors':[],
            'dtypes':dtypes,'scope':'native paged derivatives and FP32 single-block training, not full LLM convergence'}
    env=dict(os.environ,RUDA_REQUIRE_GPU='1',RUDA_PAGED_DTYPES=args.dtypes)
    env['PYTHONPATH']=str(ROOT/'ruda-torch/python')+os.pathsep+env.get('PYTHONPATH','')
    def save():dest.write_text(json.dumps(report,indent=2)+'\n')
    def run(label,cmd,*,cwd=ROOT,overrides=None):
        path=output/(label+'.log')
        with path.open('w') as stream:
            try:code=subprocess.run(cmd,cwd=cwd,env=dict(env,**(overrides or {})),stdout=stream,stderr=subprocess.STDOUT,timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:code=124
        text=path.read_text(errors='replace')
        report['commands'].append({'command':cmd,'cwd':str(cwd),'exit_code':code,'log':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});save()
        return code,text
    try:
        if args.build:
            missing=[x for x in ('cargo','rustc') if shutil.which(x) is None]
            if missing:raise RuntimeError('missing build toolchain: '+','.join(missing))
        if env.get('RUDA_CUDA_COMPILER')!='ptx' or not env.get('RUDA_PTX_VERSION'):
            raise RuntimeError('set direct PTX compiler and tested RUDA_PTX_VERSION')
        from validate_training import probe_gpu
        report['hardware']=probe_gpu()
        if 'bfloat16' in dtypes and report['hardware']['compute_capability'][0]<8:
            raise RuntimeError('BF16 requires supported device; not a successful skip')
        if tools and not shutil.which('compute-sanitizer'):raise RuntimeError('compute-sanitizer unavailable')
        if args.build:
            code,_=run('rust-build',['cargo','build','--locked','--release','-p','ruda-torch-native'])
            if code:raise RuntimeError('native Rust build failed')
            target=Path(env.get('CARGO_TARGET_DIR',str(ROOT/'target')))
            if not target.is_absolute():target=ROOT/target
            filename='ruda_torch_native.dll' if sys.platform=='win32' else 'libruda_torch_native.so'
            env['RUDA_TORCH_LIBRARY']=str(target/'release'/filename)
            code,_=run('cpp-build',[sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','-e','.'],cwd=ROOT/'ruda-torch/python')
            if code:raise RuntimeError('C++ build failed')
        lib=Path(env.get('RUDA_TORCH_LIBRARY','__missing__'))
        if not lib.is_file():raise RuntimeError('set RUDA_TORCH_LIBRARY or pass --build')
        report['native_library_sha256']=hashlib.sha256(lib.read_bytes()).hexdigest()
        report['source_sha256']={}
        for folder in ['ruDNN/src/paged_attention','ruda-torch/src','ruda-torch/python/ruda_torch','ruda-torch/python/tests','ruda-torch/python/examples']:
            for file in sorted((ROOT/folder).rglob('*')):
                if file.is_file() and file.suffix in ('.rs','.py','.cpp','.inc'):
                    report['source_sha256'][str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
        cwd=ROOT/'ruda-torch/python'
        code,text=run('collect',[sys.executable,'-m','pytest','--collect-only','-q',TEST],cwd=cwd)
        if code:raise RuntimeError('GPU collection failed')
        names=collect_names(text);report['test_count_per_run']=len(names);save()
        for mode in ('0','1'):
            for tool in [None,*tools]:
                label='gpu-'+mode+'-'+(tool or 'plain');xml=output/(label+'.xml')
                prefix=[] if tool is None else ['compute-sanitizer','--tool',tool,'--target-processes','all','--error-exitcode','97']
                cmd=prefix+[sys.executable,'-m','pytest','-q','-s','--junitxml='+str(xml),TEST]
                code,text=run(label,cmd,cwd=cwd,overrides={'RUDA_TORCH_ASYNC':mode})
                count=check_result(xml,text,code,names,tool)
                report['runs'].append({'async':mode,'sanitizer':tool,'passed':count});save()
        report['gpu_validated']=True;save();return 0
    except Exception as exc:
        report['errors'].append(str(exc));save();print('REFUSED/FAILED:',exc,file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
