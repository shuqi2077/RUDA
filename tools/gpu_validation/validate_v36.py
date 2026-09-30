#!/usr/bin/env python3
"""Strict v36 physical-page compaction Rust + native PyTorch GPU acceptance.

Requires the real CUDA/PTX backend, not host emulation. Builds the test binary
outside sanitizer and checks exact case names and count. --build also rebuilds
and tests the PyTorch bridge. The explicit benchmark includes gradient readback. --build also runs the
existing 150-case ordered/pruning PyTorch suite with COMPACT_HISTORY=1. This is not
full-model training certification.
"""
from __future__ import annotations
import argparse,json,os,sys,shutil,subprocess,hashlib,re,math
from pathlib import Path
from validate_v31 import gpu_info,executable,test_names,check_rust_report,TOOLS
ROOT=Path(__file__).resolve().parents[2]
NAMES={'v36_mla_noncausal', 'v36_gqa_empty_queries', 'v36_gqa_query_only', 'v36_mla_empty_histories', 'v36_gqa_noncausal', 'v36_gqa_empty_histories', 'v36_gqa_key_only', 'v36_mla_fp16_sparse', 'v36_gqa_fp32_sparse', 'v36_mla_latent_only', 'v36_compaction_poison_inactive_inputs', 'v36_mla_fp32_sparse', 'v36_gqa_full_occupancy', 'v36_mla_latent_512', 'v36_gqa_request_without_queries', 'v36_uniform_attention_matches_analytic_gradient', 'v36_mla_queries_only', 'v36_gqa_fp16_sparse', 'v36_gqa_wide_tail', 'v36_mla_position_only', 'v36_gqa_value_only'}
MARKER='RUDA_V36_HISTORY_COMPACTION_GPU_EXECUTED'
PREFIX='paged_attention::tests_history_compaction::v36_'

def benchmark_samples(text):
    pattern=r'RUDA_V36_HISTORY_COMPACTION_BENCH mla=(true|false) queries=(\d+) length=(\d+) reserved=(\d+) dim=(\d+) sample=(\d+) compact=(true|false) seconds_with_readback=([0-9.eE+-]+)'
    found=re.findall(pattern,text)
    expected={(mla,n,length,reserved,d,sample,compact)
              for mla,n,length,reserved,d in [('false',32,128,0,128),('false',32,128,240,128),('true',32,128,240,512)]
              for sample in range(7) for compact in ('true','false')}
    seen=set();out=[]
    for mla,n,length,reserved,d,sample,compact,seconds in found:
        key=(mla,int(n),int(length),int(reserved),int(d),int(sample),compact);value=float(seconds)
        if key in seen or key not in expected or not math.isfinite(value) or value<=0:
            raise ValueError('bad paired benchmark sample')
        seen.add(key)
        out.append({'mla':mla=='true','queries':key[1],'length':key[2],'reserved_pages':key[3],'dim':key[4],
                    'sample':key[5],'compact':compact=='true','seconds_including_readback':value})
    if seen!=expected:raise ValueError('missing paired benchmark samples')
    return out

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--build',action='store_true',help='also rebuild and test native PyTorch; otherwise Rust device acceptance only')
    parser.add_argument('--sanitizers',default='none');parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--bf16',action='store_true',help='extra BF16 tests; explicitly refuse unsupported hardware')
    parser.add_argument('--timeout',type=int,default=7200)
    args=parser.parse_args(argv);tools=list(TOOLS) if args.sanitizers=='all' else ([] if args.sanitizers=='none' else args.sanitizers.split(','))
    if args.timeout<=0 or len(tools)!=len(set(tools)) or any(t not in TOOLS for t in tools):parser.error('invalid options')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);summary=out/'summary.json'
    if summary.exists():parser.error('choose a fresh result directory')
    report={'version':'v36','rust_host_validated':False,'rust_gpu_validated':False,'pytorch_gpu_validated':False,
            'benchmark_measured':False,'commands':[],'runs':[],'errors':[]}
    env=dict(os.environ,RUDA_REQUIRE_GPU='1',RUDA_PAGED_ORDERED_COMPACT_HISTORY='1');env.setdefault('CARGO_BUILD_JOBS','1')
    def save():summary.write_text(json.dumps(report,indent=2)+'\n')
    def run(label,cmd):
        path=out/(label+'.log')
        with path.open('w') as f:
            try:code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:code=124
            except OSError as exc:f.write(str(exc)+'\n');code=127
        text=path.read_text(errors='replace');report['commands'].append({'command':cmd,'exit_code':code,'log':path.name,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()});save();return code,text
    try:
        missing=[x for x in ('rustc','cargo') if shutil.which(x) is None]
        if missing:raise RuntimeError('missing toolchain: '+','.join(missing)+'; no device tests executed')
        if env.get('RUDA_CUDA_COMPILER')!='ptx' or not env.get('RUDA_PTX_VERSION'):raise RuntimeError('set direct PTX and driver-supported RUDA_PTX_VERSION')
        report['hardware']=gpu_info()
        report['compact_history_env']=env['RUDA_PAGED_ORDERED_COMPACT_HISTORY']
        if args.bf16 and report['hardware']['compute_capability'][0]<8:
            raise RuntimeError('BF16 requested on unsupported hardware')
        if tools and not shutil.which('compute-sanitizer'):raise RuntimeError('compute-sanitizer unavailable')
        report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'ruDNN/src/paged_attention').glob('*.rs'))}
        report['cargo_lock_sha256']=hashlib.sha256((ROOT/'Cargo.lock').read_bytes()).hexdigest()
        code,_=run('rust-host',[sys.executable,str(ROOT/'tools/gpu_validation/rust_host_v36.py'),'--output',str(out/'rust-host')])
        if code:raise RuntimeError('production Rust host test failed')
        host=json.loads((out/'rust-host/summary.json').read_text())
        if not host.get('rust_host_executed') or host.get('tests_executed')!=18:raise RuntimeError('no production host test evidence')
        report['rust_host_validated']=True
        code,text=run('build-rudnn',['cargo','test','--locked','--release','-p','ruDNN','--lib',
            '--features','tensor-paged-attention,ruda-test-runtime/cuda','--no-run','--message-format=json'])
        if code:raise RuntimeError('Rust/DSL device test build failed')
        binary=executable(text,'rudnn');report['test_binary_sha256']=hashlib.sha256(Path(binary).read_bytes()).hexdigest()
        code,text=run('test-list',[binary,PREFIX,'--list'])
        if code or {n.rsplit('::',1)[-1] for n in test_names(text)}!=NAMES:raise RuntimeError('unexpected native GPU test set')
        for tool in [None,*tools]:
            prefix=[] if tool is None else ['compute-sanitizer','--tool',tool,'--error-exitcode','97']
            code,text=run('rust-'+(tool or 'plain'),prefix+[binary,PREFIX,'--nocapture','--test-threads=1'])
            count=check_rust_report(text,code,len(NAMES),MARKER,tool)
            report['runs'].append({'kind':'rust_gpu','sanitizer':tool,'passed':count});save()
        if args.bf16:
            for tool in [None,*tools]:
                prefix=[] if tool is None else ['compute-sanitizer','--tool',tool,'--error-exitcode','97']
                code,text=run('bf16-'+(tool or 'plain'),prefix+[binary,'bf16_history_compaction_v36','--nocapture','--test-threads=1'])
                count=check_rust_report(text,code,1,MARKER,tool)
                report['runs'].append({'kind':'rust_gpu_bf16','sanitizer':tool,'passed':count});save()
        report['rust_gpu_validated']=True;save()
        if args.benchmark:
            code,text=run('benchmark',[binary,'benchmark_history_compaction_v36','--ignored','--nocapture','--test-threads=1'])
            check_rust_report(text,code,1,MARKER);report['benchmark_samples']=benchmark_samples(text)
            report['benchmark_measured']=True;save()
        if args.build:
            code,_=run('pytorch',[sys.executable,str(ROOT/'ruda-torch/tools/validate_paged_pruning.py'),'--build',
                                '--sanitizers',args.sanitizers,'--output',str(out/'pytorch')])
            if code:raise RuntimeError('native PyTorch acceptance failed')
            child=json.loads((out/'pytorch/summary.json').read_text())
            if not child.get('gpu_validated'):raise RuntimeError('missing PyTorch GPU evidence')
            report['pytorch_gpu_validated']=True;save()
        save();return 0
    except Exception as exc:
        report['errors'].append(str(exc));save();print('REFUSED/FAILED:',exc,file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
