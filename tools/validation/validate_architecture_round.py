#!/usr/bin/env python3
"""Reproduce mHC/CSA/HCA/DSA/Muon evidence in independent, resumable phases.

Core and legacy phases execute CPU numerics. C++ is host protocol/registration
only. gpu/rust phases fail explicitly when native dependencies are absent.
An existing phase result is never overwritten. No skip counts as success.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['core','legacy','cpp','examples','gpu','rust'],default='core')
    parser.add_argument('--cpp-library',type=Path,help='Reuse a separately built real C++ bridge; no compilation claim for this phase')
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    result=out/(args.phase+'.json')
    if result.exists():parser.error('phase result exists; choose a fresh output directory')
    env=dict(os.environ,OMP_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    report={'phase':args.phase,'status':'failed','commands':[],'groups':[],
        'cpu_reference':args.phase in ('core','legacy','examples'),
        'gpu_executed':False,'rust_compiled':False,'cpp_compiled':False,
        'toolchain':{name:shutil.which(name) for name in ['cargo','rustc','rustfmt','clang++']}}
    def run(label,command,environment=env,timeout=300):
        log=out/(label+'.log')
        with log.open('w') as stream:
            try:
                proc=subprocess.run(command,cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
                rc=proc.returncode
            except subprocess.TimeoutExpired:
                rc=124;stream.write('\nVALIDATOR: command deadline exceeded\n')
        report['commands'].append({'label':label,'command':command,'returncode':rc,'log':log.name})
        return rc
    def tests(label,files,environment=env):
        xml=out/(label+'.xml')
        rc=run(label,[sys.executable,'-m','pytest','-q',*files,'--junitxml='+str(xml)],environment)
        cases=list(ET.parse(xml).getroot().iter('testcase')) if xml.exists() else []
        counts={tag:sum(case.find(tag) is not None for case in cases) for tag in ['failure','error','skipped']}
        report['groups'].append({'name':label,'tests':len(cases),'returncode':rc,**counts})
        if rc or not cases or any(counts.values()):raise RuntimeError(label+' failed, skipped, timed out, or collected no tests')
    code=2
    try:
        import torch
        report['torch']=torch.__version__;report['python']=sys.version
        prefix='ruda-torch/python/tests/'
        if args.phase=='core':
            tests('architecture-numerics',[prefix+'test_'+n+'.py' for n in
                ['architecture_primitives','compressed_attention','muon_optimizer','hybrid_architecture_training','training_host']])
        elif args.phase=='legacy':
            tests('previous-non-cpp-regressions',[prefix+'test_'+n+'.py' for n in
                ['static_graph_spec','static_graph_optimizer','static_graph_training','model_compile',
                 'compile_native','compile_native_extended','learned_quantization']]+
                 ['tools/validation/test_feature_reference.py','tools/validation/test_awq_packed_reference.py'])
        elif args.phase=='cpp':
            library=args.cpp_library
            if library is None:
                build=out/'cpp'
                if run('cpp-build',[sys.executable,'ruda-torch/tools/check_cpp_bridge.py','--output',str(build),'--static-graph'],timeout=300):
                    raise RuntimeError('C++ build/protocol checks failed')
                details=json.loads((build/'result.json').read_text());report['cpp_build']=details
                report['cpp_compiled']=bool(details['cpp_compiled'])
                libraries=list(build.glob('_C*.so'))
                if len(libraries)!=1:raise RuntimeError('ambiguous/missing compiled C++ library')
                library=libraries[0]
            library=library.resolve()
            if not library.is_file():raise RuntimeError('C++ library not found')
            cpp_env=dict(env,RUDA_CPP_TEST_LIBRARY=str(library))
            # Keep each ABI fixture in a fresh process. Base tests are repeated
            # for reused binaries, but counted once in this phase.
            for label,file in [('cpp-base','test_v15_cpp.py'),('cpp-static','test_static_graph_cpp.py'),
                ('cpp-extended','test_static_graph_extended_cpp.py'),('cpp-model','test_model_compile_cpp.py'),
                ('cpp-architecture-dispatch','test_architecture_dispatch_cpp.py')]:
                tests(label,[prefix+file],cpp_env)
        elif args.phase=='examples':
            example='ruda-torch/python/examples/train_hybrid_attention.py'
            for label,flags in [('example-eager',['--warmup-steps','2','--steps','3']),
                                ('example-compiled',['--compile','--warmup-steps','1','--steps','2'])]:
                target=out/(label+'.json')
                if run(label,[sys.executable,example,'--cpu-reference',*flags,'--output',str(target)],timeout=300):
                    raise RuntimeError(label+' failed')
                details=json.loads(target.read_text())
                if details['status']!='passed' or details['gpu_executed']:raise RuntimeError('invalid CPU evidence')
                report[label]=details
        elif args.phase=='gpu':
            target=out/'device-acceptance.json'
            native_env=dict(env,PYTHONPATH=str(root/'ruda-torch/python')+os.pathsep+env.get('PYTHONPATH',''))
            rc=run('device-acceptance',[sys.executable,'ruda-torch/python/examples/train_hybrid_attention.py',
                '--warmup-steps','1','--steps','2','--compile','--output',str(target)],native_env)
            details=json.loads(target.read_text()) if target.exists() else {'error':'no device report'}
            report['device']=details;report['gpu_executed']=bool(details.get('gpu_executed'))
            if rc or not report['gpu_executed'] or details.get('status')!='passed':raise RuntimeError('real RUDA acceptance did not pass')
        else:
            if shutil.which('cargo') is None:raise RuntimeError('cargo is unavailable; Rust tests were not executed')
            if run('rust-mhc',['cargo','test','-p','ruda-nn','--lib','mhc'],timeout=900):raise RuntimeError('Rust mHC tests failed')
            report['rust_compiled']=True
        report['tests']=sum(group['tests'] for group in report['groups'])
        report['status']='passed';code=0
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
    result.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False));return code

if __name__=='__main__':raise SystemExit(main())
