#!/usr/bin/env python3
"""Reproduce host validation; --gpu additionally requires real RUDA execution.

C++ callback fixtures MUST run in separate Python processes because the backend
is initialized once per process. Simulated native regions and algorithm oracles
are reported separately; they never count as GPU or Rust validation.
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
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--compiler',default='clang++')
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        parser.error('use a fresh output directory; previous validation evidence is not overwritten')
    report={'rust_compiled':False,'gpu_executed':False,'host_passed':False,
        'rust_toolchain':{name:shutil.which(name) for name in ['cargo','rustc','rustfmt']},'commands':[],'groups':[]}
    env=dict(os.environ,OMP_NUM_THREADS='1');code=2
    def run(label,command,environment=env):
        with (out/(label+'.log')).open('w') as log:
            completed=subprocess.run(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
        report['commands'].append({'name':label,'command':command,'returncode':completed.returncode})
        return completed.returncode
    def tests(label,files,environment=env):
        xml=out/(label+'.xml')
        command=[sys.executable,'-m','pytest','-q',*files,'--junitxml='+str(xml)]
        rc=run(label,command,environment)
        cases=list(ET.parse(xml).getroot().iter('testcase')) if xml.exists() else []
        counts={tag:sum(c.find(tag) is not None for c in cases) for tag in ['failure','error','skipped']}
        report['groups'].append({'name':label,'tests':len(cases),**counts,'returncode':rc})
        if rc or not cases or any(counts.values()):raise RuntimeError(label+' failed or skipped tests')
    try:
        import torch
        report['torch']=torch.__version__
        report['python']=sys.version
        prefix='ruda-torch/python/tests/'
        tests('regression',[prefix+'test_'+n+'.py' for n in ['static_graph_spec','static_graph_optimizer','static_graph_training','model_compile']]+['tools/validation/test_feature_reference.py'])
        tests('native-region-simulator',[prefix+'test_compile_native.py',prefix+'test_compile_native_extended.py'])
        tests('learned-fake-quantization',[prefix+'test_learned_quantization.py'])
        tests('packed-awq-algorithm-oracle',['tools/validation/test_awq_packed_reference.py'])
        cpp=out/'cpp'
        command=[sys.executable,'ruda-torch/tools/check_cpp_bridge.py','--output',str(cpp),'--compiler',args.compiler,'--static-graph']
        if run('cpp-build',command):raise RuntimeError('C++ bridge build/protocol check failed')
        cpp_report=json.loads((cpp/'result.json').read_text());report['cpp']=cpp_report
        libs=list(cpp.glob('_C*.so'))
        if len(libs)!=1:raise RuntimeError('missing/ambiguous built C++ library')
        cpp_env=dict(env,RUDA_CPP_TEST_LIBRARY=str(libs[0]))
        tests('cpp-extended-contract',[prefix+'test_static_graph_extended_cpp.py'],cpp_env)
        tests('cpp-model-integration',[prefix+'test_model_compile_cpp.py'],cpp_env)
        report['host_test_count']=sum(g['tests'] for g in report['groups'])+cpp_report['host_protocol_tests_passed']+cpp_report['static_graph_cpp_host_protocol_tests_passed']
        report['host_passed']=True;code=0
        if args.gpu:
            target=out/'gpu.json'
            gpu_rc=run('gpu-acceptance',[sys.executable,'ruda-torch/tools/validate_model_compile.py','--output',str(target)])
            report['gpu']=json.loads(target.read_text()) if target.exists() else {'error':'no device report'}
            report['gpu_executed']=bool(report['gpu'].get('gpu_executed'))
            if gpu_rc or not report['gpu_executed'] or report['gpu'].get('status') != 'passed':code=2
    except Exception as exc:
        code=2
        report['error']=f'{type(exc).__name__}: {exc}'
    (out/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False));return code

if __name__=='__main__':raise SystemExit(main())
