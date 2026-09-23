#!/usr/bin/env python3
"""Compile/load the real C++ bridge with host-only ABI test callbacks.
This does NOT link Rust, compile PTX or run a GPU operator. Linux only.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('cpp-bridge-validation'))
    p.add_argument('--compiler',default=os.environ.get('CXX','clang++'))
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    report={'cpp_compiled':False,'cpp_loaded_and_protocol_tested':False,'gpu_validated':False,'rust_compiled':False,'commands':[]}
    try:
        if sys.platform!='linux':raise RuntimeError('This verification script currently targets Linux shared-library linking')
        if not shutil.which(args.compiler):raise RuntimeError('C++ compiler not found')
        import torch
        from torch.utils.cpp_extension import include_paths,library_paths
        library=out/('_C'+sysconfig.get_config_var('EXT_SUFFIX'))
        cmd=[args.compiler,'-std=c++20','-shared','-fPIC','-O0','-DTORCH_EXTENSION_NAME=_C',
             '-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
        cmd+=['-I'+d for d in include_paths()+[sysconfig.get_paths()['include']]]
        cmd+=[str(root/'ruda-torch/python/ruda_torch/csrc/backend.cpp')]
        for path in library_paths():cmd+=['-L'+path,'-Wl,-rpath,'+path]
        cmd+=['-ltorch_python','-ltorch_cpu','-ltorch','-lc10','-o',str(library)]
        with (out/'build.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        report['commands'].append({'command':cmd,'returncode':result.returncode})
        if result.returncode:raise RuntimeError('C++ compilation/link failed; see build.log')
        report['cpp_compiled']=True;report['torch']=torch.__version__
        junit=out/'cpp-tests.xml'
        cmd=[sys.executable,'-m','pytest','-q',str(root/'ruda-torch/python/tests/test_v15_cpp.py'),f'--junitxml={junit}']
        with (out/'tests.log').open('w') as log:
            result=subprocess.run(cmd,cwd=root,env=dict(os.environ,RUDA_CPP_TEST_LIBRARY=str(library)),stdout=log,stderr=subprocess.STDOUT)
        report['commands'].append({'command':cmd,'returncode':result.returncode})
        if result.returncode:raise RuntimeError('C++ load/protocol test failed; see tests.log')
        cases=list(ET.parse(junit).getroot().iter('testcase'))
        if len(cases)<8 or any(c.find(tag) is not None for c in cases for tag in ['failure','error','skipped']):
            raise RuntimeError('missing or skipped C++ test cases')
        report['cpp_loaded_and_protocol_tested']=True;report['host_protocol_tests_passed']=len(cases)
        code=0
    except (RuntimeError,OSError,ImportError,ET.ParseError) as error:
        report['error']=str(error);code=2
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return code
if __name__=='__main__':raise SystemExit(main())
