#!/usr/bin/env python3
"""Compile and execute the actual Rust device-IR tests, without CANN or Cargo.
The test-only VM checks the Rust program; it is never a production CPU fallback.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    report={'rust_compiled':False,'rust_tests_passed':False,'npu_compiled':False,'npu_executed':False,'stages':[]}
    def save():(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    cc=shutil.which('rustc')
    if cc is None:report['status']='missing_rustc';save();print('missing rustc; no Rust test executed',file=sys.stderr);return 1
    crate=ROOT/'ruda-ascend-kernels';exe=out/'rust-device-program-tests'
    commands=[('compile',[cc,'--edition=2024','--test','-C','opt-level=2',str(crate/'src/lib.rs'),'-o',str(exe)]),
              ('test',[str(exe),'--nocapture'])]
    for label,cmd in commands:
        r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
        (out/(label+'.log')).write_text(r.stdout);report['stages'].append({'name':label,'command':cmd,'returncode':r.returncode})
        if r.returncode!=0:report['status']='failed';save();return 1
        report['rust_compiled' if label=='compile' else 'rust_tests_passed']=True;save()
    report['status']='host_ir_tests_passed_not_npu_validated';save();return 0
if __name__=='__main__':raise SystemExit(main())
