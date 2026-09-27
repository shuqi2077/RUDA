#!/usr/bin/env python3
"""Compile and run production graph_contract.rs alone; no GPU emulator."""
import argparse,json,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('v24-rust-host'));a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    report={'production_rust_compiled':False,'tests_passed':0,'full_crate_compiled':False,'gpu_validated':False,'commands':[]}
    code=2
    try:
        rustc=shutil.which('rustc')
        if not rustc:raise RuntimeError('missing rustc; no substitute execution')
        source=ROOT/'ruda-torch/src/graph_contract.rs';binary=out/'graph-contract-tests'
        for name,cmd in [('compile',[rustc,'--edition=2024','--test',str(source),'-o',str(binary)]),
                         ('test',[str(binary),'--test-threads=1'])]:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
            text=r.stdout+r.stderr;(out/(name+'.log')).write_text(text)
            report['commands'].append({'command':cmd,'returncode':r.returncode})
            if r.returncode:raise RuntimeError(name+' failed')
            if name=='compile':report['production_rust_compiled']=True
            elif not re.search(r'test result: ok\. 20 passed; 0 failed; 0 ignored;',text):raise RuntimeError('incomplete production Rust tests')
            else:report['tests_passed']=20
        code=0
    except (OSError,RuntimeError,subprocess.TimeoutExpired) as e:report['error']=str(e)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return code
if __name__=='__main__':raise SystemExit(main())
