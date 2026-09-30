#!/usr/bin/env python3
"""Compile the production workspace planner, not a translated/mock Rust copy."""
import argparse,json,shutil,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    report={'rust_compiled':False,'gpu_validated':False,'production_module':str(root/'ruda-optim/src/fused_adamw/stats_plan.rs')}
    try:
        if not shutil.which('rustc'):raise RuntimeError('rustc unavailable; no Rust tests executed')
        binary=out/'stats-plan-tests'
        command=['rustc','--edition=2024','--test',report['production_module'],'-o',str(binary)]
        build=subprocess.run(command,capture_output=True,text=True,timeout=120)
        (out/'build.log').write_text(build.stdout+build.stderr)
        report['build_exit_code']=build.returncode
        if build.returncode:raise RuntimeError('production Rust planner did not compile')
        report['rust_compiled']=True
        run=subprocess.run([str(binary),'--test-threads=1'],capture_output=True,text=True,timeout=120)
        (out/'tests.log').write_text(run.stdout+run.stderr);report['test_exit_code']=run.returncode
        if run.returncode or '6 passed; 0 failed' not in run.stdout:raise RuntimeError('production Rust tests incomplete or failed')
        report['passed']=6;code=0
    except (RuntimeError,OSError,subprocess.TimeoutExpired) as e:report['error']=str(e);code=2
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return code
if __name__=='__main__':raise SystemExit(main())
