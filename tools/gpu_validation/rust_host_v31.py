#!/usr/bin/env python3
"""Compile/import the actual production launch planner with rustc; no substitute."""
import argparse,json,os,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);report={'version':'v31','production_rust_compiled':False,'tests_passed':0,'gpu_validated':False,'errors':[]}
 target=out/'summary.json'
 if target.exists():p.error('use a fresh output directory')
 def save():target.write_text(json.dumps(report,indent=2)+'\n')
 rust=shutil.which('rustc')
 if not rust:report['errors'].append('missing rustc; no production code executed');save();return 2
 source=ROOT/'ruBLAS/src/tensor_grouped/backward_plan.rs'
 harness=out/'planner_harness.rs';harness.write_text('#[path = '+json.dumps(str(source))+']\nmod planner;\n')
 binary=out/('planner_tests.exe' if os.name=='nt' else 'planner_tests')
 try:
  for name,cmd in [('build',[rust,'--edition=2024','--test',str(harness),'-o',str(binary)]),('run',[str(binary),'--test-threads=1'])]:
   r=subprocess.run(cmd,text=True,capture_output=True,timeout=300);text=r.stdout+r.stderr;(out/(name+'.log')).write_text(text)
   if r.returncode:raise RuntimeError(name+' failed')
   if name=='build':report['production_rust_compiled']=True
   else:
    if not re.search(r'test result: ok\. 6 passed; 0 failed; 0 ignored;',text):raise RuntimeError('missing planner cases')
    report['tests_passed']=6
  save();return 0
 except Exception as exc:report['errors'].append(str(exc));save();return 2
if __name__=='__main__':raise SystemExit(main())
