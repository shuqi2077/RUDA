#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify extended science APIs; tool absence is BLOCKED, never a passing test.

No auto-install, no implicit GPU fallback, bounded subprocesses and separate
Python formula evidence vs real Rust/backend runs.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, os, shutil, sys
from pathlib import Path
from run_safety_regressions import run
from run_science_regressions import plans as base_plans
ROOT=Path(__file__).resolve().parents[1]
SUITES=['oracle','rust-host','sparse-host','host','build','cuda','autodiff','collective']


def plans(suite,compiler,logs,offline=False,world=2,order=32):
    env=os.environ.copy();flags=['--locked']+(['--offline']if offline else [])
    if suite=='oracle':return [],[([sys.executable,'tools/science_extended/oracle.py','--output',str(logs/'oracle.json')],env)]
    if suite=='autodiff':return ['cargo','rustc'],[(['cargo','test',*flags,'-p','ruda-autodiff','--features','solver-host','--lib','solver_host::tests','--','--test-threads=1'],env)]
    if suite=='collective':
        binary=str(ROOT/'target'/'debug'/'examples'/('distributed-cg.exe'if os.name=='nt'else'distributed-cg'))
        return ['cargo','rustc'],[(['cargo','build',*flags,'-p','ruda-solver','--features','collective','--example','distributed-cg','--target-dir',str(ROOT/'target')],env),
            ([sys.executable,'tools/science_extended/tcp_demo.py','--binary',binary,'--world',str(world),'--order',str(order),'--output',str(logs/'tcp')],env)]
    return base_plans(suite,compiler,logs,offline)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',choices=SUITES,default='rust-host');p.add_argument('--compiler',choices=['nvrtc','ptx','both'],default='both')
    p.add_argument('--timeout',type=float,default=300,help='Per-command cap, not a duration estimate');p.add_argument('--log-dir',type=Path);p.add_argument('--offline',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--world',type=int,default=2);p.add_argument('--order',type=int,default=32);a=p.parse_args()
    if not math.isfinite(a.timeout)or a.timeout<=0:p.error('timeout must be finite and positive')
    if not 1<=a.world<=16 or not 1<=a.order<=100000:p.error('demo world must be 1..16, order 1..100000')
    now=dt.datetime.now(dt.timezone.utc);logs=(a.log_dir or ROOT/'target'/'science-extended'/now.strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
    required,commands=plans(a.suite,a.compiler,logs,a.offline,a.world,a.order)
    public=[{'command':cmd,'compiler':env.get('RUDA_CUDA_COMPILER')}for cmd,env in commands]
    if a.dry_run:print(json.dumps({'status':'planned_only','plans':public},indent=2));return 0
    if logs.exists()and any(logs.iterdir()):p.error('use a new/empty log directory; prior evidence is retained')
    logs.mkdir(parents=True,exist_ok=True)
    sources=[]
    for folder in['ruSOLVER','ruINTEGRATE','tools/science_extended']:
        sources +=[f for f in(ROOT/folder).rglob('*')if f.is_file()and f.suffix in('.rs','.toml','.json','.py')]
    sources +=[ROOT/q for q in['Cargo.toml','Cargo.lock','ruda-autodiff/Cargo.toml','ruda-autodiff/src/solver_host.rs','ruda-autodiff/src/lib.rs','tools/run_science_regressions.py','tools/run_science_extended_regressions.py']]
    report={'schema':'ruda.science.extended.validation.v1','suite':a.suite,'started_utc':now.isoformat(),'status':'running','plans':public,'results':[],
        'performance_measured':False,'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in sources}}
    def save():
        report['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat();tmp=logs/'results.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(logs/'results.json')
    missing=[x for x in required if not shutil.which(x)]
    if missing:report.update(status='blocked',reason='Missing tools: '+', '.join(missing));save();print(report['reason']);return 2
    save()
    for i,(cmd,env)in enumerate(commands):
        result=run(cmd,logs/f'{i+1:02}.log',a.timeout,env)
        if a.suite=='oracle'and result.get('returncode')==3:result['status']='blocked'
        report['results'].append(result);save()
        if result['status']!='passed':break
    passed=len(report['results'])==len(commands)and all(r['status']=='passed'for r in report['results'])
    report.update(status='passed'if passed else report['results'][-1]['status'],finished_utc=dt.datetime.now(dt.timezone.utc).isoformat());save()
    print(f"{report['status']}: {logs/'results.json'}");return 0 if passed else 2 if report['status']=='blocked'else 1
if __name__=='__main__':raise SystemExit(main())
