#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run scoped Muon checks. Missing tools are BLOCKED; no auto-installation."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from run_safety_regressions import run

ROOT=Path(__file__).resolve().parents[1]

def plans(suite: str, compiler: str, logs: Path, offline: bool=False):
    flags=['--locked']+(['--offline'] if offline else [])
    env=os.environ.copy()
    if suite=='oracle':
        return [], [([sys.executable,'tools/muon/oracle.py','--report',str(logs/'oracle.json')],env)]
    if suite=='reference':
        executable=logs/('reference-tests.exe' if os.name=='nt' else 'reference-tests')
        return ['rustc'], [(['rustc','--edition','2024','--test','tools/muon/reference_tests.rs','-o',str(executable)],env),([str(executable)],env)]
    if suite=='host':
        return ['cargo','rustc'], [(['cargo','test',*flags,'-p','ruda-optim','--lib','optim::muon::','--','--test-threads=1'],env)]
    if suite=='build':
        return ['cargo','rustc'], [(['cargo','check',*flags,'-p','ruda-optim','--lib'],env),
                                  (['cargo','check',*flags,'-p','ruda-optim','--lib','--no-default-features'],env),
                                  (['cargo','check',*flags,'-p','ruda-optim','--example','muon-training'],env)]
    if suite=='example':
        return ['cargo','rustc'], [(['cargo','run',*flags,'--release','-p','ruda-optim','--example','muon-training','--','20'],env)]
    commands=[]
    for c in (['nvrtc','ptx'] if compiler=='both' else [compiler]):
        commands.append((['cargo','test',*flags,'--release','-p','ruda-optim','--features','test-cuda,ruda-driver-cuda/direct-ptx',
                         '--lib','optim::muon::','--','--test-threads=1'],{**env,'RUDA_CUDA_COMPILER':c}))
    return ['cargo','rustc'],commands

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['oracle','reference','host','build','cuda','example'],default='host')
    p.add_argument('--compiler',choices=['nvrtc','ptx','both'],default='both')
    p.add_argument('--timeout',type=float,default=300,help='Per-command cap, not an estimated duration')
    p.add_argument('--log-dir',type=Path)
    p.add_argument('--offline',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if not math.isfinite(args.timeout) or args.timeout<=0: p.error('timeout must be positive and finite')
    now=dt.datetime.now(dt.timezone.utc)
    logs=(args.log_dir or ROOT/'target'/'muon'/now.strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
    tools,commands=plans(args.suite,args.compiler,logs,args.offline)
    public=[{'command':c,'compiler':e.get('RUDA_CUDA_COMPILER')} for c,e in commands]
    if args.dry_run:
        print(json.dumps({'status':'planned_only','plans':public},indent=2));return 0
    logs.mkdir(parents=True,exist_ok=True)
    result_file=logs/'results.json'
    if result_file.exists():p.error('choose a new log directory; existing results are never overwritten')
    paths=list((ROOT/'ruda-optim/src/optim/muon').rglob('*.rs'))
    paths += list((ROOT/'tools/muon').glob('*.*'))
    paths += [ROOT/'ruda-optim/Cargo.toml', ROOT/'Cargo.lock', ROOT/'ruda-optim/src/optim/adamw.rs',
              ROOT/'ruda-optim/src/optim/momentum.rs',ROOT/'ruda-optim/examples/muon_training.rs',Path(__file__).resolve()]
    report={'schema':'ruda.muon.validation.v1','suite':args.suite,'started_utc':now.isoformat(),
            'status':'running','plans':public,'results':[], 'performance_measured':False,
            'source_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths if x.is_file()}}
    def save():
        tmp=result_file.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');tmp.replace(result_file)
    missing=[t for t in tools if shutil.which(t) is None]
    if missing:
        report.update(status='blocked',reason='Missing tools: '+', '.join(missing));save();print(report['reason']);return 2
    save()
    for index,(cmd,env) in enumerate(commands):
        result=run(cmd,logs/f'{index+1:02}.log',args.timeout,env)
        if args.suite=='oracle' and result.get('returncode')==2:result['status']='blocked'
        report['results'].append(result);save()
        if result['status']!='passed':break
    passed=len(report['results'])==len(commands) and all(x['status']=='passed' for x in report['results'])
    report.update(status='passed' if passed else report['results'][-1]['status'],finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save();print(f'{report["status"]}: {result_file}')
    return 0 if passed else 2 if report['status']=='blocked' else 1
if __name__=='__main__':sys.exit(main())
