#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scoped kernel optimization verification. No automatic installations or claimed speedup.

Python oracle != Rust build != CUDA execution != measured benchmark.
Missing required tools are BLOCKED. Logs are timestamped and never overwritten.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, importlib.util, json, math, os, shutil, subprocess, sys
from pathlib import Path
from run_safety_regressions import run
ROOT=Path(__file__).resolve().parents[1]
SUITES=['oracle','plan','host','build','cuda','sanitizer','bench']

def plans(suite,compiler,logs,offline=False,batch=256,order=16,rhs=4,samples=7,iterations=3,warmup=2):
    env=os.environ.copy();commands=[];required=[]
    base=['cargo'];extra=['--offline']if offline else[]
    common=['--locked',*extra,'-p','ruda-solver','--no-default-features']
    if suite=='oracle':
        required=[sys.executable];commands=[([sys.executable,'tools/kernel_optimization/oracle.py','--out',str(logs/'oracle.json')],env)]
    elif suite=='plan':
        required=['rustc'];exe=logs/('plan-tests.exe'if os.name=='nt'else'plan-tests')
        commands=[(['rustc','--edition=2024','--test','ruSOLVER/src/kernel_plan.rs','-o',str(exe)],env),([str(exe)],env)]
    elif suite=='host':
        required=['cargo','rustc'];commands=[(base+['test',*common,'--lib','kernel_plan::tests'],env)]
    elif suite=='build':
        required=['cargo','rustc'];commands=[(base+['check',*common,'--features','cuda,warp-solvers','--test','device_warp','--example','solver-warp-bench'],env)]
    else:
        required=['cargo','rustc']
        if suite=='sanitizer':required+=['compute-sanitizer']
        for mode in(['nvrtc','ptx']if compiler=='both'else[compiler]):
            selected=env.copy();selected['RUDA_CUDA_COMPILER']=mode
            test=base+['test','--release',*common,'--features','cuda,warp-solvers']
            if suite=='cuda':
                for name in['device_cholesky','device_advanced','device_warp']:
                    commands.append((test+['--test',name,'--','--test-threads=1'],selected))
            elif suite=='sanitizer':
                # Build outside instrumentation. Then follow cargo's child test binary.
                commands.append((test+['--test','device_warp','--no-run'],selected))
                for tool in['memcheck','racecheck','synccheck']:
                    commands.append((['compute-sanitizer','--tool',tool,'--error-exitcode','99','--target-processes','all']+
                        test+['--test','device_warp','sanitizer_smoke_all_lanes_tails_and_failures','--','--test-threads=1','--nocapture'],selected))
            elif suite=='bench':
                for kind in['cholesky','lu']:
                    commands.append((base+['run','--release',*common,'--features','cuda,warp-solvers','--example','solver-warp-bench','--',
                        '--kind',kind,'--batch',str(batch),'--order',str(order),'--rhs',str(rhs),'--samples',str(samples),
                        '--iterations',str(iterations),'--warmup',str(warmup),'--out',str(logs/f'{mode}-{kind}-bench.json')],selected))
    return required,commands

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',choices=SUITES,default='plan')
    p.add_argument('--compiler',choices=['nvrtc','ptx','both'],default='both');p.add_argument('--offline',action='store_true')
    p.add_argument('--timeout',type=float,default=900,help='per-command safety limit, not a predicted completion time')
    p.add_argument('--log-dir',type=Path);p.add_argument('--dry-run',action='store_true')
    for flag,default in[('batch',256),('order',16),('rhs',4),('samples',7),('iterations',3),('warmup',2)]:p.add_argument('--'+flag,type=int,default=default)
    a=p.parse_args()
    if not math.isfinite(a.timeout)or a.timeout<=0:p.error('timeout must be finite and positive')
    if not(1<=a.batch<=65536 and 1<=a.order<=32 and 1<=a.rhs<=8 and 3<=a.samples<=100 and 1<=a.iterations<=100 and 1<=a.warmup<=100):
        p.error('invalid bounded benchmark configuration')
    now=dt.datetime.now(dt.timezone.utc);logs=(a.log_dir or ROOT/'target/kernel-optimization'/now.strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
    required,commands=plans(a.suite,a.compiler,logs,a.offline,a.batch,a.order,a.rhs,a.samples,a.iterations,a.warmup)
    public=[{'command':c,'compiler':e.get('RUDA_CUDA_COMPILER')}for c,e in commands]
    if a.dry_run:print(json.dumps({'status':'planned_only','suite':a.suite,'commands':public},indent=2));return 0
    logs.mkdir(parents=True,exist_ok=True)
    if(logs/'results.json').exists():p.error('log directory contains results.json; choose a fresh directory')
    sources=list((ROOT/'ruSOLVER/src/tensor').glob('*.rs'))+[ROOT/'ruSOLVER/src/kernel_plan.rs',ROOT/'ruSOLVER/tests/device_warp.rs',ROOT/'ruSOLVER/examples/solver_warp_bench.rs']
    report={'schema':'ruda.warp_direct.validation.v1','suite':a.suite,'status':'running','started_utc':now.isoformat(),'planned':public,'results':[],
        'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in sources},
        'rust_executed':False,'device_tests_executed':False,'performance_measured':False}
    def save():
        report['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat();tmp=logs/'results.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(logs/'results.json')
    missing=[t for t in required if not shutil.which(t)]
    if a.suite=='oracle':missing += [m for m in ['numpy','scipy'] if importlib.util.find_spec(m) is None]
    if missing:report.update(status='blocked',reason='Missing tools: '+', '.join(missing));save();print(report['reason']);return 2
    report['environment']={k:os.environ.get(k)for k in ['RUSTUP_TOOLCHAIN','RUDA_PTX_VERSION','CUDA_VISIBLE_DEVICES','CARGO_TARGET_DIR']}
    report['environment']['python']=sys.version
    for tool,args in [('rustc',['--version']),('cargo',['--version']),('nvidia-smi',['--query-gpu=name,driver_version,pci.bus_id','--format=csv,noheader'])]:
        if shutil.which(tool):
            try:
                probe=subprocess.run([tool,*args],capture_output=True,text=True,timeout=10,check=False)
                report['environment'][tool]={'returncode':probe.returncode,'output':(probe.stdout+probe.stderr).strip()}
            except (OSError,subprocess.TimeoutExpired)as e:report['environment'][tool]={'error':str(e)}
    save()
    for i,(command,env)in enumerate(commands):
        result=run(command,logs/f'{i+1:02}.log',a.timeout,env);report['results'].append(result);save()
        if result['status']!='passed':break
    passed=len(report['results'])==len(commands)and all(x['status']=='passed'for x in report['results'])
    report['status']='passed'if passed else report['results'][-1]['status']
    if passed:
        report['rust_executed']=a.suite in['plan','host','cuda','sanitizer','bench']
        report['device_tests_executed']=a.suite in['cuda','sanitizer','bench']
        if a.suite=='bench':
            paths=sorted(logs.glob('*-bench.json'));expected=4 if a.compiler=='both'else 2
            report['benchmark_reports']=[str(x)for x in paths]
            # Never infer performance measurement from a successful cargo invocation alone.
            if len(paths)!=expected or any(json.loads(x.read_text()).get('status')!='passed'for x in paths):
                report.update(status='failed',reason='missing/invalid actual benchmark output')
            else:report['performance_measured']=True
    report['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat();save();print(f"{report['status']}: {logs/'results.json'}")
    return 0 if report['status']=='passed'else 2 if report['status']=='blocked'else 1
if __name__=='__main__':raise SystemExit(main())
