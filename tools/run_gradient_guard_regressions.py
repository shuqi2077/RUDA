#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explicit gradient-guard validation; missing tools are BLOCKED, never PASS.

No automatic installation, CPU fallback, whole-workspace all-features build,
implicit network permission, or performance estimates disguised as measurements.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from run_safety_regressions import run

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['oracle', 'reference', 'host', 'build', 'cuda', 'bench'], default='reference')
    parser.add_argument('--compiler', choices=['nvrtc', 'ptx', 'both'], default='both')
    parser.add_argument('--timeout', type=float, default=600, help='Per-command limit, not an estimated completion time')
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--elements', type=int, default=65536, help='Elements per tensor for benchmark only')
    parser.add_argument('--tensors', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--dtype', choices=['f32', 'f16', 'bf16'], default='bf16')
    parser.add_argument('--amsgrad', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or not 0 < args.elements <= (2**32-1)//4:
        parser.error('invalid timeout or element count')
    if not 1 <= args.tensors <= 1024 or args.iterations <= 0 or args.samples < 3 or args.warmup <= 0:
        parser.error('require 1..1024 tensors, positive iterations/warmup, samples >=3')
    now = dt.datetime.now(dt.timezone.utc)
    logs = (args.log_dir or ROOT/'target'/'gradient-guard'/now.strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
    env = os.environ.copy()
    flags = ['--locked'] + (['--offline'] if args.offline else [])
    plans: list[tuple[list[str], dict[str, str]]] = []
    tools: list[str] = []
    if args.suite == 'oracle':
        plans = [([sys.executable, 'tools/gradient_guard/oracle.py', '--report', str(logs/'oracle.json')], env)]
    elif args.suite == 'reference':
        tools = ['rustc']
        executable = logs/('reference-tests.exe' if os.name == 'nt' else 'reference-tests')
        plans = [(['rustc', '--edition', '2024', '--test', 'tools/gradient_guard/reference_tests.rs', '-o', str(executable)], env),
                 ([str(executable), '--test-threads=1'], env)]
    elif args.suite == 'host':
        tools = ['rustc', 'cargo']
        plans = [(['cargo','test',*flags,'-p','ruda-optim','--lib','--features','gradient-guard','fused_adamw::','--','--test-threads=1'], env)]
    elif args.suite == 'build':
        tools = ['rustc', 'cargo']
        for feature in ['fused-adamw-device', 'gradient-guard-device']:
            plans.append((['cargo','check',*flags,'-p','ruda-optim','--lib','--features',feature], env))
    else:
        tools = ['rustc', 'cargo']
        for backend in (['nvrtc','ptx'] if args.compiler == 'both' else [args.compiler]):
            e = {**env, 'RUDA_CUDA_COMPILER': backend}
            prefix = ['cargo','test',*flags,'--release','-p','ruda-optim','--features','gradient-guard-cuda']
            if args.suite == 'cuda':
                # Preserve the previous optimizer's numerical and metadata contract.
                for target in ['fused-adamw-cuda', 'gradient-guard-cuda']:
                    plans.append((prefix+['--test',target,'--','--test-threads=1'], e))
            else:
                cmd = ['cargo','run',*flags,'--release','-p','ruda-optim','--features','gradient-guard-cuda','--example','gradient-guard-bench','--',
                       '--elements',str(args.elements),'--tensors',str(args.tensors),'--iterations',str(args.iterations),
                       '--samples',str(args.samples),'--warmup',str(args.warmup),'--dtype',args.dtype,
                       '--out',str(logs/f'benchmark-{backend}.json')]
                if args.amsgrad: cmd.append('--amsgrad')
                plans.append((cmd, e))
    public = [{'command':c, 'compiler':e.get('RUDA_CUDA_COMPILER')} for c,e in plans]
    if args.dry_run:
        print(json.dumps({'status':'planned_only','suite':args.suite,'plans':public}, indent=2)); return 0
    logs.mkdir(parents=True, exist_ok=True)
    path = logs/'results.json'
    if path.exists(): parser.error('results.json already exists; choose a fresh log directory')
    source_files = sorted((ROOT/'ruda-optim/src/fused_adamw').glob('*.rs'))
    source_files += [ROOT/'ruda-optim/Cargo.toml', ROOT/'Cargo.lock', Path(__file__).resolve()]
    source_files += sorted((ROOT/'tools/gradient_guard').glob('*.*'))
    source_files += [ROOT/'ruda-optim/tests/gradient_guard_cuda.rs', ROOT/'ruda-optim/examples/gradient_guard_bench.rs', ROOT/'ruda-optim/examples/gradient_guard/staged.rs',
                     ROOT/'ruda-optim/tests/fused_adamw/common.rs', ROOT/'ruda-optim/tests/fused_adamw_cuda.rs',
                     ROOT/'ruda-optim/examples/fused_adamw/staged.rs']
    report = {'schema':'ruda.gradient_guard.validation.v1','suite':args.suite,'started_utc':now.isoformat(),'status':'running',
              'plans':public,'results':[], 'performance_measured':False,
              'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
              'environment':{'RUDA_PTX_VERSION':env.get('RUDA_PTX_VERSION')}}
    def save():
        tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)+'\n'); tmp.replace(path)
    save()
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        report.update(status='blocked',reason='Missing tools: '+', '.join(missing),finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save(); print(f'blocked: {report["reason"]}\n{path}'); return 2
    for tool in tools:
        try:
            p = subprocess.run([tool,'--version'], capture_output=True,text=True,timeout=15,check=False)
            report['environment'][tool] = {'returncode':p.returncode,'version':(p.stdout+p.stderr).strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            report['environment'][tool] = str(exc)
    for i, (cmd, e) in enumerate(plans, 1):
        print(f'[{i}/{len(plans)}] {shlex.join(cmd)}', flush=True)
        result = run(cmd, logs/f'{i:02d}.log', args.timeout, e)
        result['compiler'] = e.get('RUDA_CUDA_COMPILER')
        if args.suite == 'oracle' and result.get('returncode') == 2: result['status']='blocked'
        report['results'].append(result); save()
        if result['status'] != 'passed': break
    passed = len(report['results']) == len(plans) and all(r['status']=='passed' for r in report['results'])
    status = 'passed' if passed else ('blocked' if report['results'][-1]['status']=='blocked' else 'failed_or_incomplete')
    report.update(status=status,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),performance_measured=passed and args.suite=='bench')
    save(); print(f'{status}: {path}')
    return 0 if passed else (2 if status=='blocked' else 1)

if __name__ == '__main__':
    raise SystemExit(main())
