#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run numerical-library checks with explicit host/device evidence and no auto-install.

rust-host compiles the actual library sources with rustc and no Cargo dependency
resolution. This is NOT the Python oracle. A full workspace build remains separate.
"""
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

ROOT = Path(__file__).resolve().parents[1]


def plans(suite: str, compiler: str, logs: Path, offline: bool = False):
    env = os.environ.copy()
    flags = ['--locked'] + (['--offline'] if offline else [])
    exe = '.exe' if os.name == 'nt' else ''
    if suite == 'oracle':
        return [], [([sys.executable, 'tools/science/oracle.py', '--output', str(logs/'oracle.json')], env)]
    if suite in ('rust-host', 'sparse-host'):
        commands = [(['rustc', '--version'], env)]
        common = ['rustc', '--edition=2024', '-Dunsafe_code']
        if suite == 'rust-host':
            # ruintegrate now uses rusolver's host LU for implicit Newton steps.
            # Compile each real library, then its tests and examples; do not
            # substitute a Python model for a Rust execution result.
            for package, name, examples in [
                ('ruSOLVER', 'rusolver', ['solver_demo', 'advanced_solver']),
                ('ruINTEGRATE', 'ruintegrate', ['integrate_demo', 'advanced_integrate']),
            ]:
                binary = str(logs/(name+'-tests'+exe))
                library = str(logs/('lib'+name+'.rlib'))
                dependencies = ['-L', 'dependency='+str(logs)]
                if name == 'ruintegrate':
                    dependencies += ['--extern', 'rusolver='+str(logs/'librusolver.rlib')]
                commands.extend([
                    (common+['--crate-name', name, '--crate-type=rlib', package+'/src/lib.rs',
                             *dependencies, '-o', library], env),
                    (common+['--crate-name', name, '--test', package+'/src/lib.rs',
                             *dependencies, '-o', binary], env),
                    ([binary, '--test-threads=1'], env),
                ])
                for example in examples:
                    demo = str(logs/(example+exe))
                    commands.extend([
                        (common+[package+'/examples/'+example+'.rs', '--extern', name+'='+library,
                                 *dependencies, '-o', demo], env),
                        ([demo], env),
                    ])
        else:
            sparse_lib = str(logs/'librusparse.rlib')
            solver_lib = str(logs/'librusolver.rlib')
            binary = str(logs/('sparse-tests'+exe))
            demo = str(logs/('sparse-poisson'+exe))
            commands.extend([
                (common+['--crate-name', 'rusparse', '--crate-type=rlib', 'ruSPARSE/src/lib.rs', '-o', sparse_lib], env),
                (common+['--crate-name', 'rusolver', '--cfg', 'feature="sparse"', '--extern', 'rusparse='+sparse_lib,
                         '--test', 'ruSOLVER/src/lib.rs', '-o', binary], env),
                ([binary, '--test-threads=1'], env),
                (common+['--crate-name', 'rusolver', '--cfg', 'feature="sparse"', '--extern', 'rusparse='+sparse_lib,
                         '--crate-type=rlib', 'ruSOLVER/src/lib.rs', '-o', solver_lib], env),
                (common+['ruSOLVER/examples/sparse_poisson.rs', '--extern', 'rusparse='+sparse_lib,
                         '--extern', 'rusolver='+solver_lib, '-L', 'dependency='+str(logs), '-o', demo], env),
                ([demo], env),
            ])
        return ['rustc'], commands
    if suite == 'host':
        return ['cargo', 'rustc'], [(['cargo','test',*flags,'-p','ruda-solver','-p','ruintegrate'], env),
            (['cargo','test',*flags,'-p','ruda-solver','--features','sparse'], env)]
    if suite == 'build':
        return ['cargo', 'rustc'], [(['cargo','check',*flags,'-p','ruda-solver','-p','ruintegrate','--all-targets'], env),
            (['cargo','check',*flags,'-p','ruda-solver','--features','sparse,tensor','--all-targets'], env),
            (['cargo','check',*flags,'-p','ruda-solver','--features','sparse,cuda,collective','--all-targets'], env),
            (['cargo','check',*flags,'-p','ruda-autodiff','--features','solver-host','--all-targets'],env)]
    commands = []
    for backend in (['nvrtc','ptx'] if compiler == 'both' else [compiler]):
        for target in ['device_cholesky','device_advanced']:
            commands.append((['cargo','test',*flags,'--release','-p','ruda-solver','--features','cuda',
                             '--test',target,'--','--test-threads=1'], {**env,'RUDA_CUDA_COMPILER':backend}))
    return ['cargo','rustc'], commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['oracle','rust-host','sparse-host','host','build','cuda'], default='rust-host')
    parser.add_argument('--compiler', choices=['nvrtc','ptx','both'], default='both')
    parser.add_argument('--timeout', type=float, default=300, help='Per-command cap, NOT an estimated runtime')
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be finite and positive')
    now = dt.datetime.now(dt.timezone.utc)
    logs = (args.log_dir or ROOT/'target'/'science'/now.strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
    required, commands = plans(args.suite,args.compiler,logs,args.offline)
    public = [{'command':cmd, 'compiler':env.get('RUDA_CUDA_COMPILER')} for cmd,env in commands]
    if args.dry_run:
        print(json.dumps({'status':'planned_only','plans':public},indent=2)); return 0
    if logs.exists() and any(logs.iterdir()):
        parser.error('choose an empty/new log directory; evidence is never overwritten')
    logs.mkdir(parents=True,exist_ok=True)
    result_file = logs/'results.json'
    sources = list((ROOT/'ruSOLVER').rglob('*.rs')) + list((ROOT/'ruINTEGRATE').rglob('*.rs'))
    sources += [ROOT/'Cargo.toml', ROOT/'Cargo.lock', ROOT/'ruSOLVER/Cargo.toml', ROOT/'ruINTEGRATE/Cargo.toml',
                ROOT/'tools/science/oracle.py', ROOT/'tools/science/fixtures.json', Path(__file__).resolve()]
    report = {'schema':'ruda.science.validation.v1','suite':args.suite,'started_utc':now.isoformat(),
              'status':'running','plans':public,'results':[],'performance_measured':False,
              'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    def save():
        temporary=result_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2)+'\n'); temporary.replace(result_file)
    missing=[name for name in required if shutil.which(name) is None]
    if missing:
        report.update(status='blocked',reason='Missing tools: '+', '.join(missing),finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save(); print(report['reason']); return 2
    save()
    for i,(cmd,env) in enumerate(commands):
        result=run(cmd,logs/f'{i+1:02}.log',args.timeout,env)
        if args.suite=='oracle' and result.get('returncode')==3:
            result['status']='blocked'
        report['results'].append(result); save()
        if result['status']!='passed': break
    passed=len(report['results'])==len(commands) and all(x['status']=='passed' for x in report['results'])
    report.update(status='passed' if passed else report['results'][-1]['status'],finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save(); print(f'{report["status"]}: {result_file}')
    return 0 if passed else 2 if report['status']=='blocked' else 1

if __name__=='__main__':
    raise SystemExit(main())
