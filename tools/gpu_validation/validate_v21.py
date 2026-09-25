#!/usr/bin/env python3
"""Strict Rust/direct-PTX inferred-dependency graph/hazard acceptance; never substitutes a host model."""
from __future__ import annotations
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('v21_previous_validator', Path(__file__).with_name('validate_v20.py'))
PREVIOUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIOUS)
BASE = PREVIOUS.BASE
REQUIRED = {'graph_infer_retains_inputs_after_caller_drop', 'graph_infer_fork_join_correct', 'graph_infer_wrong_queue_is_refused', 'graph_infer_tracked_completion', 'graph_infer_build_does_not_execute_kernels', 'graph_infer_empty_is_refused', 'graph_infer_does_not_weaken_explicit_dag_rejection', 'graph_infer_scalar_update_preserves_dependencies', 'graph_infer_writes_reduce_to_chain', 'graph_infer_read_before_overwrite', 'graph_infer_partial_overlap_preserves_order', 'graph_infer_disjoint_views_stay_independent'}

def require_infer_success(text: str) -> int:
    count = BASE.require_test_success(text, len(REQUIRED))
    names = set(re.findall(r'test (graph_infer_\w+)\s+\.\.\. ok', text))
    if missing := REQUIRED - names:
        raise ValueError('missing required graph-infer cases: ' + ', '.join(sorted(missing)))
    if 'RUDA_GRAPH_INFER_RUNTIME,CudaRuntime,direct-ptx' not in text:
        raise ValueError('no native CUDA/direct-PTX runtime identity')
    return count

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('v21-hardware-validation'))
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--sanitizer', choices=['memcheck','racecheck','synccheck','initcheck'])
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    if args.timeout <= 0: parser.error('timeout must be positive')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    report = {'version':'v21','gpu_validated':False,'rust_compiled':False,
              'commands':[],'errors':[],'default_async_changed':False}
    def save():
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    for tool in ('cargo','rustc'):
        if not shutil.which(tool): report['errors'].append('missing '+tool)
    if not re.fullmatch(r'\d+\.\d+', os.environ.get('RUDA_PTX_VERSION','')):
        report['errors'].append('set RUDA_PTX_VERSION=major.minor for the real device/driver')
    if args.sanitizer and not shutil.which('compute-sanitizer'):
        report['errors'].append('requested Compute Sanitizer is missing')
    try:
        report['driver'] = BASE.probe_driver()
        if report['driver']['driver_api_version'] < 12000:
            raise RuntimeError('this acceptance profile requires Driver API >= 12.0')
        PREVIOUS.PREVIOUS.PREVIOUS.PREVIOUS.probe_graph_symbols()
        driver = ctypes.CDLL('nvcuda.dll' if os.name=='nt' else 'libcuda.so.1')
        for name in ('cuGraphExecKernelNodeSetParams_v2','cuStreamQuery','cuEventCreate','cuEventRecord',
                     'cuEventQuery','cuEventSynchronize','cuEventDestroy_v2'):
            getattr(driver,name)
    except (OSError,RuntimeError,AttributeError) as exc:
        report['errors'].append(str(exc))
    save()
    if report['errors']:
        print('Preflight failed: '+'; '.join(report['errors']),file=sys.stderr); return 2
    if args.preflight_only:
        print('Preflight passed; neither Rust compilation nor GPU correctness has been validated.'); return 0
    env = dict(os.environ, RUDA_CUDA_COMPILER='ptx')
    def run(command):
        log = output/f'{len(report["commands"]):02d}.log'
        record = {'command':command,'log':log.name}; report['commands'].append(record); save()
        with log.open('w') as stream:
            try:
                result = subprocess.run(command,cwd=ROOT,env=env,stdout=stream,
                    stderr=subprocess.STDOUT,timeout=args.timeout,check=False)
                record['returncode']=result.returncode
            except subprocess.TimeoutExpired as exc:
                record['timed_out']=True; save(); raise RuntimeError('required command timed out: '+log.name) from exc
        save()
        if result.returncode: raise RuntimeError('required command failed: '+log.name)
        return log.read_text(errors='replace')
    try:
        report['rustc_version']=run(['rustc','--version'])
        common=['cargo','test','--release','--locked','-p','ruda-driver-cuda',
                '--no-default-features','--features','std,direct-ptx']
        # Run v20's complete strict profile as well, not just new named cases.
        run([sys.executable, str(ROOT/'tools/gpu_validation/validate_v20.py'),
             '--output', str(output/'retained-v20'), '--timeout', str(args.timeout)])
        BASE.require_test_success(run([*common,'--lib','execution::graph_topology::tests',
            '--','--test-threads=1']),52)
        run(['cargo','check','--release','--locked','-p','ruda-driver-cuda',
             '--no-default-features','--features','std,direct-ptx','--example','native-graph-infer'])
        run([*common,'--test','graph-infer','--no-run'])
        report['rust_compiled']=True;save()
        new=[*common,'--test','graph-infer','--','--test-threads=1','--nocapture','--skip','graph_infer_benchmark']
        report['new_gpu_tests']=require_infer_success(run(new))
        if args.sanitizer:
            text=run(['compute-sanitizer','--tool',args.sanitizer,'--target-processes','all','--error-exitcode','86',*new])
            require_infer_success(text)
            if not re.search(r'ERROR SUMMARY: 0 errors|RACECHECK SUMMARY: 0 hazards',text):
                raise RuntimeError('no clean instrumented sanitizer summary')
        if args.benchmark:
            text=run([*common,'--test','graph-infer','graph_infer_benchmark','--','--ignored','--exact','--nocapture','--test-threads=1'])
            BASE.require_test_success(text,1)
            lines=[line for line in text.splitlines() if line.startswith('RUDA_GRAPH_INFER_TIMING,')]
            if len(lines)!=3: raise RuntimeError('incomplete same-device benchmark results')
            report['benchmark_lines']=lines
        report['gpu_validated']=True;save();return 0
    except (OSError,ValueError,RuntimeError) as exc:
        report['errors'].append(str(exc));save();print(str(exc),file=sys.stderr);return 1
if __name__=='__main__': raise SystemExit(main())
