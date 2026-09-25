#!/usr/bin/env python3
"""Strict Rust/direct-PTX graph update acceptance; never substitutes a host model."""
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
SPEC = importlib.util.spec_from_file_location('v18_previous_validator', Path(__file__).with_name('validate_v17.py'))
PREVIOUS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIOUS)
BASE = PREVIOUS.BASE
REQUIRED = {
    'graph_update_scalars_without_execution',
    'graph_update_inflight_launches_keep_old_scalars',
    'graph_update_noop_still_replays_once',
    'graph_update_mixed_scalar_types_preserve_packing',
    'graph_update_rejects_pointer_rebinding_then_remains_usable',
    'graph_update_rejects_shape_metadata_then_remains_usable',
    'graph_update_rejects_grid_and_kernel_changes',
    'graph_update_rejects_wrong_queue_and_node_index',
    'graph_update_only_selected_node_changes',
    'graph_update_query_try_close_and_closed_errors',
    'graph_update_repeated_buffers_are_retained',
}

def require_update_success(text: str) -> int:
    count = BASE.require_test_success(text, len(REQUIRED))
    names = set(re.findall(r'test (graph_update_\w+)\s+\.\.\.', text))
    if missing := REQUIRED - names:
        raise ValueError('missing required graph-update cases: ' + ', '.join(sorted(missing)))
    if 'RUDA_GRAPH_UPDATE_RUNTIME,CudaRuntime,direct-ptx' not in text:
        raise ValueError('no native CUDA/direct-PTX runtime identity')
    return count

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('v18-hardware-validation'))
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--sanitizer', choices=['memcheck','racecheck','synccheck','initcheck'])
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    if args.timeout <= 0: parser.error('timeout must be positive')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    report = {'version':'v18','gpu_validated':False,'rust_compiled':False,
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
        PREVIOUS.probe_graph_symbols()
        driver = ctypes.CDLL('nvcuda.dll' if os.name=='nt' else 'libcuda.so.1')
        for name in ('cuGraphExecKernelNodeSetParams_v2','cuStreamQuery'):
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
        for selector,minimum in [('execution::graph_update::tests',10),
                ('execution::context::launch::tests',2),('graph::tests',2)]:
            BASE.require_test_success(run([*common,'--lib',selector,'--','--test-threads=1']),minimum)
        report['rust_compiled']=True;save()
        old=[*common,'--test','graph-replay','--','--test-threads=1','--nocapture','--skip','graph_replay_benchmark']
        report['retained_gpu_tests']=PREVIOUS.require_graph_success(run(old))
        new=[*common,'--test','graph-update','--','--test-threads=1','--nocapture','--skip','graph_update_benchmark']
        report['new_gpu_tests']=require_update_success(run(new))
        if args.sanitizer:
            text=run(['compute-sanitizer','--tool',args.sanitizer,'--target-processes','all','--error-exitcode','86',*new])
            require_update_success(text)
            if not re.search(r'ERROR SUMMARY: 0 errors|RACECHECK SUMMARY: 0 hazards',text):
                raise RuntimeError('no clean instrumented sanitizer summary')
        if args.benchmark:
            text=run([*common,'--test','graph-update','graph_update_benchmark','--','--ignored','--exact','--nocapture','--test-threads=1'])
            BASE.require_test_success(text,1)
            lines=[line for line in text.splitlines() if line.startswith('RUDA_GRAPH_UPDATE_TIMING,')]
            if len(lines)!=3: raise RuntimeError('incomplete same-device benchmark results')
            report['benchmark_lines']=lines
        report['gpu_validated']=True;save();return 0
    except (OSError,ValueError,RuntimeError) as exc:
        report['errors'].append(str(exc));save();print(str(exc),file=sys.stderr);return 1
if __name__=='__main__': raise SystemExit(main())
