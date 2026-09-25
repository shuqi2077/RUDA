#!/usr/bin/env python3
"""Compile and run the EXACT production topology/hazard validator with rustc, without Cargo/GPU.

Success certifies only this dependency-free module, NOT the native driver crate.
No Python or mocked-driver implementation is substituted when rustc is absent.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('v20-rust-host'))
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    if args.timeout <= 0: parser.error('timeout must be positive')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    report = {'production_module_compiled': False, 'tests_passed': False,
              'full_crate_compiled': False, 'gpu_validated': False, 'commands': [], 'errors': []}
    def save(): (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    compiler = shutil.which('rustc')
    if not compiler:
        report['errors'].append('missing rustc; production Rust was not compiled or run')
        save(); print(report['errors'][0], file=sys.stderr); return 2
    source = ROOT/'ruda-driver-cuda/src/execution/graph_topology.rs'
    binary = out/('graph_topology_tests.exe' if os.name=='nt' else 'graph_topology_tests')
    try:
        for label, command in [('compile', [compiler,'--edition=2024','--test',str(source),'-o',str(binary)]),
                               ('test', [str(binary),'--test-threads=1'])]:
            result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout, check=False)
            text = result.stdout + result.stderr
            (out/f'{label}.log').write_text(text)
            report['commands'].append({'command':command,'returncode':result.returncode})
            if result.returncode: raise RuntimeError(label+' failed')
            if label == 'compile': report['production_module_compiled'] = True
            else:
                match = re.search(r'test result: ok\. (\d+) passed; 0 failed; 0 ignored;', text)
                if not match or int(match.group(1)) < 29:
                    raise RuntimeError('missing retained Rust result (at least 29 tests, zero failed/ignored)')
                report['tests_passed'] = True
            save()
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        report['errors'].append(str(error)); save(); print(str(error),file=sys.stderr); return 1
if __name__ == '__main__': raise SystemExit(main())
