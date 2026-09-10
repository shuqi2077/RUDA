#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Infrastructure tests only: do not pretend to compile or run Rust/CUDA."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT/'tools/run_gradient_guard_regressions.py'
sys.path.insert(0, str(ROOT/'tools'))
from run_safety_regressions import run

class RunnerTests(unittest.TestCase):
    def cli(self, *args, env=None):
        return subprocess.run([sys.executable, str(RUNNER), *args], text=True, capture_output=True, timeout=15, env=env)

    def test_dry_run_runs_old_and_new_tests_on_both_compilers(self):
        p = self.cli('--suite','cuda','--compiler','both','--dry-run')
        self.assertEqual(p.returncode, 0, p.stderr)
        d = json.loads(p.stdout)
        self.assertEqual(d['status'], 'planned_only')
        self.assertEqual(len(d['plans']), 4)
        self.assertEqual([x['compiler'] for x in d['plans']], ['nvrtc','nvrtc','ptx','ptx'])

    def test_each_suite_produces_a_plan(self):
        for suite in ('reference','host','build','cuda','bench','oracle'):
            p = self.cli('--suite',suite,'--offline','--dry-run')
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(json.loads(p.stdout)['plans'])

    def test_invalid_numbers_reject(self):
        for option, value in [('--timeout','nan'),('--timeout','inf'),('--timeout','0'),('--elements','0'),('--samples','2'),('--tensors','0')]:
            self.assertNotEqual(self.cli('--dry-run',option,value).returncode, 0)

    def test_missing_rust_is_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            env = {**os.environ, 'PATH':''}
            p = self.cli('--suite','reference','--log-dir',temp,env=env)
            self.assertEqual(p.returncode, 2, p.stdout+p.stderr)
            d = json.loads((Path(temp)/'results.json').read_text())
            self.assertEqual(d['status'], 'blocked')
            self.assertFalse(d['performance_measured'])
            self.assertEqual(d['results'], [])

    def test_log_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'results.json'; target.write_text('old')
            p = self.cli('--suite','reference','--log-dir',temp)
            self.assertNotEqual(p.returncode, 0)
            self.assertEqual(target.read_text(), 'old')

    def test_process_success_failure_and_timeout_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            cases = [('pass','pass',5,'passed'),('fail','raise SystemExit(7)',5,'failed'),('timeout','import time; time.sleep(30)',0.2,'timeout')]
            for name, code, timeout, expected in cases:
                r = run([sys.executable,'-c',code],Path(temp)/(name+'.log'),timeout,os.environ.copy())
                self.assertEqual(r['status'], expected)

if __name__ == '__main__':
    unittest.main()
