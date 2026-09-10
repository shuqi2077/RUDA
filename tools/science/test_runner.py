# SPDX-License-Identifier: Apache-2.0
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TOOLS))
from run_science_regressions import plans
from run_safety_regressions import run

class RunnerTests(unittest.TestCase):
    def execute(self, code, timeout=5):
        with tempfile.TemporaryDirectory() as d:
            return run([sys.executable,'-c',code],Path(d)/'command.log',timeout,os.environ.copy())
    def test_success(self):
        self.assertEqual(self.execute('print("ok")')['status'],'passed')
    def test_nonzero_is_failure(self):
        self.assertEqual(self.execute('raise SystemExit(7)')['status'],'failed')
    def test_missing_command_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            r=run(['ruda-deliberately-nonexistent-tool'],Path(d)/'log',5,os.environ.copy())
            self.assertEqual(r['status'],'blocked')
    def test_timeout_is_not_a_pass(self):
        self.assertEqual(self.execute('import time;time.sleep(2)',.05)['status'],'timeout')
    def test_host_plan_uses_actual_rust_sources_and_no_oracle(self):
        tools,commands=plans('rust-host','both',Path('/tmp/test-ruda'))
        self.assertEqual(tools,['rustc'])
        flat=' '.join(' '.join(c) for c,e in commands)
        self.assertIn('ruSOLVER/src/lib.rs',flat)
        self.assertIn('ruINTEGRATE/src/lib.rs',flat)
        self.assertNotIn('oracle.py',flat)
    def test_cuda_plan_explicitly_tests_both_compilers(self):
        _,commands=plans('cuda','both',Path('/tmp/test-ruda'))
        self.assertEqual([e['RUDA_CUDA_COMPILER'] for _,e in commands],['nvrtc','nvrtc','ptx','ptx'])
        self.assertEqual([c[c.index('--test')+1] for c,_ in commands], ['device_cholesky','device_advanced','device_cholesky','device_advanced'])
    def test_offline_cargo_flag_is_before_test_arguments(self):
        _,commands=plans('cuda','ptx',Path('/tmp/test-ruda'),True)
        command=commands[0][0]
        self.assertLess(command.index('--offline'),command.index('--'))

if __name__=='__main__':
    unittest.main()
