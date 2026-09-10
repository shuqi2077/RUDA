#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_muon_regressions as runner
from run_safety_regressions import run

class RunnerTests(unittest.TestCase):
    def test_cpu_plan_is_scoped(self):
        tools, p = runner.plans('host', 'both', Path('/tmp/muon'), True)
        self.assertEqual(tools, ['cargo', 'rustc'])
        self.assertIn('--offline', p[0][0])
        self.assertIn('optim::muon::', p[0][0])
        self.assertNotIn('--all-features', p[0][0])
    def test_cuda_paths_remain_separate(self):
        _, p = runner.plans('cuda', 'both', Path('/tmp/muon'))
        self.assertEqual([env['RUDA_CUDA_COMPILER'] for _,env in p], ['nvrtc','ptx'])
        self.assertTrue(all('test-cuda,ruda-driver-cuda/direct-ptx' in cmd for cmd,_ in p))
    def test_reference_is_not_a_ruda_test(self):
        tools,p=runner.plans('reference','both',Path('/tmp/muon'))
        self.assertEqual(tools,['rustc'])
        self.assertEqual(p[0][0][0],'rustc')
    def test_build_includes_no_std_and_example(self):
        _,p=runner.plans('build','both',Path('/tmp/muon'))
        self.assertTrue(any('--no-default-features' in cmd for cmd,_ in p))
        self.assertTrue(any('muon-training' in cmd for cmd,_ in p))
    def test_real_subprocess_success(self):
        with tempfile.TemporaryDirectory() as d:
            r=run([sys.executable,'-c','print("ok")'],Path(d)/'ok.log',5,os.environ.copy())
            self.assertEqual(r['status'],'passed')
    def test_real_subprocess_failure(self):
        with tempfile.TemporaryDirectory() as d:
            r=run([sys.executable,'-c','raise SystemExit(3)'],Path(d)/'fail.log',5,os.environ.copy())
            self.assertEqual(r['status'],'failed')
            self.assertEqual(r['returncode'],3)
    def test_real_subprocess_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            r=run([sys.executable,'-c','import time;time.sleep(10)'],Path(d)/'timeout.log',.05,os.environ.copy())
            self.assertEqual(r['status'],'timeout')
if __name__=='__main__':unittest.main()
