#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for evidence plumbing, not Rust/backend execution."""
import importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run_science_extended_regressions as runner
from run_safety_regressions import run

class RunnerTests(unittest.TestCase):
    def test_host_dependency_compiled_before_integrator(self):
        with tempfile.TemporaryDirectory()as d:
            req,cmds=runner.plans('rust-host','both',Path(d))
            integrator=[c for c,_ in cmds if 'ruINTEGRATE/src/lib.rs' in c]
            self.assertTrue(integrator)
            for c in integrator:self.assertIn('--extern',c);self.assertTrue(any(x.startswith('rusolver=')for x in c))
            self.assertTrue(any('ruSOLVER/examples/advanced_solver.rs'in c for c,_ in cmds))
            self.assertTrue(any('ruINTEGRATE/examples/advanced_integrate.rs'in c for c,_ in cmds))
    def test_device_plans_test_both_paths(self):
        _,cmds=runner.plans('cuda','both',Path('/tmp/logs'))
        self.assertEqual(len(cmds),4);self.assertEqual({env['RUDA_CUDA_COMPILER']for _,env in cmds},{'nvrtc','ptx'})
        self.assertEqual(sum('device_advanced'in c for c,_ in cmds),2)
    def test_autodiff_real_graph(self):
        _,cmds=runner.plans('autodiff','both',Path('/tmp/logs'),True)
        self.assertIn('ruda-autodiff',cmds[0][0]);self.assertIn('solver-host',cmds[0][0]);self.assertIn('--offline',cmds[0][0])
    def test_collective_build_then_actual_binary(self):
        _,cmds=runner.plans('collective','both',Path('/tmp/logs'),world=3,order=2)
        self.assertIn('build',cmds[0][0]);self.assertIn('--world',cmds[1][0]);self.assertIn('tools/science_extended/tcp_demo.py',cmds[1][0])
    def test_success_failure_and_timeout(self):
        with tempfile.TemporaryDirectory()as d:
            for script,status,timeout in[("print('ok')",'passed',2),("raise SystemExit(1)",'failed',2),("import time;time.sleep(5)",'timeout',.05)]:
                r=run([sys.executable,'-c',script],Path(d)/(status+'.log'),timeout,os.environ.copy());self.assertEqual(r['status'],status)
    def test_missing_tools_are_blocked(self):
        with tempfile.TemporaryDirectory()as d,patch.object(runner.shutil,'which',return_value=None),patch.object(sys,'argv',['runner','--suite','rust-host','--log-dir',d]):
            self.assertEqual(runner.main(),2);r=json.loads((Path(d)/'results.json').read_text());self.assertEqual(r['status'],'blocked');self.assertEqual(r['results'],[])
    def test_dry_run_does_not_create_logs(self):
        with tempfile.TemporaryDirectory()as d:
            out=Path(d)/'not-created';p=subprocess.run([sys.executable,str(ROOT/'run_science_extended_regressions.py'),'--suite','cuda','--dry-run','--log-dir',str(out)],capture_output=True,text=True,timeout=5)
            self.assertEqual(p.returncode,0);self.assertEqual(json.loads(p.stdout)['status'],'planned_only');self.assertFalse(out.exists())
    def test_bad_timeout_rejected(self):
        p=subprocess.run([sys.executable,str(ROOT/'run_science_extended_regressions.py'),'--timeout','nan'],capture_output=True,text=True,timeout=5);self.assertEqual(p.returncode,2)
if __name__=='__main__':unittest.main()
