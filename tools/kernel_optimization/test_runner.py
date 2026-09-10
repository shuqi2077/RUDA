#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the verification plumbing and Python layout model, not GPU execution."""
import contextlib,io,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
TOOLS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(TOOLS));sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_kernel_optimization_regressions as runner
from run_safety_regressions import run
from oracle import Shared
class RunnerTests(unittest.TestCase):
    def test_both_compilers_and_original_baselines_are_in_cuda_plan(self):
        _,cmds=runner.plans('cuda','both',Path('/tmp/logs'))
        self.assertEqual(len(cmds),6);self.assertEqual({e['RUDA_CUDA_COMPILER']for _,e in cmds},{'nvrtc','ptx'})
        self.assertEqual(sum('device_warp'in c for c,_ in cmds),2)
    def test_sanitizer_builds_then_checks_memory_races_and_sync(self):
        required,cmds=runner.plans('sanitizer','ptx',Path('/tmp/logs'))
        self.assertIn('compute-sanitizer',required);self.assertIn('--no-run',cmds[0][0]);self.assertEqual(len(cmds),4)
        for c,_ in cmds[1:]:self.assertIn('--error-exitcode',c);self.assertIn('99',c);self.assertIn('--target-processes',c)
    def test_benchmark_preserves_kind_and_compiler_separation(self):
        _,cmds=runner.plans('bench','both',Path('/tmp/logs'),True,batch=65,order=7,rhs=3)
        self.assertEqual(len(cmds),4)
        self.assertEqual(len({c[-1]for c,_ in cmds}),4)
        for c,_ in cmds:self.assertIn('--offline',c);self.assertIn('65',c);self.assertIn('7',c);self.assertIn('3',c)
    def test_missing_rust_blocks_without_fake_results(self):
        with tempfile.TemporaryDirectory()as d,patch.object(runner.shutil,'which',return_value=None),patch.object(sys,'argv',['run','--suite','plan','--log-dir',d]):
            with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(runner.main(),2)
            report=json.loads((Path(d)/'results.json').read_text());self.assertEqual(report['status'],'blocked');self.assertEqual(report['results'],[]);self.assertFalse(report['performance_measured'])
    def test_dry_run_creates_no_results(self):
        with tempfile.TemporaryDirectory()as d:
            out=Path(d)/'none';result=subprocess.run([sys.executable,str(TOOLS/'run_kernel_optimization_regressions.py'),'--suite','cuda','--dry-run','--log-dir',str(out)],capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0);self.assertEqual(json.loads(result.stdout)['status'],'planned_only');self.assertFalse(out.exists())
    def test_invalid_timeout_or_shape_is_rejected(self):
        for args in[['--timeout','nan'],['--order','33'],['--samples','0']]:
            result=subprocess.run([sys.executable,str(TOOLS/'run_kernel_optimization_regressions.py'),*args],capture_output=True,text=True,timeout=5);self.assertEqual(result.returncode,2)
    def test_actual_subprocess_success_failure_timeout(self):
        with tempfile.TemporaryDirectory()as d:
            for code,status,timeout in[("print('ok')",'passed',2),("raise SystemExit(1)",'failed',2),("import time;time.sleep(5)",'timeout',.05)]:
                self.assertEqual(run([sys.executable,'-c',code],Path(d)/(status+'.log'),timeout,os.environ.copy())['status'],status)
    def test_model_detects_shared_handoff_without_barrier(self):
        a=Shared(4,True);a.write(0,0,1.);a.read(1,0)
        with self.assertRaises(AssertionError):a.barrier()
        a=Shared(4,True);a.write(0,0,1.);a.barrier();self.assertEqual(a.read(1,0),1.);a.barrier()
    def test_model_rejects_multiple_writers_and_allows_same_lane_dependency(self):
        a=Shared(4,True);a.write(0,0,1.);a.write(1,0,2.)
        with self.assertRaises(AssertionError):a.barrier()
        a=Shared(4,True);a.write(0,0,1.);a.write(0,0,a.read(0,0)+1.);a.barrier()
if __name__=='__main__':unittest.main()
