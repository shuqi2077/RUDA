"""Host-only validation protocol tests. No production Rust/GPU execution."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('validator_v22',Path(__file__).with_name('validate_v22.py'))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
NAME='suite::exact::example'
MARKER='EXACT_FFT_RUNTIME,ruda_driver_cuda::runtime::CudaRuntime'
GOOD=f'\nrunning 1 test\ntest {NAME} ... {MARKER}\nok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 33 filtered out; finished in 0.74s\n'

class ParserTests(unittest.TestCase):
    def test_plain_pass(self): v.parse_success(GOOD,0,NAME,MARKER)
    def test_runtime_marker_rejection_is_not_numerical_failure(self):
        with self.assertRaisesRegex(ValueError,'runtime marker'): v.parse_success(GOOD.replace(MARKER,''),0,NAME,MARKER)
    def test_nonzero_exit(self):
        with self.assertRaises(ValueError): v.parse_success(GOOD,86,NAME,MARKER)
    def test_wrong_name(self):
        with self.assertRaises(ValueError): v.parse_success(GOOD,0,'another',MARKER)
    def test_zero_executed(self):
        with self.assertRaises(ValueError): v.parse_success(GOOD.replace('1 passed','0 passed'),0,NAME,MARKER)
    def test_ignored_is_not_pass(self):
        with self.assertRaises(ValueError): v.parse_success(GOOD.replace('0 ignored','1 ignored'),0,NAME,MARKER)
    def test_no_combined_process_counts(self):
        with self.assertRaises(ValueError): v.parse_success(GOOD+GOOD,0,NAME,MARKER)
    def test_each_memory_tool(self):
        for tool in ('memcheck','synccheck','initcheck'):
            v.parse_success(GOOD+'========= ERROR SUMMARY: 0 errors\n',0,NAME,MARKER,tool)
    def test_real_race_summary(self):
        v.parse_success(GOOD+'========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n',0,NAME,MARKER,'racecheck')
    def test_race_warnings_rejected(self):
        with self.assertRaises(ValueError): v.parse_success(GOOD+'RACECHECK SUMMARY: 1 hazard displayed (0 errors, 1 warning)',0,NAME,MARKER,'racecheck')
    def test_earlier_nonzero_summary_rejected(self):
        with self.assertRaises(ValueError): v.parse_success(GOOD+'ERROR SUMMARY: 4 errors\nERROR SUMMARY: 0 errors',0,NAME,MARKER,'memcheck')
    def test_missing_sanitizer_summary(self):
        with self.assertRaises(ValueError): v.parse_success(GOOD,0,NAME,MARKER,'memcheck')
    def test_zero_error_does_not_substitute_race_report(self):
        with self.assertRaises(ValueError): v.parse_success(GOOD+'ERROR SUMMARY: 0 errors',0,NAME,MARKER,'racecheck')
    def test_listing(self):
        self.assertEqual(v.list_names('a: test\nb: benchmark\nc: test\n2 tests'),{'a','c'})


def graph_bench():
    return '\n'.join(f'RUDA_GRAPH_BALANCED_TIMING,n={n},nodes={k},trial={t},graph={arm},repeats=1000,elapsed_s={elapsed}'
         for n,k in [(256,2),(256,16),(4096,16)] for t in range(7)
         for arm,elapsed in [('false',2.0),('true',1.0)])

class BenchmarkTests(unittest.TestCase):
    def test_full_pairs(self):
        results=v.parse_benchmarks(graph_bench(),'graph');self.assertEqual(len(results),3)
        self.assertTrue(all(r['ratio_reference_over_candidate']['median']==2.0 for r in results))
    def test_single_observation_refused(self):
        with self.assertRaises(ValueError): v.parse_benchmarks('\n'.join(graph_bench().splitlines()[:2]),'graph')
    def test_missing_arm(self):
        with self.assertRaises(ValueError): v.parse_benchmarks('\n'.join(graph_bench().splitlines()[1:]),'graph')
    def test_duplicate(self):
        with self.assertRaises(ValueError): v.parse_benchmarks(graph_bench()+'\n'+graph_bench().splitlines()[0],'graph')
    def test_nan(self):
        with self.assertRaises(ValueError): v.parse_benchmarks(graph_bench().replace('elapsed_s=1.0','elapsed_s=nan',1),'graph')
    def test_different_work(self):
        with self.assertRaises(ValueError): v.parse_benchmarks(graph_bench().replace('repeats=1000','repeats=999',1),'graph')
    def test_all_fft_dimensions(self):
        text='\n'.join(f'RUDA_FFT_FUSION_TIMING,n={n},batch={b},inverse={m},trial={t},fused={arm},repeats=30,elapsed_s=1,retained_bytes=2048'
             for n in (6,1009,2049,4097) for b in (1,16) for m in ('true','false') for t in range(7) for arm in ('false','true'))
        self.assertEqual(len(v.parse_benchmarks(text,'fft')),16)

class CheckpointTests(unittest.TestCase):
    def test_exact_artifact_and_log_required(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);b=p/'binary';log=p/'case.log';b.write_bytes(b'not a GPU executable; hash fixture');log.write_text(GOOD)
            r={'status':'passed','binary_sha256':v.sha(b),'log_sha256':v.sha(log),'exit_code':0}
            self.assertTrue(v.reusable(r,b,log,NAME,MARKER,None))
            b.write_bytes(b'changed');self.assertFalse(v.reusable(r,b,log,NAME,MARKER,None))
    def test_modified_log_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);b=p/'binary';log=p/'case.log';b.write_bytes(b'fixture');log.write_text(GOOD)
            r={'status':'passed','binary_sha256':v.sha(b),'log_sha256':v.sha(log),'exit_code':0}
            log.write_text(GOOD+'changed');self.assertFalse(v.reusable(r,b,log,NAME,MARKER,None))
    def test_no_reuse_running_or_failed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);b=p/'binary';log=p/'case.log';b.write_bytes(b'fixture');log.write_text(GOOD)
            for status in ('running','failed','pending'):
                r={'status':status,'binary_sha256':v.sha(b),'log_sha256':v.sha(log),'exit_code':0}
                self.assertFalse(v.reusable(r,b,log,NAME,MARKER,None))
    def test_source_and_lock_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);out=p/'results';out.mkdir();(p/'Cargo.lock').write_text('old');(p/'a.rs').write_text('fn a() {}')
            first=v.source_hash(p,out);(out/'log').write_text('runtime result');self.assertEqual(first,v.source_hash(p,out))
            (p/'target').mkdir();(p/'target'/'artifact').write_text('build');self.assertEqual(first,v.source_hash(p,out))
            (p/'Cargo.lock').write_text('new');self.assertNotEqual(first,v.source_hash(p,out))
    def test_reject_source_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a').write_text('data');(p/'link').symlink_to(p/'a')
            with self.assertRaises(ValueError): v.source_hash(p,p/'out')
    def test_atomic_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'checkpoint.json';v.atomic_json(p,{'status':'incomplete'});self.assertEqual(json.loads(p.read_text())['status'],'incomplete')
    def test_actual_host_process_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(subprocess.TimeoutExpired):
                v.run([sys.executable,'-c','import time; time.sleep(30)'],Path(td)/'timeout.log',dict(os.environ),1)
    def test_case_inventory_includes_new_and_old(self):
        self.assertEqual(sum(len(cfg['required']) for cfg in v.GROUPS.values()),87)
        names=v.GROUPS['fft-exact']['required']
        self.assertIn('suite::exact::exact_fft_tensor_api_preserves_legacy_shape',names)
        self.assertIn('suite::exact::exact_fft_fused_four_step_batched',names)


class RunnerStateTests(unittest.TestCase):
    """Simulated process results test only the checkpoint state machine."""
    def exercise(self, operation):
        from unittest.mock import patch
        import contextlib, io
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);root=base/'src';root.mkdir();out=base/'results'
            (root/'Cargo.lock').write_text('fixture lock')
            artifact=base/'fixture-binary';artifact.write_bytes(b'host fixture, NOT executable GPU code')
            groups={'graph-replay':{'package':'fixture','features':'none','target':'graph-replay',
                'prefix':'graph_','marker':MARKER,'required':['graph_one','graph_two']}}
            calls={'builds':0,'tests':0};fail=set()
            def process(command,log,env,timeout):
                if command[0]=='cargo':
                    calls['builds']+=1
                    text=json.dumps({'reason':'compiler-artifact','profile':{'test':True},
                        'target':{'name':'graph-replay'},'executable':str(artifact)})+'\n'
                    code=0
                elif '--list' in command:
                    text='ignored_benchmark: test\n' if '--ignored' in command else 'graph_one: test\ngraph_two: test\nignored_benchmark: test\n'
                    code=0
                else:
                    calls['tests']+=1;name=command[command.index('--exact')+1]
                    text=GOOD.replace(NAME,name);code=0
                    if command[0]=='compute-sanitizer':
                        tool=command[command.index('--tool')+1]
                        text+=('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors\n')
                        if (tool,name) in fail: code=86
                log.write_text(text);return code,text
            args=['--output',str(out),'--groups','graph-replay','--sanitizers','all']
            with patch.object(v,'ROOT',root),patch.object(v,'GROUPS',groups),patch.object(v,'probe',return_value={'gpu_uuid':'HOST-FIXTURE-NOT-GPU'}),patch.object(v,'run',side_effect=process),patch.object(v.shutil,'which',return_value='/fixture/tool'),patch.object(v.subprocess,'check_output',return_value='fixture version'),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                operation(root,out,args,calls,fail)
    def test_chunk_then_resume_executes_remaining_and_builds_once(self):
        def check(root,out,args,calls,fail):
            self.assertEqual(v.main(args+['--max-cases','3']),3)
            r=json.loads((out/'result.json').read_text());self.assertEqual(r['counts'],{'passed':3,'failed':0,'pending':7,'total':10})
            self.assertFalse(r['all_requested_passed'])
            self.assertEqual(v.main(args+['--resume']),0)
            self.assertEqual(calls,{'builds':1,'tests':10})
            self.assertTrue(json.loads((out/'result.json').read_text())['all_requested_passed'])
        self.exercise(check)
    def test_changed_source_does_not_destroy_previous_evidence(self):
        def check(root,out,args,calls,fail):
            self.assertEqual(v.main(args),0);old=(out/'result.json').read_bytes()
            (root/'changed.rs').write_text('changed source')
            self.assertEqual(v.main(args+['--resume']),2)
            self.assertEqual((out/'result.json').read_bytes(),old)
            self.assertEqual(calls['tests'],10)
        self.exercise(check)
    def test_corrupted_passed_log_is_rerun(self):
        def check(root,out,args,calls,fail):
            self.assertEqual(v.main(args),0)
            r=json.loads((out/'result.json').read_text());record=next(iter(r['cases'].values()))
            (out/record['log']).write_text('corrupted')
            self.assertEqual(v.main(args+['--resume']),0)
            self.assertEqual(calls,{'builds':1,'tests':11})
        self.exercise(check)
    def test_failed_job_is_not_complete_and_is_retried(self):
        def check(root,out,args,calls,fail):
            fail.add(('memcheck','graph_one'));self.assertEqual(v.main(args),1)
            r=json.loads((out/'result.json').read_text());self.assertEqual(r['counts']['failed'],1);self.assertFalse(r['all_requested_passed'])
            fail.clear();self.assertEqual(v.main(args+['--resume']),0)
            self.assertEqual(calls,{'builds':1,'tests':11})
        self.exercise(check)

if __name__=='__main__': unittest.main()
