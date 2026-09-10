#!/usr/bin/env python3
"""Explicit stack-autotune validation tiers. Missing tools are NOT_RUN, never PASS.

No automatic installation, hardware reset, model download, or deployment changes.
Reference checks are independent Python models/source guards; core compiles the
actual production Rust engine with std only. GPU tiers run real ignored tests.
"""
from __future__ import annotations
import argparse, json, os, platform, re, shlex, shutil, signal, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SUITES=('reference','core','runtime','feature-compile','gpu-nvidia','gpu-amd')

def run(name, command, output, timeout, min_tests=0):
    record={'name':name,'command':command,'log':f'{name}.log'}
    logfile=output/record['log']
    if shutil.which(command[0]) is None:
        record.update(status='NOT_RUN',reason=f'executable unavailable: {command[0]}')
        logfile.write_text(record['reason']+'\n'); return record
    started=time.monotonic()
    with logfile.open('w',encoding='utf-8') as stream:
        stream.write('$ '+shlex.join(command)+'\n'); stream.flush()
        try:
            child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=(os.name=='posix'))
            try:
                code=child.wait(timeout=timeout)
                record.update(status='PASS' if code==0 else 'FAIL',returncode=code)
            except subprocess.TimeoutExpired:
                if os.name=='posix': os.killpg(child.pid,signal.SIGKILL)
                else: child.kill()
                child.wait(); record.update(status='TIMEOUT',reason=f'command exceeded configured {timeout} seconds')
        except OSError as error: record.update(status='FAIL',reason=str(error))
    if record['status']=='PASS' and min_tests:
        counts=[int(x) for x in re.findall(r'running (\d+) tests?',logfile.read_text(errors='replace'))]
        if not counts or sum(counts)<min_tests:
            record.update(status='FAIL',reason='expected Rust tests were not discovered',test_counts=counts)
    record['elapsed_seconds']=round(time.monotonic()-started,3)
    return record

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',action='append',choices=SUITES)
    parser.add_argument('--output',type=Path,default=ROOT/'validation/stack-autotune-v3')
    parser.add_argument('--timeout',type=int,default=1800)
    args=parser.parse_args()
    if args.timeout<=0: parser.error('--timeout must be positive')
    suites=list(dict.fromkeys(args.suite or ['reference','core']))
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    checks=[]
    def execute(name,command,min_tests=0):
        result=run(name,command,output,args.timeout,min_tests);checks.append(result);print(f"{name}: {result['status']}",flush=True);return result
    if 'reference' in suites:
        execute('stack-python-reference',[sys.executable,'tools/stack_autotune/reference_checks.py'])
        execute('v1-python-reference',[sys.executable,'ruLLM/tools/verify_reference.py'])
        execute('v2-python-reference',[sys.executable,'ruLLM/tools/verify_runtime_reference.py'])
    if 'core' in suites:
        with tempfile.TemporaryDirectory(prefix='ruda-autotune-') as directory:
            binary=str(Path(directory)/('tests.exe' if os.name=='nt' else 'tests'))
            result=execute('rust-std-core-compile',['rustc','--edition=2024','--test','tools/stack_autotune/host_tests.rs','-o',binary])
            if result['status']=='PASS':execute('rust-std-core-tests',[binary,'--test-threads=2'],min_tests=60)
            else:checks.append({'name':'rust-std-core-tests','status':'NOT_RUN','reason':'core compilation did not pass'})
    if 'runtime' in suites:
        execute('cargo-runtime-integration',['cargo','test','--locked','-p','ruda','--lib','--test','stack_autotune'],min_tests=1)
        execute('legacy-runtime-tests',['cargo','test','--locked','-p','ruda','--test','runtime'],min_tests=1)
    if 'feature-compile' in suites:
        execute('fusion-stack-features',['cargo','check','--locked','-p','ruda-fusion','--no-default-features','--features','device-stack-autotune'])
        execute('fusion-with-legacy-checks',['cargo','check','--locked','-p','ruda-fusion','--no-default-features','--features','device-stack-autotune,device-autotune-checks'])
        execute('rullm-stack-host',['cargo','check','--locked','-p','ruda-llm','--features','stack-autotune','--lib'])
        for backend in ('nvidia','amd','nvidia,amd'):
            execute('stack-'+backend.replace(',','-'),['cargo','check','--locked','-p','ruda-llm','--features','stack-autotune,'+backend,'--lib','--examples','--tests'])
        execute('legacy-no-std',['cargo','check','--locked','-p','ruda','--no-default-features','--features','runtime'])
    for backend in ('nvidia','amd'):
        if 'gpu-'+backend in suites:
            execute('gpu-'+backend,['cargo','test','--locked','-p','ruda-llm','--features','stack-autotune,'+backend,
                '--test','stack_autotune_gpu',backend+'::','--','--ignored','--test-threads=1'],min_tests=3)
    report={'schema':1,'utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),
        'python':platform.python_version(),'suites_requested':suites,'checks':checks,
        'all_requested_passed':all(c['status']=='PASS' for c in checks),
        'note':'Python references/source guards do not execute Rust. No GPU speedup can be inferred from these checks. Unrequested suites have not been tested by this run.'}
    (output/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"report: {output/'validation.json'}")
    return 1 if any(c['status'] in ('FAIL','TIMEOUT') for c in checks) else (0 if report['all_requested_passed'] else 3)
if __name__=='__main__':sys.exit(main())
