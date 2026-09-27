#!/usr/bin/env python3
"""Resumable Linux/NVIDIA direct-PTX validation; execute compiled tests, never Cargo under Sanitizer.

Exit 0: every requested case/benchmark passed. 1: failure. 2: setup error.
Exit 3: deliberately incomplete (--max-cases); resume with the same output directory.
Results from another source tree, binary, GPU UUID, driver or environment are not reused.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PARSER_VERSION = 1
SANITIZERS = ('memcheck', 'racecheck', 'synccheck', 'initcheck')
GROUPS = json.loads(Path(__file__).with_name('v22_cases.json').read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def source_hash(root: Path, output: Path) -> str:
    """Bound to actual sources and Cargo.lock, not just a claimed Git revision."""
    result = hashlib.sha256()
    excluded = {'.git', 'target', '__pycache__', '.pytest_cache'}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if excluded.intersection(rel.parts) or path.is_relative_to(output):
            continue
        if path.is_symlink():
            raise ValueError('source symlink needs explicit review: ' + str(rel))
        if path.is_file():
            result.update(str(rel).encode() + b'\0' + bytes.fromhex(sha(path)))
    return result.hexdigest()


def parse_success(text: str, returncode: int, name: str, marker: str,
                  sanitizer: str | None = None) -> None:
    """A clean tool summary alone cannot prove that a test was executed."""
    if returncode != 0:
        raise ValueError(f'process exit code {returncode}')
    summaries = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', text)
    if summaries != [('1', '0', '0')]:
        raise ValueError('expected exactly one successful, non-ignored Rust test')
    if not re.search(r'^test ' + re.escape(name) + r'\s+\.\.\.', text, re.M):
        raise ValueError('expected exact Rust test name is absent')
    if marker not in text:
        raise ValueError('missing required native runtime marker')
    if sanitizer:
        if sanitizer not in SANITIZERS:
            raise ValueError('unknown sanitizer')
        errors = re.findall(r'ERROR SUMMARY:\s*(\d+) errors', text)
        races = re.findall(r'RACECHECK SUMMARY:\s*(\d+) hazards? displayed\s*'
                           r'\((\d+) errors?,\s*(\d+) warnings?\)', text)
        if any(int(v) for v in errors) or any(any(int(v) for v in row) for row in races):
            raise ValueError('nonzero sanitizer error/hazard/warning summary')
        if sanitizer == 'racecheck':
            if not races:
                raise ValueError('missing complete Racecheck summary')
        elif not errors:
            raise ValueError('missing complete sanitizer error summary')


def list_names(text: str) -> set[str]:
    return set(re.findall(r'^(\S+): test$', text, re.M))


def parse_benchmarks(text: str, kind: str) -> list[dict]:
    prefix = 'RUDA_GRAPH_BALANCED_TIMING,' if kind == 'graph' else 'RUDA_FFT_FUSION_TIMING,'
    variant = 'graph' if kind == 'graph' else 'fused'
    dimensions = ('n', 'nodes') if kind == 'graph' else ('n', 'batch', 'inverse')
    records = {}
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        fields = dict(piece.split('=', 1) for piece in line[len(prefix):].split(','))
        key = tuple(fields[name] for name in dimensions)
        trial = int(fields['trial']); arm = fields[variant]
        elapsed = float(fields['elapsed_s']); repeats = int(fields['repeats'])
        if arm not in ('true', 'false') or trial not in range(7) or repeats <= 0 or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('invalid benchmark record')
        identity = (key, trial, arm)
        if identity in records:
            raise ValueError('duplicate benchmark record')
        records[identity] = (elapsed, repeats)
    expected_keys = ({(str(n),str(k)) for n,k in ((256,2),(256,16),(4096,16))} if kind == 'graph'
                     else {(str(n),str(b),mode) for n in (6,1009,2049,4097) for b in (1,16) for mode in ('true','false')})
    if {key for key,_,_ in records} != expected_keys:
        raise ValueError('incomplete/unexpected benchmark shapes')
    result = []
    for key in sorted(expected_keys):
        ratios = []
        for trial in range(7):
            try:
                old, old_repeats = records[key,trial,'false']
                new, new_repeats = records[key,trial,'true']
            except KeyError as exc:
                raise ValueError('incomplete paired benchmark trials') from exc
            if old_repeats != new_repeats:
                raise ValueError('unequal work between benchmark arms')
            ratios.append(old / new)
        result.append({**dict(zip(dimensions,key)), 'paired_trials':7,
                       'ratio_reference_over_candidate':{'median':statistics.median(ratios),
                           'min':min(ratios),'max':max(ratios),'raw':ratios},
                       'metric':'host submission plus synchronized GPU wall time; not model throughput'})
    return result


def probe() -> dict:
    if sys.platform != 'linux':
        raise RuntimeError('this strict profile currently targets Linux/NVIDIA')
    for tool in ('cargo','rustc'):
        if not shutil.which(tool):
            raise RuntimeError('missing ' + tool)
    if not re.fullmatch(r'\d+\.\d+', os.environ.get('RUDA_PTX_VERSION','')):
        raise RuntimeError('set RUDA_PTX_VERSION=major.minor supported by your driver')
    driver = ctypes.CDLL('libcuda.so.1')
    def checked(name, *args):
        status = getattr(driver,name)(*args)
        if status: raise RuntimeError(f'{name}: driver status {status}')
    checked('cuInit',0)
    count=ctypes.c_int(); checked('cuDeviceGetCount',ctypes.byref(count))
    if count.value < 1: raise RuntimeError('no visible NVIDIA device')
    device=ctypes.c_int(); checked('cuDeviceGet',ctypes.byref(device),0)
    version=ctypes.c_int(); checked('cuDriverGetVersion',ctypes.byref(version))
    name=ctypes.create_string_buffer(256); checked('cuDeviceGetName',name,256,device)
    uuid=(ctypes.c_ubyte*16)(); checked('cuDeviceGetUuid',ctypes.byref(uuid),device)
    result={'gpu_name':name.value.decode(errors='replace'),'gpu_uuid':bytes(uuid).hex(),
            'driver_api':version.value,'visible_count':count.value,
            'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),
            'cargo':subprocess.check_output(['cargo','--version'],text=True).strip()}
    # API version is not a driver build identifier. Include the installed build,
    # so a patch-driver upgrade cannot silently reuse old hardware evidence.
    if shutil.which('nvidia-smi'):
        result['driver_inventory']=subprocess.check_output(
            ['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    elif Path('/proc/driver/nvidia/version').exists():
        result['driver_inventory']=Path('/proc/driver/nvidia/version').read_text()
    else:
        raise RuntimeError('cannot identify exact driver build for resumable results')
    return result


def run(command: list[str], log: Path, env: dict, timeout: int) -> tuple[int,str]:
    """Kill the full child process group on timeout/cancel, not just Cargo."""
    with log.open('w') as stream:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
            raise
    return code, log.read_text(errors='replace')


def reusable(record: dict, binary: Path, log: Path, name: str, marker: str,
             sanitizer: str | None) -> bool:
    try:
        if record.get('status')!='passed' or record.get('binary_sha256')!=sha(binary) or record.get('log_sha256')!=sha(log):
            return False
        parse_success(log.read_text(errors='replace'),record['exit_code'],name,marker,sanitizer)
        return True
    except (OSError,ValueError,KeyError):
        return False


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('v22-hardware-validation'))
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--sanitizers',default='none',help='none, all, or comma-separated tool names')
    ap.add_argument('--groups',default=','.join(GROUPS))
    ap.add_argument('--max-cases',type=int,help='execute at most this many new case/tool jobs; incomplete exits 3')
    ap.add_argument('--benchmark',action='store_true')
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--timeout',type=int,default=1800,help='per command timeout in seconds')
    args=ap.parse_args(argv)
    selected=args.groups.split(',')
    tools=list(SANITIZERS) if args.sanitizers=='all' else ([] if args.sanitizers=='none' else args.sanitizers.split(','))
    if args.timeout<=0 or (args.max_cases is not None and args.max_cases<=0): ap.error('limits must be positive')
    if not selected or len(set(selected))!=len(selected) or any(g not in GROUPS for g in selected): ap.error('invalid/duplicate groups')
    if len(set(tools))!=len(tools) or any(t not in SANITIZERS for t in tools): ap.error('invalid/duplicate sanitizer tools')
    out=args.output.resolve()
    if ROOT==out or ROOT.is_relative_to(out): ap.error('output must not contain the source root')
    out.mkdir(parents=True,exist_ok=True)
    # Concurrent notebook cells must not overwrite one another's checkpoints.
    import fcntl
    lock=(out/'.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('another validation process owns this output directory',file=sys.stderr);return 2
    report={'version':'v22','status':'preflight','parser_version':PARSER_VERSION,
            'all_requested_passed':False,'builds':{},'cases':{},'benchmarks':{},'errors':[]}
    checkpoint=out/'result.json'
    # Do not overwrite even failed checkpoints without explicit resume.
    if checkpoint.exists() and not args.resume:
        print('output already has a checkpoint; use --resume or a new output directory',file=sys.stderr);return 2
    try:
        hardware=probe()
        if tools and not shutil.which('compute-sanitizer'): raise RuntimeError('Compute Sanitizer is missing')
        sanitizer_version=subprocess.check_output(['compute-sanitizer','--version'],text=True,stderr=subprocess.STDOUT).strip() if tools else None
        relevant={k:v for k,v in os.environ.items() if k.startswith(('RUDA_','CUDA_','CARGO_','CUDARC_','RUST_','RUSTUP_')) or k in ('RUSTFLAGS','RUSTC','RUSTC_WRAPPER','LD_LIBRARY_PATH')}
        identity={'source_sha256':source_hash(ROOT,out),'hardware':hardware,'sanitizer_version':sanitizer_version,
            'environment':relevant,'groups':selected,'sanitizers':tools,'benchmark':args.benchmark,
            'parser_version':PARSER_VERSION,'profile':'release'}
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        if checkpoint.exists():
            previous=json.loads(checkpoint.read_text())
            if previous.get('fingerprint')!=fingerprint:
                raise RuntimeError('source/GPU/driver/toolchain/options changed: use a NEW output directory; no old GPU results reused')
            report=previous;report['errors']=[]
        report.update(identity=identity,fingerprint=fingerprint,status='building',all_requested_passed=False)
        def save(): atomic_json(checkpoint,report)
        save()
        if args.preflight_only:
            report['status']='preflight_only';save();print('Preflight only: no Rust build or GPU test executed.');return 0
        (out/'logs').mkdir(exist_ok=True);(out/'artifacts').mkdir(exist_ok=True)
        env=dict(os.environ,RUDA_CUDA_COMPILER='ptx',RUDA_FFT_REQUIRE_CUDA='1',RUST_TEST_THREADS='1',CARGO_TERM_COLOR='never')
        binaries={}
        # Batch targets by package/features; do not recursively invoke v16..v21.
        batches={}
        for group in selected:
            cfg=GROUPS[group];batches.setdefault((cfg['package'],cfg['features']),[]).append(group)
        for (package,features), groups in batches.items():
            old=report['builds'].get(package,{})
            cached=old.get('status')=='passed'
            for group in groups:
                item=old.get('artifacts',{}).get(group,{})
                path=out/'artifacts'/item.get('sha256','missing')
                cached=cached and path.is_file() and sha(path)==item.get('sha256')
            if not cached:
                command=['cargo','test','--release','--locked','--no-default-features','--no-run','--message-format=json','-p',package,'--features',features]
                for group in groups: command+=['--test',GROUPS[group]['target']]
                log=out/'logs'/('build-'+package+'.log')
                report['builds'][package]={'status':'running','command':command,'log':str(log.relative_to(out))};save()
                code,text=run(command,log,env,args.timeout)
                if code: raise RuntimeError(f'{package} build failed; see {log}')
                emitted={}
                for line in text.splitlines():
                    try: obj=json.loads(line)
                    except ValueError: continue
                    if obj.get('reason')=='compiler-artifact' and obj.get('executable') and obj.get('profile',{}).get('test'):
                        emitted[obj['target']['name']]=Path(obj['executable'])
                artifacts={}
                for group in groups:
                    original=emitted.get(GROUPS[group]['target'])
                    if original is None or not original.is_file(): raise RuntimeError('no compiler-produced test executable for '+group)
                    digest=sha(original);path=out/'artifacts'/digest
                    shutil.copy2(original,path);path.chmod(0o755)
                    artifacts[group]={'sha256':digest,'original':str(original)}
                report['builds'][package]={'status':'passed','artifacts':artifacts,'command':command,'log':str(log.relative_to(out))};save()
            for group in groups: binaries[group]=out/'artifacts'/report['builds'][package]['artifacts'][group]['sha256']
        inventory={}
        for group in selected:
            texts=[]
            for ignored in (False,True):
                command=[str(binaries[group]),'--list','--format','terse']+(['--ignored'] if ignored else [])
                code,text=run(command,out/'logs'/f'list-{group}-{ignored}.log',env,args.timeout)
                if code: raise RuntimeError('cannot discover test inventory: '+group)
                texts.append(text)
            normal=list_names(texts[0])-list_names(texts[1])
            cases=sorted(n for n in normal if n.startswith(GROUPS[group]['prefix']))
            missing=set(GROUPS[group]['required'])-set(cases)
            if missing: raise RuntimeError(f'{group}: missing required tests {sorted(missing)}')
            inventory[group]=cases
        report['inventory']=inventory
        jobs=[(tool,g,n) for tool in [None,*tools] for g in selected for n in inventory[g]]
        report['planned_case_tool_jobs']=len(jobs);report['status']='running';save()
        new_count=0
        for tool,group,name in jobs:
            key='|'.join([tool or 'plain',group,name]);digest=hashlib.sha256(key.encode()).hexdigest()[:20]
            log=out/'logs'/f'{digest}.log';previous=report['cases'].get(key,{})
            marker=GROUPS[group]['marker']
            if reusable(previous,binaries[group],log,name,marker,tool): continue
            if args.max_cases is not None and new_count>=args.max_cases: break
            command=[str(binaries[group]),'--exact',name,'--nocapture','--test-threads=1']
            if tool: command=['compute-sanitizer','--tool',tool,'--target-processes','all','--error-exitcode','86',*command]
            record={'status':'running','command':command,'name':name,'group':group,'sanitizer':tool,
                    'binary_sha256':sha(binaries[group]),'log':str(log.relative_to(out))}
            report['cases'][key]=record;save();started=time.monotonic()
            try:
                code,text=run(command,log,env,args.timeout);record['exit_code']=code
                parse_success(text,code,name,marker,tool);record['status']='passed'
            except (ValueError,subprocess.TimeoutExpired,OSError) as exc:
                record.update(status='failed',reason=str(exc))
            finally:
                record['seconds']=time.monotonic()-started
                if log.exists(): record['log_sha256']=sha(log)
                save()
            new_count+=1
            print(f'{tool or "plain"} {group} {name}: {record["status"]}',flush=True)
        current=[report['cases'].get('|'.join([t or 'plain',g,n]),{}) for t,g,n in jobs]
        passed=sum(r.get('status')=='passed' for r in current)
        failed=sum(r.get('status')=='failed' for r in current)
        # Jobs with stale/corrupted logs are NOT counted as passed at a chunk boundary.
        passed=0
        for tool,g,name in jobs:
            key='|'.join([tool or 'plain',g,name]);r=report['cases'].get(key,{})
            log=out/'logs'/(hashlib.sha256(key.encode()).hexdigest()[:20]+'.log')
            if reusable(r,binaries[g],log,name,GROUPS[g]['marker'],tool): passed+=1
        report['counts']={'passed':passed,'failed':failed,'pending':len(jobs)-passed-failed,'total':len(jobs)}
        complete=passed==len(jobs)
        if args.benchmark and complete:
            for kind,group,name in [('graph','graph-replay','graph_replay_balanced_benchmark'),('fft','fft-exact','suite::exact::exact_fft_spectrum_fusion_benchmark')]:
                if group not in selected: continue
                log=out/'logs'/f'benchmark-{kind}.log'
                command=[str(binaries[group]),'--exact',name,'--ignored','--nocapture','--test-threads=1']
                r=report['benchmarks'].get(kind,{})
                if not reusable(r,binaries[group],log,name,GROUPS[group]['marker'],None):
                    code,text=run(command,log,env,args.timeout)
                    parse_success(text,code,name,GROUPS[group]['marker'])
                    r={'status':'passed','exit_code':code,'command':command,'binary_sha256':sha(binaries[group]),'log_sha256':sha(log),'log':str(log.relative_to(out))}
                r['comparisons']=parse_benchmarks(log.read_text(),kind)
                report['benchmarks'][kind]=r;save()
        if source_hash(ROOT,out)!=identity['source_sha256']: raise RuntimeError('source changed during validation; evidence is invalid')
        report['all_requested_passed']=complete and failed==0
        report['status']='passed' if report['all_requested_passed'] else ('failed' if failed else 'incomplete')
        save();return 0 if report['all_requested_passed'] else (1 if failed else 3)
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError,KeyboardInterrupt) as exc:
        # A mismatched resume must never overwrite the original evidence.
        if 'fingerprint' in report:
            report['all_requested_passed']=False;report['status']='blocked';report['errors'].append(str(exc))
            atomic_json(checkpoint,report)
        else:
            atomic_json(out/'preflight-failure.json',{'status':'blocked','errors':[str(exc)],'gpu_executed':False})
        print(str(exc),file=sys.stderr);return 2

if __name__=='__main__': raise SystemExit(main())
