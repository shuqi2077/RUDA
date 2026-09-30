#!/usr/bin/env python3
"""Compile and execute the PRODUCTION host plan/inverse index modules, no GPU. Includes v34 visibility bounds.

Requires rustc. No alternate implementation is substituted when it is missing.
"""
import argparse,json,shutil,subprocess,sys,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():p.error('use a fresh output directory')
    result={'version':'v34','rust_host_executed':False,'tests_executed':0,'gpu_executed':False,'errors':[]}
    try:
        rustc=shutil.which('rustc')
        if rustc is None:raise RuntimeError('rustc unavailable; production Rust NOT executed')
        src=ROOT/'ruDNN/src/paged_attention'
        harness=out/'harness.rs'
        def literal(path):return json.dumps(str(path))
        harness.write_text('#[derive(Debug,PartialEq,Eq)] pub struct PagedAttentionError(pub &\'static str);\n'
            +f'#[path={literal(src/"plan.rs")}] mod plan;\npub use plan::HostPlan;\n'
            +f'#[path={literal(src/"history_index.rs")}] mod history_index;\n')
        exe=out/'tests';commands=[[rustc,'--edition=2024','--test',str(harness),'-o',str(exe)],[str(exe),'--nocapture']]
        for i,command in enumerate(commands):
            proc=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
            (out/f'command-{i}.log').write_text(proc.stdout)
            if proc.returncode:raise RuntimeError(f'production Rust command {i} failed: {proc.returncode}')
        if not re.search(r'test result: ok\. 12 passed; 0 failed; 0 ignored;',proc.stdout):
            raise RuntimeError('expected exactly 12 executed production host tests')
        result['source_sha256']={name:hashlib.sha256((src/name).read_bytes()).hexdigest()
                                for name in ('plan.rs','history_index.rs')}
        result['tests_executed']=12
        result['rust_host_executed']=True
    except Exception as e:result['errors'].append(str(e))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['rust_host_executed'] else 2
if __name__=='__main__':raise SystemExit(main())
