#!/usr/bin/env python3
"""Read-only, scoped SOURCE audit. Lexical evidence is not compilation or GPU acceptance."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[2]
SCOPES={
    'adapters':['ruda-torch/src','ruda-torch/python/ruda_torch','ruLLM/src'],
    'native_drivers':['ruda-driver-cuda/src','ruda-driver-hip/src'],
    'loaders':['ruLLM/src'],
}
def scan(root, scope, pattern):
    result=[]; count=0;regex=re.compile(pattern)
    for directory in scope:
        for file in sorted((root/directory).rglob('*')):
            if not file.is_file() or file.suffix not in {'.rs','.cpp','.h','.py'}:continue
            count+=1
            for line,text in enumerate(file.read_text(errors='replace').splitlines(),1):
                if regex.search(text):result.append({'path':file.relative_to(root).as_posix(),'line':line,'text':text.strip()[:240]})
    return {'scopes':scope,'source_files_scanned':count,'pattern':pattern,'matches':result,
            'interpretation':'references only; no match is scoped absence, not a proof about all implementations'}
def excerpt(root,path,pattern):
    p=root/path;rows=[]
    for line,text in enumerate(p.read_text().splitlines(),1):
        if re.search(pattern,text):rows.append({'line':line,'text':text.strip()})
    return {'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'matches':rows}
def audit(root):
    report={'version':'v21','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'source_audit_only':True,'native_rust_compiled_by_this_tool':False,'gpu_validated_by_this_tool':False}
    report['graph_references_in_adapters']=scan(root,SCOPES['adapters'],r'\bCudaGraph\b|\bcuGraphLaunch\b|\bgraph_build\b|\bgraph_replay\b')
    report['named_deepseek_kimi_references']=scan(root,SCOPES['loaders'],r'(?i)deepseek|kimi')
    report['runtime_capture_managed_ipc_references']=scan(root,SCOPES['native_drivers'],r'\b(?:cuStreamBeginCapture\w*|cuStreamEndCapture\w*|cuMemAllocManaged|cuMemPrefetchAsync\w*|cuIpc(?:Get|Open|Close)\w*|hipGraph\w*)\b')
    report['single_device_evidence']=excerpt(root,'ruda-torch/python/ruda_torch/csrc/backend.cpp',r'only ruda:0|deviceCount\(\).*return 1')
    report['inference_only_evidence']=excerpt(root,'ruda-torch/python/ruda_torch/csrc/backend.cpp',r'paged attention is inference-only|MLA is inference-only')
    report['ptx_debug_evidence']=excerpt(root,'ruda-compiler/src/ptx/emit.rs',r'debug_symbols|unsupported\("debug symbols"\)')
    report['named_loaders']=excerpt(root,'ruLLM/src/huggingface.rs',r'pub fn load_huggingface_|pub use.*load_huggingface')
    report['host_staged_evidence']=excerpt(root,'ruda-communication/src/data_service.rs',r'B::float_into_data')
    report['nccl_evidence']=excerpt(root,'ruda-driver-cuda/src/execution/server.rs',r'cudarc::nccl::result::')
    report['tools']={tool:shutil.which(tool) for tool in ['rustc','cargo','nvidia-smi','compute-sanitizer']}
    try:
        ctypes.CDLL('nvcuda.dll' if os.name=='nt' else 'libcuda.so.1')
        report['driver_library_loadable']=True
    except OSError as exc:
        report['driver_library_loadable']=False;report['driver_library_error']=str(exc)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=Path('v21-source-audit.json'))
    args=parser.parse_args();root=args.root.resolve()
    if not (root/'Cargo.toml').is_file():parser.error('root must be the full RUDA source tree')
    report=audit(root);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Source-only audit written; it is not GPU acceptance:',args.output)
    return 0
if __name__=='__main__':raise SystemExit(main())
