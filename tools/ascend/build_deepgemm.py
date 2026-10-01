#!/usr/bin/env python3
"""Build Rust-authored Ascend kernels. Never include/link a DeepGEMM C++ body.

Rust IR -> generated CANN-intrinsic CCE source -> Bisheng -> linked code object.
The .asc file is a target intermediate. This is NOT direct rustc-to-NPU ISA.
Missing rustc or SDK is an error, including for --emit-only; no fake emitter.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,struct,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CRATE=ROOT/'ruda-ascend-kernels'
SCHEMA='ruda.ascend.rust.bf16.v2'

def run(cmd:list[str],log:Path,timeout:int=600):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
    log.write_text(p.stdout)
    if p.returncode:raise RuntimeError(f'command failed ({p.returncode}); see {log}')
    return p.stdout

def source_digest():
    h=hashlib.sha256()
    for p in sorted(CRATE.rglob('*')):
        if p.is_file() and (p.suffix=='.rs' or p.name=='Cargo.toml'):
            h.update(p.relative_to(CRATE).as_posix().encode()+b'\0');h.update(p.read_bytes())
    return h.hexdigest()

def rust_emitter(directory:Path)->Path:
    compiler=shutil.which('rustc')
    if not compiler:raise ValueError('missing rustc: the device algorithm must be generated from actual Rust source')
    directory.mkdir(parents=True,exist_ok=True)
    lib=directory/'libruda_ascend_kernels.rlib';exe=directory/'ruda-ascend-emit'
    run([compiler,'--edition=2024','--crate-name','ruda_ascend_kernels','--crate-type','rlib',
         '-C','opt-level=2',str(CRATE/'src/lib.rs'),'-o',str(lib)],directory/'rust-library.log')
    run([compiler,'--edition=2024','-C','opt-level=2',str(CRATE/'src/main.rs'),
         '--extern',f'ruda_ascend_kernels={lib}','-o',str(exe)],directory/'rust-emitter.log')
    return exe

def emit(out:Path,only:list[str]|None=None):
    if out.exists():raise ValueError('refusing to overwrite output; use a fresh directory')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ruda-rust-compiler-',dir=out.parent) as temp:
        tmp=Path(temp);exe=rust_emitter(tmp)
        cmd=[str(exe),'--out',str(out)]
        for key in only or []:cmd+=['--only',key]
        run(cmd,tmp/'emit.log')
        # Preserve actual Rust compilation records with the generated sources.
        for log in tmp.glob('*.log'):shutil.copy2(log,out/log.name)
    generated=sorted(p.parent.name for p in out.glob('*/kernel.asc'))
    if not generated:raise RuntimeError('Rust generator produced no kernels')
    for p in out.glob('*/kernel.asc'):
        text=p.read_text()
        if '#include <deep_gemm/' in text or 'bf16_gemm_impl' in text or 'asc_mmad(' not in text:
            raise RuntimeError('unexpected foreign kernel or missing device matrix operation')
    (out/'emit-info.json').write_text(json.dumps({'status':'source_only','kernels':generated,
        'rust_source_sha256':source_digest(),'target':'dav-c310','soc':'Ascend950DT',
        'compiled_for_npu':False,'source_language':'rust-device-ir','lowering':'cann-c-intrinsics'},indent=2)+'\n')
    return generated

def kernel_name(data:bytes)->str:
    if len(data)<64 or data[:6]!=b'\x7fELF\x02\x01':raise ValueError('expected little-endian ELF64')
    if struct.unpack_from('<H',data,16)[0]!=2:raise ValueError('expected linked ET_EXEC, not relocatable input')
    shoff=struct.unpack_from('<Q',data,40)[0];size,count,idx=struct.unpack_from('<HHH',data,58)
    if size!=64 or not count or idx>=count or shoff+64*count>len(data):raise ValueError('invalid section table')
    strings=struct.unpack_from('<IIQQQQIIQQ',data,shoff+64*idx);start,end=strings[4],strings[4]+strings[5]
    if end>len(data):raise ValueError('invalid string table')
    names=data[start:end];found=[]
    for i in range(count):
        off=struct.unpack_from('<I',data,shoff+64*i)[0]
        if off>=len(names):raise ValueError('name out of bounds')
        term=names.find(b'\0',off)
        if term<0:raise ValueError('unterminated name')
        name=names[off:term].decode('ascii')
        if name.startswith('.ascend.meta.'):found.append(name[len('.ascend.meta.'):])
    if len(found)!=1 or not found[0]:raise ValueError('expected exactly one kernel metadata section')
    return found[0]

def build(out:Path,toolkit:Path,only:list[str]|None=None):
    if out.exists():raise ValueError('refusing to overwrite artifact root')
    cc=toolkit/'bin/bisheng';ld=toolkit/'bin/ld.lld'
    for exe in (cc,ld):
        if not exe.is_file() or not os.access(exe,os.X_OK):raise ValueError(f'missing executable: {exe}')
    includes=[toolkit/'aarch64-linux/asc/include',toolkit/'aarch64-linux/asc/include/adv_api']
    if any(not p.is_dir() for p in includes):raise ValueError('missing installed CANN intrinsic headers')
    out.mkdir(parents=True)
    generated=out/'.rust-generated'
    keys=emit(generated,only)
    compiler_id=hashlib.sha256(cc.read_bytes()+ld.read_bytes()).hexdigest()
    for key in keys:
        temp=Path(tempfile.mkdtemp(prefix='.building-',dir=out))
        src=temp/'kernel.asc';rel=temp/'kernel.rel.o';obj=temp/'kernel.o'
        shutil.copy2(generated/key/'kernel.asc',src)
        shutil.copy2(generated/key/'kernel.rust-ir.txt',temp/'kernel.rust-ir.txt')
        cmd=[str(cc),'-x','cce','-std=c++20','-O2','--cce-aicore-only',
             '--cce-aicore-arch=dav-c310','--cce-disable-vf-stack-reserved-ubuf']
        for inc in includes:cmd+=['-I',str(inc)]
        cmd+=['-c',str(src),'-o',str(rel)]
        link=[str(ld),'-m','aicorelinux','-Ttext','0','--no-mmap-output-file',str(rel),'-o',str(obj)]
        (temp/'commands.json').write_text(json.dumps([cmd,link],indent=2)+'\n')
        run(cmd,temp/'compile.log');run(link,temp/'link.log')
        data=obj.read_bytes();name=kernel_name(data)
        if name!=f'ruda_rust_{key}':raise ValueError(f'unexpected compiled kernel entry: {name}')
        values={'schema':SCHEMA,'arch':'dav-c310','soc':'Ascend950DT','cores':'32','key':key,
            'object':'kernel.o','kernel_name':name,'sha256':hashlib.sha256(data).hexdigest(),
            'source_sha256':hashlib.sha256(src.read_bytes()+source_digest().encode()).hexdigest(),
            'compiler_sha256':compiler_id,'source_language':'rust-device-ir','lowering':'cann-c-intrinsics'}
        (temp/'kernel.ruda').write_text(''.join(f'{k}={v}\n' for k,v in values.items()))
        temp.rename(out/key)  # Only publish a loadable contract after successful linking.

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--emit-only',action='store_true')
    p.add_argument('--toolkit',type=Path);p.add_argument('--only',action='append',default=[])
    a=p.parse_args()
    try:
        if a.emit_only:emit(a.out.resolve(),a.only)
        else:
            env=os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_TOOLKIT_HOME')
            toolkit=a.toolkit or (Path(env) if env else None)
            if toolkit is None:raise ValueError('set ASCEND_HOME_PATH or --toolkit')
            build(a.out.resolve(),toolkit.resolve(),a.only)
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as e:p.exit(1,f'{e}\n')
if __name__=='__main__':main()
