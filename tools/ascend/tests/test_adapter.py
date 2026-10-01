"""Host structural/failure tests. These do not compile/execute the Rust kernel."""
from __future__ import annotations
import importlib.util,json,struct,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('ascend_build',ROOT/'tools/ascend/build_deepgemm.py')
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)

def test_production_no_longer_contains_original_cpp_device_headers():
    assert not (ROOT/'ruda-driver-cann/vendor/deepgemm-ascend').exists()
    assert not list((ROOT/'ruda-ascend-kernels').rglob('*.hpp'))
    for p in (ROOT/'ruda-ascend-kernels/src').glob('*.rs'):
        if p.name=='tests.rs':continue
        for line in p.read_text().splitlines():
            if line.lstrip().startswith('//!') or line.lstrip().startswith('//'):continue
            assert '#include <deep_gemm/' not in line
            assert 'bf16_gemm_impl' not in line

def test_kernel_is_structured_rust_not_cpp_template():
    ir=(ROOT/'ruda-ascend-kernels/src/ir.rs').read_text()
    core=(ROOT/'ruda-ascend-kernels/src/kernel.rs').read_text()
    assert 'Raw(' not in ir and 'Cpp(' not in ir
    for op in ['Op::GlobalToL1','Op::L1ToL0','Op::Mmad','Op::Store','scheduler::persistent']:
        assert op in core
    assert 'b.for_("kb"' in core and 'b.for_("ki"' in core
    assert 'asc_mmad(' not in core  # Target spellings belong to lowering, not algorithms.

def test_runtime_refuses_the_old_cpp_schema():
    src=(ROOT/'ruda-driver-cann/src/tensor/deepgemm/artifact.rs').read_text()
    assert 'ruda.ascend.rust.bf16.v2' in src
    assert 'ruda.deepgemm.bf16.v1' not in src
    assert 'source_language' in src and 'rust-device-ir' in src

def test_provenance_and_license_retained():
    meta=json.loads((ROOT/'ruda-ascend-kernels/UPSTREAM.json').read_text())
    assert meta['archive_commit_comment']=='8491bbb4b8c02a094a2318965f50c70438a3e73c'
    assert 'Permission is hereby granted' in (ROOT/'ruda-ascend-kernels/LICENSE').read_text()

def test_no_rustc_cannot_emit_even_source_only(monkeypatch,tmp_path):
    monkeypatch.setattr(b.shutil,'which',lambda _:None)
    out=tmp_path/'new'
    with pytest.raises(ValueError,match='missing rustc'):b.emit(out)
    assert not list(tmp_path.rglob('kernel.ruda'))
    assert not list(tmp_path.rglob('kernel.asc'))

def test_output_is_never_overwritten(tmp_path):
    out=tmp_path/'existing';out.mkdir();(out/'sentinel').write_text('old')
    with pytest.raises(ValueError,match='overwrite'):b.emit(out)
    with pytest.raises(ValueError,match='overwrite'):b.build(out,tmp_path/'sdk')
    assert (out/'sentinel').read_text()=='old'

def test_missing_sdk_cannot_publish(tmp_path):
    with pytest.raises(ValueError,match='missing executable'):b.build(tmp_path/'out',tmp_path/'sdk')
    assert not list(tmp_path.rglob('kernel.ruda'))

def elf(names,etype=2):
    strings=b'\0';offsets=[]
    for n in names:offsets.append(len(strings));strings+=n.encode()+b'\0'
    shoff=64;count=len(names)+2;start=64+64*count
    data=bytearray(start)+bytearray(strings);data[:6]=b'\x7fELF\x02\x01'
    struct.pack_into('<H',data,16,etype);struct.pack_into('<Q',data,40,shoff);struct.pack_into('<HHH',data,58,64,count,1)
    struct.pack_into('<IIQQQQIIQQ',data,shoff+64,0,3,0,0,start,len(strings),0,0,1,0)
    for i,o in enumerate(offsets,2):struct.pack_into('<I',data,shoff+64*i,o)
    return bytes(data)

def test_elf_extracts_untruncated_name():
    name='ruda_rust_'+'kernel'*80;assert b.kernel_name(elf(['.text','.ascend.meta.'+name]))==name

@pytest.mark.parametrize('data',[b'',b'\x7fELF',elf(['.text']),elf(['.ascend.meta.a','.ascend.meta.b']),elf(['.ascend.meta.a'],1),elf(['.ascend.meta.'])])
def test_invalid_elf(data):
    with pytest.raises(ValueError):b.kernel_name(data)

@pytest.mark.parametrize('ta',[False,True])
@pytest.mark.parametrize('tb',[False,True])
@pytest.mark.parametrize('batch',[1,3])
def test_independent_global_address_formula(ta,tb,batch):
    m,n,k=32,64,48;rng=np.random.default_rng(12)
    a=rng.normal(size=(batch,k,m) if ta else (batch,m,k)).astype(np.float32)
    bb=rng.normal(size=(batch,n,k) if tb else (batch,k,n)).astype(np.float32)
    ai=np.arange(m)[:,None]+np.arange(k)[None,:]*m if ta else np.arange(m)[:,None]*k+np.arange(k)[None,:]
    bi=np.arange(n)[:,None]*k+np.arange(k)[None,:] if tb else np.arange(n)[:,None]+np.arange(k)[None,:]*n
    for i in range(batch):
        aa=a.ravel()[i*m*k+ai];br=bb.ravel()[i*n*k+bi]
        ref=(a[i].T if ta else a[i])@(bb[i].T if tb else bb[i])
        np.testing.assert_allclose(aa@br.T,ref,rtol=1e-4,atol=1e-4)

def test_linear_gradient_identity_reference():
    rng=np.random.default_rng(9);x=rng.normal(size=(32,48));w=rng.normal(size=(64,48));dy=rng.normal(size=(32,64));dx=dy@w
    for p,q in [(0,0),(8,3),(31,47)]:
        xx=x.copy();xx[p,q]+=1e-5
        assert abs(np.sum(((xx@w.T)-(x@w.T))*dy)/1e-5-dx[p,q])<1e-6

def test_standalone_rust_has_no_host_device_runtime_dependencies():
    text=(ROOT/'ruda-ascend-kernels/Cargo.toml').read_text()
    assert text.split('[dependencies]')[1].strip()==''
    assert '#![forbid(unsafe_code)]' in (ROOT/'ruda-ascend-kernels/src/lib.rs').read_text()
