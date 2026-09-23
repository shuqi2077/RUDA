"""CPU references and actual Python API tests. NOT Rust/PTX/GPU execution.

The NumPy model follows the partition/online-merge algorithm independently of
production Rust; dense PyTorch attention is the numerical oracle.
"""
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from test_v14_host import modules, schedule, dense

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[3]

def ranges(length, splits):
    base,extra=divmod(length,splits)
    return [(s*base+min(s,extra),s*base+min(s,extra)+base+(s<extra)) for s in range(splits)]

def split_reference(q,k,v,tables,lengths,ids,positions,scale,causal,splits,qp=None,kp=None,workspace=None):
    q,k,v=(x.float().numpy() for x in (q,k,v))
    if qp is not None:qp,kp=qp.float().numpy(),kp.float().numpy()
    rows,heads,_=q.shape;dv=v.shape[-1];ps=k.shape[1]
    if workspace is None:workspace=np.full((rows,heads,splits,dv+2),np.nan,dtype=np.float32)
    lookups=0
    for row,seq in enumerate(ids):
        visible=min(lengths[seq],positions[row]+1) if causal else lengths[seq]
        for h in range(heads):
            kh=h//(heads//k.shape[2])
            for part,(begin,end) in enumerate(ranges(visible,splits)):
                acc=np.zeros(dv,dtype=np.float32);m=np.float32(0);den=np.float32(0);token=begin
                while token<end:
                    logical,offset=divmod(token,ps);page=tables[seq][logical];lookups+=1
                    stop=token+min(ps-offset,end-token)
                    while token<stop:
                        score=np.dot(q[row,h],k[page,offset,kh])
                        if qp is not None:score+=np.dot(qp[row,h],kp[page,offset,kh])
                        score=np.float32(score*scale)
                        nxt=np.maximum(m,score) if token!=begin else score
                        a=np.float32(np.exp(m-nxt)) if token!=begin else np.float32(0)
                        b=np.float32(np.exp(score-nxt));den=np.float32(den*a+b)
                        acc=acc*a+v[page,offset,kh]*b;m=nxt;token+=1;offset+=1
                workspace[row,h,part,:dv]=acc
                workspace[row,h,part,dv:]=(m,den)
    out=np.zeros((rows,heads,dv),dtype=np.float32)
    for row in range(rows):
        for h in range(heads):
            m=np.float32(0);den=np.float32(0);acc=out[row,h]
            for part in range(splits):
                vec=workspace[row,h,part];part_den=vec[dv+1]
                if part_den!=0:
                    nxt=np.maximum(m,vec[dv]) if den!=0 else vec[dv]
                    a=np.float32(np.exp(m-nxt)) if den!=0 else np.float32(0)
                    b=np.float32(np.exp(vec[dv]-nxt))
                    acc[:]=acc*a+vec[:dv]*b;den=np.float32(den*a+part_den*b);m=nxt
            if den!=0:acc[:]/=den
    return torch.from_numpy(out),workspace,lookups


def data(dtype,mla=False,page_size=7):
    g=torch.Generator().manual_seed(105)
    d,dv,h,kh=(512,512,4,1) if mla else (33,17,4,2)
    lengths=[25,9,0];counts=[math.ceil(x/page_size) for x in lengths];pages=sum(counts)
    order=torch.randperm(pages,generator=g).tolist();tables=[order[:counts[0]],order[counts[0]:],[]]
    ids=[0,1,0,2];positions=[13,8,24,0]
    def rand(shape):return torch.randn(shape,generator=g).to(dtype)
    q=rand((4,h,d));k=rand((pages,page_size,kh,d));v=k if mla else rand((pages,page_size,kh,dv))
    qp=rand((4,h,64)) if mla else None;kp=rand((pages,page_size,1,64)) if mla else None
    return q,k,v,tables,lengths,ids,positions,192**-.5 if mla else d**-.5,qp,kp

@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('splits',[1,2,3,8,32])
@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('mla',[False,True])
def test_split_math_against_dense(dtype,splits,causal,mla):
    q,k,v,tables,lengths,ids,pos,scale,qp,kp=data(dtype,mla)
    actual,parts,_=split_reference(q,k,v,tables,lengths,ids,pos,scale,causal,splits,qp,kp)
    expect=dense(q,k,v,tables,lengths,ids,pos,scale,causal,qp,kp)
    torch.testing.assert_close(actual,expect,rtol=8e-5,atol=4e-6)
    assert parts.dtype==np.float32 and np.isfinite(parts).all()
    assert not np.any(parts[-1]) and torch.count_nonzero(actual[-1])==0

@pytest.mark.parametrize('length',[0,1,2,7,31,32,33,4096,2**32-1])
@pytest.mark.parametrize('splits',[1,2,3,8,31,32])
def test_partition_u32_bounds_no_gaps(length,splits):
    pieces=ranges(length,splits)
    assert pieces[0][0]==0 and pieces[-1][1]==length
    assert all(a<=b<=2**32-1 for a,b in pieces)
    assert all(pieces[i][1]==pieces[i+1][0] for i in range(splits-1))
    sizes=[b-a for a,b in pieces];assert max(sizes)-min(sizes)<=1

@pytest.mark.parametrize('splits',[2,3,8,32])
@pytest.mark.parametrize('page_size',[1,3,7,16])
def test_non_page_aligned_partitions(splits,page_size):
    q,k,v,tables,lengths,ids,pos,scale,qp,kp=data(torch.float32,page_size=page_size)
    out,_,_=split_reference(q,k,v,tables,lengths,ids,pos,scale,True,splits)
    ref=dense(q,k,v,tables,lengths,ids,pos,scale,True)
    torch.testing.assert_close(out,ref,rtol=1e-4,atol=4e-6)

@pytest.mark.parametrize('splits',[2,8,32])
def test_workspace_overwritten_and_reused(splits):
    q,k,v,tables,lengths,ids,pos,scale,_,_=data(torch.float16)
    _,scratch,_=split_reference(q,k,v,tables,lengths,ids,pos,scale,True,splits)
    storage=scratch.ctypes.data
    scratch[:]=np.nan
    out,scratch2,_=split_reference(q,k,v,tables,[1,0,0],ids,[0,0,0,0],scale,True,splits,workspace=scratch)
    assert scratch2.ctypes.data==storage and np.isfinite(scratch).all()
    ref=dense(q,k,v,tables,[1,0,0],ids,[0,0,0,0],scale,True)
    torch.testing.assert_close(out,ref)

@pytest.mark.parametrize('splits',[2,3,32])
def test_large_scores_are_rescaled(splits):
    q=torch.ones(1,1,1);k=torch.tensor([1000.,-1000.,999.,0.,1001.,-500.]).reshape(2,3,1,1)
    v=torch.arange(6.).reshape(2,3,1,1)
    out,parts,_=split_reference(q,k,v,[[1,0]],[6],[0],[5],1.0,True,splits)
    ref=dense(q,k,v,[[1,0]],[6],[0],[5],1.0,True)
    assert torch.isfinite(out).all() and np.isfinite(parts).all()
    torch.testing.assert_close(out,ref,rtol=2e-5,atol=2e-6)

@pytest.mark.parametrize('bad',[0,33,-1,True,1.25,'8',2**32])
def test_real_plan_rejects_invalid_splits_before_native(bad):
    n,p,_=modules()
    with pytest.raises((ValueError,TypeError)):p.PagedAttentionPlan(**schedule(),splits=bad)
    assert not n.plans

@pytest.mark.parametrize('splits',[1,2,8,32])
def test_real_plan_spec_and_workspace_size(splits):
    _,p,_=modules();plan=p.PagedAttentionPlan(**schedule(),splits=splits)
    assert plan._spec[-1]==splits
    assert plan.workspace_bytes(32,128)==(0 if splits==1 else 4*32*splits*130*4)


def test_workspace_budget_fails_instead_of_cpu_fallback():
    n,p,_=modules();plan=p.PagedAttentionPlan(**schedule(),splits=32)
    with pytest.raises(ValueError,match='64 MiB'):plan.workspace_bytes(4096,1024)
    assert not n.plans


def test_page_table_read_is_per_page_not_per_token():
    q=torch.zeros(1,1,8);k=torch.zeros(8,16,1,8);v=torch.ones_like(k)
    _,_,lookups=split_reference(q,k,v,[list(range(8))],[128],[0],[127],1.0,True,1)
    assert lookups==8  # algorithmic count, NOT a GPU memory transaction measurement


def test_source_has_shared_kernel_and_queue_private_scratch():
    text=(ROOT/'ruDNN/src/paged_attention/kernel.rs').read_text()
    assert 'fn attention<F: Float, O: Float>' in text
    assert 'while token < stop' in text and 'if token!=begin' in text
    assert 'fn merge<F: Float>' in text and 'if part_den != 0.0' in text
    native=(ROOT/'ruda-torch/src/paged.rs').read_text()
    assert 'Mutex<Option<SplitWorkspace<CudaRuntime>>>' in native
    assert 'LAUNCHES.fetch_add(2,' in native
    assert 'workspace.as_ref().map(|w| w.matches' in native


def test_append_cow_decision_precedes_handle_moves():
    text=(ROOT/'ruDNN/src/paged_attention/mod.rs').read_text()
    body=text.split('pub fn append_with_report',1)[1]
    assert 'key_guard' not in body and 'value_guard' not in body
    assert body.index('let value_mutable=v.can_mut()')<body.index('let k=if key_mutable')
    assert body.index('if size==0')<body.index('let key_mutable=')
    # Source contract only; real handle counts are checked by required Rust GPU tests.


def test_abi_nine_six_word_plan_header():
    cpp=(ROOT/'ruda-torch/python/ruda_torch/csrc/backend.cpp').read_text()
    native=(ROOT/'ruda-torch/src/paged.rs').read_text()
    assert 'spec.size()==6' in cpp and 'std::slice::from_raw_parts(spec,6)' in native
    assert 'm.attr("abi_version") = 9' in cpp


def load_gate():
    import importlib.util
    path=ROOT/'ruda-torch/tools/validate_v15_gpu.py'
    spec=importlib.util.spec_from_file_location('v15_gate_for_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize('text',[
 'test result: ok. 0 passed; 0 failed; 0 ignored;',
 'test result: ok. 6 passed; 0 failed; 1 ignored;',
 'test result: FAILED. 5 passed; 1 failed; 0 ignored;',
 'test result: ok. 4 passed; 0 failed; 0 ignored;',
 '',
])
def test_gpu_gate_rejects_zero_partial_or_ignored(text):
    with pytest.raises(ValueError):load_gate().cargo_passed(text,6)


def test_gpu_gate_accepts_required_rust_count():
    assert load_gate().cargo_passed('test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured;',6)==6

@pytest.mark.parametrize('tag',['skipped','failure','error'])
def test_gpu_gate_rejects_junit_nonpasses(tmp_path,tag):
    path=tmp_path/'report.xml';path.write_text(f'<testsuite><testcase><{tag}/></testcase></testsuite>')
    with pytest.raises(ValueError):load_gate().pytest_passed(path,1)


def test_gpu_gate_checks_actual_testcase_count(tmp_path):
    path=tmp_path/'report.xml';path.write_text('<testsuite tests="999"><testcase/></testsuite>')
    with pytest.raises(ValueError):load_gate().pytest_passed(path,2)
    assert load_gate().pytest_passed(path,1)==1
