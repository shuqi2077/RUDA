"""CPU algorithm oracle for the AWQ packed kernel; NOT execution of Rust/GPU.

The packed path decodes one K row at a time. The independent comparison constructs
its dense matrix directly from original integer codes (never from packed words).
"""
import pytest
import torch


def pack(codes):
    order=[0,2,4,6,1,3,5,7]
    groups=codes.reshape(codes.shape[0],-1,8).long()
    words=torch.zeros(groups.shape[:-1],dtype=torch.int64)
    for nibble,column in enumerate(order):words |= groups[...,column] << (4*nibble)
    return words.to(torch.int32)


def packed_rows(x,qweight,qzeros,scales,bias,group_size):
    m,k=x.shape;n=scales.shape[1]
    # Only M*N accumulator + N decoded values; no K*N floating weight buffer.
    accumulator=torch.zeros(m,n,dtype=torch.float32)
    channels=torch.arange(n);shift=((channels%2)*4+(channels%8)//2)*4
    for row in range(k):
        q=((qweight[row,channels//8].long() & 0xffffffff)>>shift)&15
        z=((qzeros[row//group_size,channels//8].long() & 0xffffffff)>>shift)&15
        weight=((q-z).float()*scales[row//group_size].float()).to(x.dtype)
        accumulator+=x[:,row:row+1].float()*weight.float()
    result=accumulator.to(x.dtype)
    return result if bias is None else result+bias


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('batch',[1,5])
@pytest.mark.parametrize('has_bias',[False,True])
def test_packed_weight_gemm_dtypes_groups_and_rounding(dtype,batch,has_bias):
    torch.manual_seed(238);k,n,group=12,24,3
    codes=torch.randint(0,16,(k,n));zeros=torch.randint(0,16,(k//group,n))
    scales=(torch.rand(k//group,n)*.2+.01).to(dtype)
    weights=pack(codes);z=pack(zeros)
    assert (weights<0).any() # Exercise signed I32 storage, unsigned nibble shifts.
    x=torch.randn(k,batch,dtype=dtype).t() # Noncontiguous input boundary.
    bias=torch.randn(n,dtype=dtype) if has_bias else None
    actual=packed_rows(x,weights,z,scales,bias,group)
    dense=((codes-zeros.repeat_interleave(group,0)).float()*scales.repeat_interleave(group,0).float()).to(dtype)
    expected=(x.float()@dense.float()).to(dtype)
    if bias is not None:expected=expected+bias
    torch.testing.assert_close(actual,expected,rtol=2e-5 if dtype==torch.float32 else .02,atol=1e-5 if dtype==torch.float32 else .02)
    stored=sum(t.numel()*t.element_size() for t in (weights,z,scales))
    assert stored==k*n//2+(k//group)*n//2+(k//group)*n*scales.element_size()


@pytest.mark.parametrize('dtype,expected',[(torch.float16,17664),(torch.bfloat16,17664),(torch.float32,18688)])
def test_logical_packed_bytes_formula_matches_layout_tests(dtype,expected):
    k,n,group=128,256,64
    stored=(k*n)//2+(k//group)*n//2+(k//group)*n*torch.empty((),dtype=dtype).element_size()
    assert stored==expected and stored<k*n*torch.empty((),dtype=dtype).element_size()
