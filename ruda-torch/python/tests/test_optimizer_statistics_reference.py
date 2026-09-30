"""Independent host simulation of warp partitioning and scaled sum-of-squares.

Checks the proposed algorithm, not generated Rust/PTX. Never imported at runtime.
"""
import math
import numpy as np
import pytest


def scan(x,inverse,warps):
    x=np.asarray(x,dtype=np.float32).reshape(-1)
    size=32*warps
    scales=np.zeros(size,np.float32);sums=np.zeros(size,np.float32);bad=np.zeros(size,np.float32)
    for start in range(0,x.size,size):
        raw=np.zeros(size,np.float32);raw[:min(size,x.size-start)]=x[start:start+size]
        with np.errstate(over='ignore',invalid='ignore'):values=raw*np.float32(inverse)
        valid=np.isfinite(raw)&np.isfinite(values);bad=np.maximum(bad,(~valid).astype(np.float32))
        a=np.where(valid,np.abs(values),0)
        greater=a>scales
        r=np.divide(scales,a,out=np.zeros_like(a),where=greater)
        new=1+sums*r*r
        q=np.divide(a,scales,out=np.zeros_like(a),where=(~greater)&(a>0))
        sums=np.where(greater,new,sums+q*q)
        scales=np.maximum(scales,a)
    scales=scales.reshape(warps,32);sums=sums.reshape(warps,32)
    largest=scales.max(1)
    ratios=np.divide(scales,largest[:,None],out=np.zeros_like(scales),where=largest[:,None]>0)
    return np.stack((largest,(sums*ratios*ratios).sum(1),bad.reshape(warps,32).max(1)),axis=1)


def merge(parts):
    stats=np.concatenate(parts,axis=0)
    scale=float(stats[:,0].max());bad=float(stats[:,2].max())
    ratio=stats[:,0]/np.float32(scale) if scale else np.zeros(len(stats),np.float32)
    ssq=float(np.sum(stats[:,1]*ratio*ratio,dtype=np.float32))
    return bad,scale*math.sqrt(ssq)

@pytest.mark.parametrize('size',[0,1,31,32,33,129,4097,32769])
@pytest.mark.parametrize('mag',[0.,1e-30,1.,1e20])
@pytest.mark.parametrize('inverse',[1.,.125])
def test_norm_partition_reference(size,mag,inverse):
    rng=np.random.default_rng(size+35);x=(rng.normal(size=size)*mag).astype(np.float32)
    w=max(1,min(1024,(size+31)//32));actual=merge([scan(x,inverse,w)])
    expected=np.linalg.norm((x*np.float32(inverse)).astype(np.float64))
    assert actual[0]==0
    assert actual[1]==pytest.approx(expected,rel=3e-6,abs=1e-42)

@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('index',[0,31,32,10002])
def test_invalid_values_and_tail(bad,index):
    x=np.ones(10003,np.float32);x[index]=bad
    assert merge([scan(x,1.,313)])[0]==1


def test_multiple_parameter_global_not_per_parameter_norm():
    parts=[scan([3.],1.,1),scan([4.],1.,1),scan([],1.,1)]
    assert merge(parts)==(0.,5.)


def test_raw_finite_unscale_overflow_flagged():
    assert merge([scan([3e38],2.,1)])[0]==1


def test_global_norm_above_fp32_range_stays_finite():
    bad,norm=merge([scan(np.full(4,3e38,dtype=np.float32),1.,1)])
    assert bad==0 and math.isfinite(norm) and norm>float(np.finfo(np.float32).max)
