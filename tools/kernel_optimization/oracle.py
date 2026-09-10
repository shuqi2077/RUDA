#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FP32 algorithm/layout models, NOT execution of the Rust or GPU kernels.

Compare serial vs staged warp ownership and independent NumPy/SciPy FP64.
Access tracking checks the MODEL's shared-memory phases. It is neither formal
verification nor a substitute for device memcheck/racecheck/synccheck.
"""
from __future__ import annotations
import argparse, json, math, hashlib
from pathlib import Path
import numpy as np
import scipy.linalg
F=np.float32

class Shared:
    def __init__(self,size,tracked):
        self.data=np.full(size,np.nan,dtype=np.float32)
        self.tracked=tracked;self.reads={};self.writes={};self.phases=0
    def read(self,lane,i):
        if not 0<=i<len(self.data):raise AssertionError(('out of range read',i,len(self.data)))
        if self.tracked:self.reads.setdefault(i,set()).add(lane)
        return self.data[i]
    def write(self,lane,i,v):
        if not 0<=i<len(self.data):raise AssertionError(('out of range write',i,len(self.data)))
        if self.tracked:self.writes.setdefault(i,set()).add(lane)
        self.data[i]=v
    def barrier(self):
        if self.tracked:
            for i,w in self.writes.items():
                if len(w)>1:raise AssertionError(('multiple writers in one phase',i,w))
                if any(reader not in w for reader in self.reads.get(i,())):
                    raise AssertionError(('cross-lane read/write without barrier',i,w,self.reads[i]))
        self.reads.clear();self.writes.clear();self.phases+=1

def solve(a,b,kind,warp=True,shift=0.,atol=0.,rtol=None):
    """Identical per-output FP32 recurrence, different ownership/layout.

    Serial model uses one owner/tightly packed storage. Warp model assigns
    input/output elements by lane, rows to lanes, RHS columns to lanes and pads
    the factor stride. Phase boundaries mirror explicit sync_ruda handoffs.
    """
    a=np.array(a,dtype=F,copy=True);b=np.array(b,dtype=F,copy=True)
    n=a.shape[0];nr=b.shape[1];pitch=n+(n%2==0)if warp else n
    m=Shared(n*pitch,warp);r=Shared(n*nr,warp)
    owner=lambda i:i%32 if warp else 0
    row_owner=lambda row:row if warp else 0
    def barrier():m.barrier();r.barrier()
    for i in range(n*n):m.write(owner(i),(i//n)*pitch+i%n,a.flat[i])
    for i in range(n*nr):r.write(owner(i),i,b.flat[i])
    barrier();piv=np.arange(n,dtype=np.int32)
    code=-1 if not(np.isfinite(a).all()and np.isfinite(b).all())else 0
    scale=F(np.max(np.abs(a))) if code==0 else F(0)
    atol=F(atol);rtol=F((1e-5 if kind=='cholesky' else 1e-6)if rtol is None else rtol);shift=F(shift)
    if kind=='cholesky':
        if code==0:
            for i in range(n):
                lane=row_owner(i)
                for j in range(i):
                    u=m.read(lane,i*pitch+j);v=m.read(lane,j*pitch+i);s=F(max(abs(u),abs(v)))
                    if s>0 and abs(F(u/s-v/s))>max(F(atol/s),rtol):code=-2
        barrier()
        for j in range(n):
            if code!=0:continue
            d=F(m.read(0,j*pitch+j)+shift)
            for k in range(j):
                v=m.read(0,j*pitch+k);d=F(d-F(v*v))
            if not np.isfinite(d):code=-3
            elif d<=0:code=j+1
            else:m.write(0,j*pitch+j,F(np.sqrt(d)))
            barrier()
            if code==0:
                for i in range(j+1,n):
                    lane=row_owner(i);v=m.read(lane,i*pitch+j)
                    for k in range(j):v=F(v-F(m.read(lane,i*pitch+k)*m.read(lane,j*pitch+k)))
                    v=F(v/m.read(lane,j*pitch+j))
                    if not np.isfinite(v):code=-3
                    else:m.write(lane,i*pitch+j,v)
            barrier()
    else:
        if code==0:
            if scale==0:code=1
            else:
                for i in range(n*n):
                    lane=owner(i);pos=(i//n)*pitch+i%n;m.write(lane,pos,F(m.read(lane,pos)/scale))
                for i in range(n*nr):
                    lane=owner(i);v=F(r.read(lane,i)/scale);r.write(lane,i,v)
                    if not np.isfinite(v):code=-3
        barrier()
        for k in range(n):
            if code!=0:continue
            candidates=[(abs(m.read(row_owner(i),i*pitch+k)),i)for i in range(k,n)]
            best=max(v for v,_ in candidates);p=min(i for v,i in candidates if v==best)
            barrier()
            if best<=max(F(atol/scale),rtol):code=k+1;continue
            piv[k]=p
            if p!=k:
                for j in range(n):
                    lane=row_owner(j);v=m.read(lane,k*pitch+j);w=m.read(lane,p*pitch+j)
                    m.write(lane,k*pitch+j,w);m.write(lane,p*pitch+j,v)
                for c in range(nr):
                    lane=row_owner(c);v=r.read(lane,k*nr+c);w=r.read(lane,p*nr+c)
                    r.write(lane,k*nr+c,w);r.write(lane,p*nr+c,v)
            barrier()
            for i in range(k+1,n):
                lane=row_owner(i);ratio=F(m.read(lane,i*pitch+k)/m.read(lane,k*pitch+k));m.write(lane,i*pitch+k,ratio)
                if not np.isfinite(ratio):code=-3
                for j in range(k+1,n):
                    v=F(m.read(lane,i*pitch+j)-F(ratio*m.read(lane,k*pitch+j)));m.write(lane,i*pitch+j,v)
                    if not np.isfinite(v):code=-3
            barrier()
    if code==0:
        for c in range(nr):
            lane=row_owner(c)
            for i in range(n):
                v=r.read(lane,i*nr+c)
                for k in range(i):v=F(v-F(m.read(lane,i*pitch+k)*r.read(lane,k*nr+c)))
                if kind=='cholesky':v=F(v/m.read(lane,i*pitch+i))
                if kind=='cholesky'and not np.isfinite(v):code=-3
                else:r.write(lane,i*nr+c,v)
            for i in reversed(range(n)):
                v=r.read(lane,i*nr+c)
                for k in range(i+1,n):
                    pos=k*pitch+i if kind=='cholesky'else i*pitch+k
                    v=F(v-F(m.read(lane,pos)*r.read(lane,k*nr+c)))
                v=F(v/m.read(lane,i*pitch+i))
                if not np.isfinite(v):code=-3
                if np.isfinite(v)or kind=='lu':r.write(lane,i*nr+c,v)
    barrier()
    if kind=='lu'and code==0:
        for i in range(n*n):
            if i//n<=i%n:
                lane=owner(i);pos=(i//n)*pitch+i%n;v=F(m.read(lane,pos)*scale);m.write(lane,pos,v)
                if not np.isfinite(v):code=-3
    barrier()
    factor=np.zeros_like(a);x=np.zeros_like(b)
    if code==0:
        for i in range(n*n):
            if kind=='lu'or i//n>=i%n:factor.flat[i]=m.read(owner(i),(i//n)*pitch+i%n)
        for i in range(n*nr):x.flat[i]=r.read(owner(i),i)
    else:piv[:]=-1
    barrier()
    # Padding remains poison and is never used in any computation.
    if warp and pitch>n:assert np.isnan(m.data.reshape(n,pitch)[:,n:]).all()
    return factor,x,piv,code,m.phases

def check_case(a,b,kind,stats,**kw):
    original_a=np.array(a,dtype=F,copy=True);original_b=np.array(b,dtype=F,copy=True)
    with np.errstate(all='ignore'):
        serial=solve(a,b,kind,False,**kw);warp=solve(a,b,kind,True,**kw)
    assert serial[3]==warp[3],(kind,serial[3],warp[3])
    for i in[0,1,2]:
        np.testing.assert_array_equal(serial[i],warp[i]);stats['serial_warp_array_comparisons']+=1
    stats['cases']+=1;stats['model_shared_phases_checked']+=warp[4]
    if warp[3]!=0:
        assert np.all(warp[0]==0)and np.all(warp[1]==0)and np.all(warp[2]==-1)
        stats['error_cases']+=1;return warp
    aa=original_a.astype(np.float64)
    if kind=='cholesky':aa+=np.eye(len(aa))*kw.get('shift',0.)
    bb=original_b.astype(np.float64)
    expected=scipy.linalg.solve(aa,bb,assume_a='pos'if kind=='cholesky'else'gen')
    np.testing.assert_allclose(warp[1],expected,rtol=3e-4,atol=2e-5);stats['independent_comparisons']+=1
    f=warp[0].astype(np.float64);xx=warp[1].astype(np.float64)
    residual=np.max(np.abs(aa@xx-bb))/(np.linalg.norm(aa,np.inf)*np.max(np.abs(xx))+np.max(np.abs(bb))+np.finfo(float).tiny)
    assert residual<2e-5;stats['max_relative_residual']=max(stats['max_relative_residual'],float(residual))
    if kind=='cholesky':
        rec=f@f.T;want=aa
        expected_f=scipy.linalg.cholesky(aa,lower=True)
        np.testing.assert_allclose(f,expected_f,rtol=2e-4,atol=2e-5*np.sqrt(np.max(np.abs(aa))));stats['independent_comparisons']+=1
    else:
        want=aa.copy()
        for k,p in enumerate(warp[2]):want[[k,p]]=want[[p,k]]
        rec=(np.tril(f,-1)+np.eye(len(f)))@np.triu(f)
    scale=np.max(np.abs(aa));err=float(np.max(np.abs(rec-want))/scale)
    assert err<8e-5;stats['max_factor_scaled_error']=max(stats['max_factor_scaled_error'],err)
    stats['independent_comparisons']+=1
    return warp

def run_all():
    stats={'schema':'ruda.warp_direct.oracle.v1','status':'running','cases':0,'error_cases':0,'serial_warp_array_comparisons':0,
           'independent_comparisons':0,'model_shared_phases_checked':0,'max_relative_residual':0.,'max_factor_scaled_error':0.,
           'executed_rust':False,'executed_gpu':False,'performance_measured':False,'scope':'Python FP32 recurrence/layout/ownership model only; shared phase checks are not a proof about Rust/CUDA code'}
    rng=np.random.default_rng(2077)
    for kind in['cholesky','lu']:
        for n in range(1,33):
            raw=rng.normal(size=(n,n))*.1
            a=raw@raw.T+np.eye(n)*(n+1)if kind=='cholesky'else raw+np.eye(n)*(n+1)
            if kind=='lu':a=a[::-1].copy()
            nr=[1,3,8][n%3];b=rng.normal(size=(n,nr))
            check_case(a,b,kind,stats)
        for n in[2,7,16,32]:
            raw=rng.normal(size=(n,n))*.1
            a=raw@raw.T+np.eye(n)*(n+1)if kind=='cholesky'else raw+np.eye(n)*(n+1)
            if kind=='lu':a=a[::-1].copy()
            x=rng.normal(size=(n,8))
            for scale in[1e-30,1e-15,1e15,1e30]:
                scaled=(a*scale).astype(F);b=(scaled.astype(float)@x).astype(F)
                check_case(scaled,b,kind,stats)
    for kind in['cholesky','lu']:
        for a,b in[(np.eye(2),[[np.nan],[1.]]),([[np.inf,0],[0,1]],[[1.],[1.]]),
                    ([[np.nan,0],[0,1]],[[1.],[1.]]),(np.zeros((2,2)),[[1.],[1.]]),
                    ([[1,2],[2,1]]if kind=='cholesky'else [[1,2],[2,4]],[[1.],[1.]])]:
            check_case(a,b,kind,stats)
    check_case([[1,2],[0,1]],[[1],[1]],'cholesky',stats)
    check_case(np.zeros((2,2)),[[2],[4]],'cholesky',stats,shift=2.)
    check_case([[np.finfo(F).max]],[[1]],'cholesky',stats,shift=float(np.finfo(F).max))
    check_case([[1e-30]],[[1e30]],'lu',stats)
    check_case([[1,0],[0,1e-7]],[[1],[1]],'lu',stats)
    tied=check_case([[2,1,0],[-2,3,1],[1,0,4]],[[1],[2],[3]],'lu',stats)
    assert tied[2][0]==0
    stats['status']='passed';return stats

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.out and args.out.exists():parser.error('refusing to overwrite evidence')
    report=run_all()
    root=Path(__file__).resolve().parents[2]
    report['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()for p in [root/'ruSOLVER/src/tensor/warp_kernel.rs',Path(__file__)]}
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text)
    print(text)
if __name__=='__main__':main()
