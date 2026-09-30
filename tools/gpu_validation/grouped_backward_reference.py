"""Independent host tile/layout oracle for v31. NOT a Rust/PTX interpreter.

Checks the shared-tile transpose and output ownership with exactly representable
stored inputs. Matmul here runs on the CPU; no results are GPU validation.
"""
from __future__ import annotations
import numpy as np

def tiled_backward(x, w, dy, lengths):
    x=np.asarray(x,dtype=np.float32);w=np.asarray(w,dtype=np.float32);dy=np.asarray(dy,dtype=np.float32)
    e,n,k=w.shape;m=x.shape[0]
    if x.shape!=(m,k) or dy.shape!=(m,n) or len(lengths)!=e or sum(lengths)!=m or any(v<0 for v in lengths):
        raise ValueError('inconsistent grouped shapes')
    dx=np.full((m,k),np.nan,np.float32);dw=np.full((e,n,k),np.nan,np.float32)
    writes_x=np.zeros_like(dx,dtype=np.int32);writes_w=np.zeros_like(dw,dtype=np.int32)
    prefix=np.r_[0,np.cumsum(lengths)]
    # dX A is row-major dY, B is column-major W, staged with coalesced loads.
    for expert in range(e):
        begin,end=map(int,prefix[expert:expert+2])
        for kb in range(0,k,16):
            for rb in range(begin,end,16):
                acc=np.zeros((16,16),np.float32)
                for nb in range(0,n,16):
                    left=np.full(256,np.nan,np.float32);right=left.copy()
                    ownership_l=np.zeros(256,np.int32);ownership_r=ownership_l.copy()
                    for lane in range(32):
                        for i in range(8):
                            t=lane+i*32;tr,tc=divmod(t,16)
                            a=x.dtype.type(0);b=a
                            if rb+tr<end and nb+tc<n:a=dy[rb+tr,nb+tc]
                            if nb+tr<n and kb+tc<k:b=w[expert,nb+tr,kb+tc]
                            left[t]=a;right[tc*16+tr]=b
                            ownership_l[t]+=1;ownership_r[tc*16+tr]+=1
                    assert np.all(ownership_l==1) and np.all(ownership_r==1)
                    acc+=left.reshape(16,16)@right.reshape(16,16).T
                for i in range(16):
                    for j in range(16):
                        if rb+i<end and kb+j<k:
                            dx[rb+i,kb+j]=acc[i,j];writes_x[rb+i,kb+j]+=1
        # dW: both input tiles load coalesced and transpose inside shared memory.
        for kb in range(0,k,16):
            for nb in range(0,n,16):
                acc=np.zeros((16,16),np.float32)
                for rb in range(begin,end,16):
                    left=np.full(256,np.nan,np.float32);right=left.copy()
                    for lane in range(32):
                        for i in range(8):
                            t=lane+i*32;tr,tc=divmod(t,16)
                            a=np.float32(0);b=a
                            if rb+tr<end and nb+tc<n:a=dy[rb+tr,nb+tc]
                            if rb+tr<end and kb+tc<k:b=x[rb+tr,kb+tc]
                            left[tc*16+tr]=a;right[tc*16+tr]=b
                    assert not np.isnan(left).any() and not np.isnan(right).any()
                    acc+=left.reshape(16,16)@right.reshape(16,16).T
                for i in range(16):
                    for j in range(16):
                        if nb+i<n and kb+j<k:
                            dw[expert,nb+i,kb+j]=acc[i,j];writes_w[expert,nb+i,kb+j]+=1
    if not np.all(writes_x==1) or not np.all(writes_w==1):raise AssertionError('unwritten or multiply-written gradient')
    return dx,dw

def dense_backward(x,w,dy,lengths):
    dx=np.zeros_like(x,dtype=np.float64);dw=np.zeros_like(w,dtype=np.float64)
    start=0
    for expert,count in enumerate(lengths):
        sl=slice(start,start+count)
        dx[sl]=dy[sl].astype(np.float64)@w[expert].astype(np.float64)
        dw[expert]=dy[sl].astype(np.float64).T@x[sl].astype(np.float64)
        start+=count
    return dx,dw
