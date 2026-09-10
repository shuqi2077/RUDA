# SPDX-License-Identifier: Apache-2.0
"""FP32 recurrence models, NOT GPU execution or a compiler correctness proof."""
import numpy as np
F=np.float32


def lu(a,b):
    a=np.array(a,np.float32);x=np.array(b,np.float32);n=len(a);scale=np.max(abs(a));w=a/scale;x/=scale
    for k in range(n):
        p=k+np.argmax(abs(w[k:,k]));w[[k,p]]=w[[p,k]];x[[k,p]]=x[[p,k]]
        if abs(w[k,k])<=1e-6:raise ValueError('pivot')
        for i in range(k+1,n):
            w[i,k]=F(w[i,k]/w[k,k])
            for j in range(k+1,n):w[i,j]=F(w[i,j]-F(w[i,k]*w[k,j]))
    for c in range(x.shape[1]):
        for i in range(n):
            for j in range(i):x[i,c]=F(x[i,c]-F(w[i,j]*x[j,c]))
        for i in reversed(range(n)):
            for j in range(i+1,n):x[i,c]=F(x[i,c]-F(w[i,j]*x[j,c]))
            x[i,c]=F(x[i,c]/w[i,i])
    return x


def qr(a):
    a=np.array(a,np.float32);m,n=a.shape;scale=np.max(abs(a));w=a/scale;ts=np.zeros(n,np.float32);q=np.eye(m,n,dtype=np.float32)
    for k in range(n):
        local=np.max(abs(w[k:,k]));s=F(0)
        for i in range(k,m):z=F(w[i,k]/local);s=F(s+F(z*z))
        norm=F(local*np.sqrt(s));first=w[k,k];sign=F(-1 if first<0 else 1);divisor=F(first/norm+sign);t=F(1+abs(first)/norm);ts[k]=t
        for i in range(k+1,m):w[i,k]=F(F(w[i,k]/norm)/divisor)
        w[k,k]=F(-sign*norm)
        for j in range(k+1,n):
            dot=w[k,j]
            for i in range(k+1,m):dot=F(dot+F(w[i,k]*w[i,j]))
            dot=F(dot*t);w[k,j]=F(w[k,j]-dot)
            for i in range(k+1,m):w[i,j]=F(w[i,j]-F(w[i,k]*dot))
    for k in reversed(range(n)):
        for j in range(n):
            dot=q[k,j]
            for i in range(k+1,m):dot=F(dot+F(w[i,k]*q[i,j]))
            dot=F(dot*ts[k]);q[k,j]=F(q[k,j]-dot)
            for i in range(k+1,m):q[i,j]=F(q[i,j]-F(w[i,k]*dot))
    return q,np.triu(w[:n])*scale


def eigen(a,rtol=1e-6,max_sweeps=64):
    a=np.array(a,np.float32);n=len(a);scale=np.max(abs(a));w=a/scale if scale else a.copy();v=np.eye(n,dtype=np.float32);target=F(rtol*np.sqrt(np.sum(w*w,dtype=np.float32)))
    for it in range(max_sweeps+1):
        off=F(0)
        for i in range(n):
            for j in range(i+1,n):off=F(off+F(2*w[i,j]*w[i,j]))
        if np.sqrt(off)<=target or scale==0:break
        if it==max_sweeps:raise ValueError('no convergence')
        for p in range(n):
            for q in range(p+1,n):
                apq=w[p,q]
                if apq==0:continue
                delta=F(.5*(w[q,q]-w[p,p]));t=F(1)
                if delta!=0:
                    mag=np.sqrt(F(delta*delta+apq*apq));t=F(apq/F(delta+np.copysign(mag,delta)))
                c=F(1/np.sqrt(F(1+t*t)));s=F(t*c);w[p,p]=F(w[p,p]-t*apq);w[q,q]=F(w[q,q]+t*apq);w[p,q]=w[q,p]=0
                for k in range(n):
                    if k!=p and k!=q:
                        x,y=w[k,p],w[k,q];xp=F(c*x-s*y);yq=F(s*x+c*y);w[k,p]=w[p,k]=xp;w[k,q]=w[q,k]=yq
                    x,y=v[k,p],v[k,q];v[k,p]=F(c*x-s*y);v[k,q]=F(s*x+c*y)
    order=np.argsort(np.diag(w));return np.diag(w)[order]*scale,v[:,order]


def cg(a,b,tol=1e-5,maxiter=512):
    a=np.array(a,np.float32);b=np.array(b,np.float32);sa=max(abs(a).ravel());sb=max(abs(b));a/=sa
    if sb==0:return np.zeros_like(b)
    b=b/sb;d=np.diag(a);x=np.zeros_like(b);r=b.copy();z=r/d;p=z.copy();rho=F(r@z);target=F(tol*np.linalg.norm(b))
    def mv(v):
        y=np.zeros(len(v),np.float32)
        for i in range(len(v)):
            for j in range(len(v)):y[i]=F(y[i]+F(a[i,j]*v[j]))
        return y
    for it in range(1,maxiter+1):
        ap=mv(p);cur=F(p@ap)
        if rho<=0 or cur<=0:raise ValueError('curvature')
        alpha=F(rho/cur);x=x+alpha*p;r=r-alpha*ap;rn=np.linalg.norm(r)
        replace=it%32==0 or rn<=target or it==maxiter
        if replace:r=b-mv(x);rn=np.linalg.norm(r)
        if rn<=target:return(x*sb)/sa
        z=r/d;new=F(r@z);p=z.copy()if replace else z+F(new/rho)*p;rho=new
    raise ValueError('CG no convergence')


def validate(check,rng):
    for n in[1,2,7,16,32]:
        a=rng.normal(size=(n,n)).astype(np.float32);a+=F(n+3)*np.eye(n,dtype=np.float32);b=rng.normal(size=(n,3)).astype(np.float32)
        check('device_fp32_lu_formula',lu(a,b),np.linalg.solve(a.astype(float),b.astype(float)),rtol=2e-4,atol=5e-6)
        for m in[n,min(64,n+5)]:
            aa=rng.normal(size=(m,n)).astype(np.float32);aa[:n]+=F(3)*np.eye(n,dtype=np.float32);q,r=qr(aa)
            check('device_fp32_qr_rebuild',q.astype(float)@r.astype(float),aa,rtol=1e-4,atol=1e-5)
            check('device_fp32_qr_orthogonality',q.T.astype(float)@q.astype(float),np.eye(n),rtol=0,atol=8e-6)
        aa=a@a.T+np.eye(n,dtype=np.float32);w,v=eigen(aa)
        check('device_fp32_eigen_formula',w,np.linalg.eigvalsh(aa.astype(float)),rtol=2e-4,atol=1e-4)
        check('device_fp32_eigen_residual',aa.astype(float)@v.astype(float),v.astype(float)*w,rtol=3e-4,atol=max(abs(aa).ravel())*5e-6)
    for n in[1,7,32,128]:
        a=np.diag(np.full(n,3,np.float32))
        if n>1:a+=np.diag(np.full(n-1,-1,np.float32),1)+np.diag(np.full(n-1,-1,np.float32),-1)
        b=a@np.ones(n,np.float32);check('device_fp32_cg_formula',cg(a,b),np.ones(n),rtol=0,atol=5e-4)
