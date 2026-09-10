#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent host formula experiments. Never a substitute for running RUDA Rust/GPU code.

NumPy/SciPy reference solvers and PyTorch CPU autograd provide the oracles.
Algorithm models below deliberately have no RUDA bindings.
"""
from __future__ import annotations
import argparse, datetime as dt, importlib.util, json, math, sys, time
from pathlib import Path
try:
    import numpy as np
    import scipy
    import scipy.linalg as sla
    import scipy.integrate as sint
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    import torch
except ImportError as error:
    print(f'blocked: {error}', file=sys.stderr); raise SystemExit(3)


def jacobi_svd(a, tol=1e-12, rank_tol=1e-14, max_sweeps=100):
    a=np.array(a,dtype=np.float64); m,n=a.shape
    if m<n:
        u,s,vt=jacobi_svd(a.T,tol,rank_tol,max_sweeps); return vt.T,s,u.T
    scale=np.max(np.abs(a)); b=a/scale if scale else a.copy(); v=np.eye(n)
    cutoff=rank_tol*np.linalg.norm(b)
    for sweep in range(max_sweeps+1):
        largest=0.
        for p in range(n):
            for q in range(p+1,n):
                x,y=b[:,p].copy(),b[:,q].copy(); nx=np.linalg.norm(x); ny=np.linalg.norm(y)
                if nx<=cutoff or ny<=cutoff: continue
                rho=(x/nx)@(y/ny); largest=max(largest,abs(rho))
                if abs(rho)<=tol or sweep==max_sweeps: continue
                d=max(nx,ny); gamma=rho*(nx/d)*(ny/d); delta=.5*((ny/d)**2-(nx/d)**2)
                t=np.sign(gamma) if delta==0 else gamma/(delta+math.copysign(math.hypot(delta,gamma),delta))
                c=1/math.sqrt(1+t*t); sn=t*c
                b[:,p]=c*x-sn*y; b[:,q]=sn*x+c*y
                x,y=v[:,p].copy(),v[:,q].copy(); v[:,p]=c*x-sn*y;v[:,q]=sn*x+c*y
        if largest<=tol: break
    else: raise RuntimeError('Jacobi SVD nonconvergence')
    norms=np.linalg.norm(b,axis=0); order=np.argsort(-norms,kind='stable'); u=np.zeros((m,n)); s=np.zeros(n)
    for j,old in enumerate(order):
        if norms[old]>cutoff and scale>0:
            s[j]=norms[old]*scale;u[:,j]=b[:,old]/norms[old]
        else:
            best=None;bn=0.
            for axis in range(m):
                z=np.eye(m)[:,axis]
                for _ in range(2):
                    for k in range(j):z-=u[:,k]*(u[:,k]@z)
                zn=np.linalg.norm(z)
                if zn>bn:best=z;bn=zn
            if bn<=64*np.finfo(float).eps:raise RuntimeError('null completion')
            u[:,j]=best/bn
    return u,s,v[:,order].T


def complex_lu(a,b,adjoint=False):
    a=np.array(a,complex);n=a.shape[0];x=np.array(b,complex);piv=[]
    for k in range(n):
        p=k+np.argmax(np.abs(a[k:,k]));piv.append(p);a[[k,p]]=a[[p,k]]
        if abs(a[k,k])==0:raise ValueError('singular')
        for i in range(k+1,n):
            a[i,k]/=a[k,k]
            for j in range(k+1,n):a[i,j]-=a[i,k]*a[k,j]
    if not adjoint:
        for k,p in enumerate(piv):x[[k,p]]=x[[p,k]]
        for i in range(n):x[i]-=a[i,:i]@x[:i]
        for i in reversed(range(n)):x[i]=(x[i]-a[i,i+1:]@x[i+1:])/a[i,i]
    else:
        for i in range(n):x[i]=(x[i]-a[:i,i].conj()@x[:i])/a[i,i].conj()
        for i in reversed(range(n)):x[i]-=a[i+1:,i].conj()@x[i+1:]
        for k in reversed(range(n)):p=piv[k];x[[k,p]]=x[[p,k]]
    return x


def complex_chol(a):
    a=np.asarray(a,complex);n=a.shape[0];l=np.zeros_like(a)
    for j in range(n):
        d=a[j,j].real-sum(abs(l[j,k])**2 for k in range(j))
        if d<=0:raise ValueError('not positive')
        l[j,j]=math.sqrt(d)
        for i in range(j+1,n):l[i,j]=(a[i,j]-l[i,:j]@l[j,:j].conj())/l[j,j]
    return l


def complex_qr(a):
    a=np.asarray(a,complex);m,n=a.shape;r=a.copy();vectors=[]
    for k in range(n):
        norm=np.linalg.norm(r[k:,k]);v=np.zeros(m-k,complex)
        if norm:
            phase=r[k,k]/abs(r[k,k]) if abs(r[k,k]) else 1.
            v=r[k:,k]/norm;v[0]+=phase;v/=np.linalg.norm(v)
            r[k:,k:]-=2*np.outer(v,v.conj()@r[k:,k:]);r[k+1:,k]=0
        vectors.append(v)
    q=np.eye(m,n,dtype=complex)
    for k in reversed(range(n)):
        v=vectors[k];q[k:]-=2*np.outer(v,v.conj()@q[k:])
    return q,r[:n]


def sparse_lu(a,b,transpose=False):
    a=sp.csr_matrix(a);n=a.shape[0];rows=[dict(zip(a.indices[a.indptr[i]:a.indptr[i+1]],a.data[a.indptr[i]:a.indptr[i+1]]))for i in range(n)]
    piv=[]
    for k in range(n):
        p=max(range(k,n),key=lambda i:abs(rows[i].get(k,0.)));piv.append(p);rows[k],rows[p]=rows[p],rows[k]
        pivot=rows[k].get(k,0.)
        if not pivot:raise ValueError('singular')
        tail=[(j,v)for j,v in rows[k].items()if j>k]
        for i in range(k+1,n):
            if k not in rows[i]:continue
            f=rows[i][k]/pivot;rows[i][k]=f
            for j,v in tail:
                new=rows[i].get(j,0.)-f*v
                if new:rows[i][j]=new
                else:rows[i].pop(j,None)
    x=np.asarray(b,float).copy()
    if not transpose:
        for k,p in enumerate(piv):x[[k,p]]=x[[p,k]]
        for i in range(n):
            for j,v in sorted(rows[i].items()):
                if j<i:x[i]-=v*x[j]
        for i in reversed(range(n)):
            for j,v in sorted(rows[i].items()):
                if j>i:x[i]-=v*x[j]
            x[i]/=rows[i][i]
    else:
        for i in range(n):
            x[i]/=rows[i][i]
            for j,v in sorted(rows[i].items()):
                if j>i:x[j]-=v*x[i]
        for i in reversed(range(n)):
            for j,v in sorted(rows[i].items()):
                if j<i:x[j]-=v*x[i]
        for k in reversed(range(n)):p=piv[k];x[[k,p]]=x[[p,k]]
    return x,sum(map(len,rows))


def svd_vjp(u,s,vt,gu,gs,gvt):
    au=u.T@gu;av=vt@gvt.T;k=len(s);mid=np.diag(gs.copy())
    for i in range(k):
        for j in range(k):
            if i==j:continue
            scale=max(s[i],s[j]);pi=s[i]/scale;pj=s[j]/scale
            mid[i,j]=((au[i,j]-au[j,i])*pj+pi*(av[i,j]-av[j,i]))/((pj-pi)*(pj+pi))/scale
    return u@mid@vt+((gu-u@au)/s)@vt+u@((gvt.T-vt.T@av)/s).T


def qr_vjp(q,r,gq,gr):
    gr=np.triu(gr);z=r@gr.T-gq.T@q;c=np.tril(z)+np.tril(z,-1).T
    return sla.solve_triangular(r,(gq+q@c).T,lower=False).T


def chol_vjp(l,g):
    p=np.tril(l.T@np.tril(g));p[np.diag_indices_from(p)]*=.5;linv=sla.solve_triangular(l,np.eye(len(l)),lower=True)
    a=linv.T@p@linv;return(a+a.T)*.5


def eigen_vjp(w,v,gw,gv):
    b=v.T@gv;m=np.diag(gw.copy());scale=max(abs(w))
    for i in range(len(w)):
        for j in range(len(w)):
            if i!=j:m[i,j]=.5*(b[i,j]-b[j,i])/(w[j]/scale-w[i]/scale)/scale
    a=v@m@v.T;return .5*(a+a.T)


def distributed_cg_model(a,b,world,jacobi=True,tol=1e-10,maxiter=1000):
    # Local CSR products and reductions model the row-partitioned algorithm.
    # No TCP, threads, or Rust execution takes place here.
    n=len(b);splits=np.array_split(np.arange(n),world);local=[sp.csr_matrix(a)[idx]for idx in splits]
    def apply(x):return np.concatenate([m@x for m in local])
    def dot(x,y):return math.fsum(float(x[idx]@y[idx])for idx in splits)
    x=np.zeros(n);r=b.copy();d=np.diag(a)if jacobi else np.ones(n);z=r/d;p=z.copy();rho=dot(r,z);target=tol*np.linalg.norm(b)
    if np.linalg.norm(r)<=target:return x
    for it in range(1,maxiter+1):
        ap=apply(p);alpha=rho/dot(p,ap);x+=alpha*p;r-=alpha*ap
        replace=it%32==0 or np.linalg.norm(r)<=target or it==maxiter
        if replace:r=b-apply(x)
        if np.linalg.norm(r)<=target:return x
        z=r/d;new=dot(r,z);p=z.copy()if replace else z+(new/rho)*p;rho=new
    raise RuntimeError('CG did not converge')


def bdf1(f,jac,t0,t1,y0,atol=1e-7,rtol=1e-5,initial=None,maxstep=np.inf,maxeval=1000000):
    y=np.array(y0,float);t=t0;habs=min(initial or abs(t1-t0)*.01,maxstep);direc=np.sign(t1-t0);count=[0];accepted=0;rejected=0
    def ev(t,y):
        count[0]+=1
        if count[0]>maxeval:raise RuntimeError('eval budget')
        val=np.asarray(f(t,y),float)
        if not np.isfinite(val).all():raise RuntimeError('nonfinite f')
        return val
    def norm(r,a,b):return np.max(abs(r)/(atol+rtol*np.maximum(abs(a),abs(b))))
    def step(t,old,h):
        z=old.copy()
        for _ in range(12):
            fy=ev(t,z);r=z-old-h*fy;rn=norm(r,old,z)
            if rn<=.03:return z,fy
            if jac is not None:j=np.array(jac(t,z),float)
            else:
                j=np.zeros((len(z),len(z)))
                for col in range(len(z)):
                    yy=z.copy();yy[col]+=np.sqrt(np.finfo(float).eps)*max(1,abs(z[col]));j[:,col]=(ev(t,yy)-fy)/(yy[col]-z[col])
            delta=np.linalg.solve(np.eye(len(z))-h*j,-r);lam=1.;found=False
            for _ in range(9):
                trial=z+lam*delta;ft=ev(t,trial);rn2=norm(trial-old-h*ft,old,trial)
                if rn2<=.03:return trial,ft
                if rn2<rn:z=trial;found=True;break
                lam*=.5
            if not found:return None
        return None
    f0=ev(t,y)
    for _ in range(100000):
        if t==t1:return y,accepted,rejected,count[0]
        habs=min(habs,abs(t1-t),maxstep);tn=t1 if habs==abs(t1-t)else t+direc*habs;h=tn-t;mid=t+.5*h
        full=step(tn,y,h);half=step(mid,y,mid-t)if full else None;fine=step(tn,half[0],tn-mid)if half else None
        if fine is None:rejected+=1;habs*=.25;continue
        error=norm(fine[0]-full[0],y,fine[0])
        if error<=1:
            y,f0=fine;t=tn;accepted+=1;habs*=2 if error==0 else np.clip(.9/error**.5,.2,2)
        else:rejected+=1;habs*=np.clip(.9/error**.5,.1,.8)
    raise RuntimeError('step budget')


def hermite(y0,f0,y1,f1,h,s):return (2*s**3-3*s*s+1)*y0+(s**3-2*s*s+s)*h*f0+(-2*s**3+3*s*s)*y1+(s**3-s*s)*h*f1


def event_bisect(g,t0,y0,f0,t1,y1,f1,tol=1e-10):
    lo=0.;hi=1.;gl=g(t0,y0)
    for _ in range(100):
        mid=(lo+hi)/2;t=t0+mid*(t1-t0);y=hermite(y0,f0,y1,f1,t1-t0,mid);gm=g(t,y)
        if gm==0 or abs(t1-t0)*(hi-lo)<=tol:return t,y
        if np.signbit(gm)==np.signbit(gl):lo=mid;gl=gm
        else:hi=mid
    raise RuntimeError('root budget')


def experiment():
    rng=np.random.default_rng(20770910);checks=[];fixtures=[];start=time.monotonic()
    def check(name,actual,expected,rtol=1e-9,atol=1e-10):
        x=np.asarray(actual);y=np.asarray(expected);np.testing.assert_allclose(x,y,rtol=rtol,atol=atol,err_msg=name)
        checks.append({'name':name,'max_absolute_error':float(np.max(abs(x-y)))if x.size else 0.,'rtol':float(rtol),'atol':float(atol)})
    for m,n in[(1,1),(5,1),(1,5),(3,2),(2,3),(7,4),(4,7),(9,9),(16,8)]:
        for scale in[1e-200,1.,1e200]:
            a=rng.normal(size=(m,n));u,s,vt=jacobi_svd(a*scale);_,want,_=np.linalg.svd(a,full_matrices=False)
            check(f'svd_values_{m}_{n}_{scale}',s/scale,want);check('svd_reconstruction',(u*(s/scale))@vt,a,atol=1e-9)
            check('svd_u_orthogonal',u.T@u,np.eye(min(m,n)),atol=2e-10);check('svd_v_orthogonal',vt@vt.T,np.eye(min(m,n)),atol=2e-10)
    for m,n in[(5,3),(3,5),(4,4)]:
        for rank in[0,1,2]:
            a=rng.normal(size=(m,rank))@rng.normal(size=(rank,n));u,s,vt=jacobi_svd(a)
            check('svd_rankdef_reconstruction',(u*s)@vt,a);check('svd_rankdef_u',u.T@u,np.eye(min(m,n)));check('svd_rankdef_v',vt@vt.T,np.eye(min(m,n)))
            inv=np.array([1/z if z else 0. for z in s]);check('svd_pseudoinverse',(vt.T*inv)@u.T,np.linalg.pinv(a, rcond=1e-13))
    for n in[1,2,5,9]:
        a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));a+=n*np.eye(n);b=rng.normal(size=(n,3))+1j*rng.normal(size=(n,3))
        check('complex_lu',complex_lu(a,b),np.linalg.solve(a,b));check('complex_lu_adjoint',complex_lu(a,b,True),np.linalg.solve(a.conj().T,b))
        spd=a@a.conj().T+np.eye(n);l=complex_chol(spd);check('complex_chol',l,np.linalg.cholesky(spd))
        for m in[n,n+3]:
            aa=rng.normal(size=(m,n))+1j*rng.normal(size=(m,n));q,r=complex_qr(aa);check('complex_qr_rebuild',q@r,aa);check('complex_qr_unitary',q.conj().T@q,np.eye(n))
    for n in[1,2,5,13,32]:
        a=sp.diags([-np.ones(max(0,n-1)),np.full(n,4.),-np.ones(max(0,n-1))],[-1,0,1],shape=(n,n)).toarray()
        if n>1:a[[0,-1]]=a[[-1,0]]
        b=rng.normal(size=(n,3));x,nnz=sparse_lu(a,b);check('sparse_lu',x,spla.spsolve(sp.csc_matrix(a),b).reshape(n,3));check('sparse_lu_transpose',sparse_lu(a,b,True)[0],np.linalg.solve(a.T,b))
    for n in[2,11,33]:
        a=sp.diags([-np.ones(n-1),np.full(n,4.),-np.ones(n-1)],[-1,0,1]).toarray();b=rng.normal(size=n)
        for world in[1,2,3,5]:check('distributed_cg_formula',distributed_cg_model(a,b,world),np.linalg.solve(a,b))
    torch.set_num_threads(1)
    for m,n in[(2,2),(5,3),(3,5),(8,4)]:
        for _ in range(3):
            a=torch.tensor(rng.normal(size=(m,n)),dtype=torch.float64,requires_grad=True);u,s,vt=torch.linalg.svd(a,full_matrices=False)
            gu=torch.tensor(rng.normal(size=u.shape));gs=torch.tensor(rng.normal(size=s.shape));gvt=torch.tensor(rng.normal(size=vt.shape));((u*gu).sum()+(s*gs).sum()+(vt*gvt).sum()).backward()
            actual=svd_vjp(u.detach().numpy(),s.detach().numpy(),vt.detach().numpy(),gu.numpy(),gs.numpy(),gvt.numpy())
            check('svd_full_vjp_torch',actual,a.grad.numpy(),rtol=2e-8)
            if m>=n:
                a=torch.tensor(rng.normal(size=(m,n)),dtype=torch.float64,requires_grad=True);q,r=torch.linalg.qr(a);gq=torch.tensor(rng.normal(size=q.shape));gr=torch.tensor(rng.normal(size=r.shape));((q*gq).sum()+(r*gr).sum()).backward()
                check('qr_vjp_torch',qr_vjp(q.detach().numpy(),r.detach().numpy(),gq.numpy(),gr.numpy()),a.grad.numpy(),rtol=2e-8)
    for n in[1,2,5,8]:
        z=rng.normal(size=(n,n));a0=z@z.T+np.eye(n);a=torch.tensor(a0,requires_grad=True);g=torch.tensor(rng.normal(size=(n,n)));l=torch.linalg.cholesky(a);(l*g).sum().backward();check('chol_vjp_torch',chol_vjp(l.detach().numpy(),g.numpy()),a.grad.numpy())
        a=torch.tensor(a0,requires_grad=True);w,v=torch.linalg.eigh(a);gw=torch.tensor(rng.normal(size=w.shape));gv=torch.tensor(rng.normal(size=v.shape));((w*gw).sum()+(v*gv).sum()).backward();check('eigen_full_vjp_torch',eigen_vjp(w.detach().numpy(),v.detach().numpy(),gw.numpy(),gv.numpy()),a.grad.numpy())
    stiff_stats=[]
    cases=[('decay',lambda t,y:-1000*y,lambda t,y:np.array([[-1000.]]),[1.],0.,1.),
        ('forced',lambda t,y:-1000*(y-np.cos(t))-np.sin(t),lambda t,y:np.array([[-1000.]]),[1.],0.,1.),
        ('reverse',lambda t,y:y,lambda t,y:np.eye(len(y)),[np.e],1.,0.),
        ('robertson',lambda t,y:np.array([-.04*y[0]+1e4*y[1]*y[2],.04*y[0]-1e4*y[1]*y[2]-3e7*y[1]**2,3e7*y[1]**2]),
         lambda t,y:np.array([[-.04,1e4*y[2],1e4*y[1]],[.04,-1e4*y[2]-6e7*y[1],-1e4*y[1]],[0,6e7*y[1],0]]),[1.,0.,0.],0.,1.)]
    for name,f,j,y,t0,t1 in cases:
        ref=sint.solve_ivp(f,(t0,t1),y,method='Radau',rtol=1e-11,atol=1e-13).y[:,-1]
        for analytic in[False,True]:
            got,acc,rej,ev=bdf1(f,j if analytic else None,t0,t1,y,atol=1e-9 if name=='robertson'else 1e-7)
            check('bdf1_'+name+'_'+str(analytic),got,ref,rtol=0,atol=3e-3 if name=='reverse'else 3e-4)
            stiff_stats.append({'name':name,'analytic_jacobian':analytic,'accepted':acc,'rejected':rej,'f_evaluations':ev})
    # Rational mappings use existing independently implemented GK formula model.
    spec=importlib.util.spec_from_file_location('base',Path(__file__).parents[1]/'science/oracle.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    # scipy.quad is an external oracle. Test transformed integrals with SciPy too:
    # this validates mapping, not RUDA adaptive quadrature execution.
    for f,lower,upper,expect in[(lambda x:np.exp(-x),0,np.inf,1.),(lambda x:np.exp(x),-np.inf,0,1.),(lambda x:np.exp(-x*x),-np.inf,np.inf,np.sqrt(np.pi)),(lambda x:1/(1+x*x),-np.inf,np.inf,np.pi)]:
        if lower==-np.inf and upper==np.inf:
            val=sint.quad(lambda t:(f(t/(1-t))+f(-t/(1-t)))/(1-t)**2,0,1,epsabs=1e-10)[0]
        else:
            endpoint=lower if upper==np.inf else upper;sign=1 if upper==np.inf else -1
            val=sint.quad(lambda t:f(endpoint+sign*t/(1-t))/(1-t)**2,0,1,epsabs=1e-10)[0]
        check('infinite_mapping',val,expect,atol=1e-9)
    for h in[.01,.025,.05]:
        center=np.pi/2;t0=center-.4*h;t1=t0+h
        def y(t):return np.array([np.cos(t),-np.sin(t)])
        def f(t):return np.array([-np.sin(t),-np.cos(t)])
        root,state=event_bisect(lambda t,y:y[0],t0,y(t0),f(t0),t1,y(t1),f(t1))
        check('event_hermite_time',root,center,atol=1e-8);check('event_hermite_state',state,y(root),atol=3e-8)
    # Independently generated tiny references to be consumed by REAL Rust tests.
    for a in[np.array([[3.,1.],[0.,2.],[1.,-1.]]),np.array([[1.,2.,3.],[4.,5.,6.]]),np.array([[1.,2.],[2.,4.],[3.,6.]])]:
        s=sla.svdvals(a);pinv=sla.pinv(a)
        fixtures.append({'kind':'svd','m':len(a),'n':a.shape[1],'a':a.ravel().tolist(),'singular_values':s.tolist(),'pinv':pinv.ravel().tolist()})
    import device_models
    device_models.validate(check,rng)
    return {'schema':'ruda.science.extended.oracle.v1','status':'passed','actual_rust_execution':False,'actual_gpu_execution':False,'performance_measured':False,
        'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'comparison_count':len(checks),'checks':checks,'bdf1_formula_counters':stiff_stats,'seconds':time.monotonic()-start},fixtures


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--write-fixtures',action='store_true');args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:r,fixtures=experiment()
    except Exception as error:
        args.output.write_text(json.dumps({'status':'failed','error':repr(error),'actual_rust_execution':False},indent=2)+'\n');raise
    r['timestamp_utc']=dt.datetime.now(dt.timezone.utc).isoformat();args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    if args.write_fixtures:
        here=Path(__file__).parent;(here/'fixtures.json').write_text(json.dumps({'source':'SciPy SVD/pinv; generated independently from the Rust implementation','fixtures':fixtures},indent=2)+'\n')
        parts=['// SPDX-License-Identifier: Apache-2.0','// Independently generated SciPy references; Rust tests still need execution.','use super::*;']
        for i,f in enumerate(fixtures):
            def array(values):return ','.join(repr(float(x)) for x in values)
            parts.append(f'''#[test]fn external_svd_fixture_{i}(){{
    let a=Matrix::new({f['m']},{f['n']},vec![{array(f['a'])}]).unwrap();
    let s=Svd::factor(a.view(),Default::default()).unwrap();
    let expected=[{array(f['singular_values'])}];
    for(g,e)in s.singular_values.iter().zip(expected){{assert!((g-e).abs()<1e-10*(1.+e.abs()));}}
    let expected=[{array(f['pinv'])}];let p=s.pseudo_inverse().unwrap();
    for(g,e)in p.values().iter().zip(expected){{assert!((g-e).abs()<1e-9*(1.+e.abs()));}}
}}''')
        (here.parents[1]/'ruSOLVER/src/advanced_fixtures.rs').write_text('\n'.join(parts)+'\n')
    print(json.dumps({k:v for k,v in r.items()if k not in ['checks','bdf1_formula_counters']},indent=2))
if __name__=='__main__':main()
