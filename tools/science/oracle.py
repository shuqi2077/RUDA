#!/usr/bin/env python3
"""Independent numerical experiments and reproducible Rust fixture generation.

These experiments validate Python translations of the mathematical methods;
THEY DO NOT EXECUTE OR VALIDATE THE RUST OR CUDA IMPLEMENTATIONS.
NumPy/SciPy supply independent expected solutions, not self-generated targets.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

try:
    import numpy as np
    import scipy
    from scipy.integrate import quad, solve_ivp as scipy_ivp
except ImportError as error:
    print(json.dumps({"status": "blocked", "reason": str(error)}))
    raise SystemExit(3)

ROOT = Path(__file__).resolve().parents[2]


def norm(x):
    scale, ss = 0., 1.
    for value in np.asarray(x).flat:
        a = abs(float(value))
        if not math.isfinite(a):
            raise ValueError("nonfinite norm input")
        if a:
            if scale < a:
                ss = 1 + ss * (scale / a) ** 2
                scale = a
            else:
                ss += (a / scale) ** 2
    return scale * math.sqrt(ss)


def lu_solve(a, b):
    w, x = np.array(a, dtype=float, copy=True), np.array(b, dtype=float, copy=True)
    n = len(w)
    for k in range(n):
        p = k + int(np.argmax(abs(w[k:, k])))
        if w[p, k] == 0:
            raise ValueError("singular")
        w[[k, p]] = w[[p, k]]
        x[[k, p]] = x[[p, k]]
        for i in range(k+1, n):
            w[i, k] /= w[k, k]
            for j in range(k+1, n):
                w[i, j] -= w[i, k] * w[k, j]
    for i in range(n):
        for c in range(x.shape[1]):
            x[i, c] -= math.fsum(w[i, j] * x[j, c] for j in range(i))
    for i in reversed(range(n)):
        for c in range(x.shape[1]):
            x[i, c] = (x[i, c] - math.fsum(w[i, j] * x[j, c] for j in range(i+1, n))) / w[i, i]
    return x


def cholesky_solve(a, b):
    n = len(a)
    l, x = np.zeros_like(a), b.copy()
    for j in range(n):
        pivot = a[j, j] - math.fsum(float(l[j, k])**2 for k in range(j))
        if pivot <= 0:
            raise ValueError("not SPD")
        l[j, j] = math.sqrt(pivot)
        for i in range(j+1, n):
            l[i, j] = (a[i, j] - math.fsum(l[i, k] * l[j, k] for k in range(j))) / l[j, j]
    for i in range(n):
        for c in range(x.shape[1]):
            x[i, c] = (x[i, c] - math.fsum(l[i, k] * x[k, c] for k in range(i))) / l[i, i]
    for i in reversed(range(n)):
        for c in range(x.shape[1]):
            x[i, c] = (x[i, c] - math.fsum(l[k, i] * x[k, c] for k in range(i+1, n))) / l[i, i]
    return l, x


def qr_solve(a, b):
    w, y = a.copy(), b.copy()
    m, n = a.shape
    tau, permutation = np.zeros(n), np.arange(n)
    for k in range(n):
        norms = [norm(w[k:, j]) for j in range(k, n)]
        best = k + int(np.argmax(norms))
        length = max(norms)
        w[:, [k, best]] = w[:, [best, k]]
        permutation[[k, best]] = permutation[[best, k]]
        if length == 0:
            continue
        x0 = w[k, k]
        sign = 1. if x0 >= 0 else -1.
        divisor = x0 / length + sign
        tau[k] = 1 + abs(x0) / length
        w[k+1:, k] = (w[k+1:, k] / length) / divisor
        w[k, k] = -sign * length
        for j in range(k+1, n):
            s = tau[k] * math.fsum([w[k, j]] + [w[i, k]*w[i, j] for i in range(k+1, m)])
            w[k, j] -= s
            for i in range(k+1, m):
                w[i, j] -= w[i, k]*s
    for k in range(n):
        for c in range(y.shape[1]):
            s = tau[k] * math.fsum([y[k, c]] + [w[i, k]*y[i, c] for i in range(k+1, m)])
            y[k, c] -= s
            for i in range(k+1, m):
                y[i, c] -= w[i, k]*s
    residuals = np.array([norm(y[n:, j]) for j in range(y.shape[1])])
    for i in reversed(range(n)):
        for c in range(y.shape[1]):
            y[i, c] = (y[i, c] - math.fsum(w[i, j]*y[j, c] for j in range(i+1, n))) / w[i, i]
    x = np.zeros((n, y.shape[1]))
    x[permutation, :] = y[:n, :]
    return x, residuals


def jacobi(a):
    n = len(a)
    scale = np.max(abs(a))
    if scale == 0:
        return np.zeros(n), np.eye(n)
    w, v = a.copy() / scale, np.eye(n)
    cutoff = 1e-12 * norm(w)
    for sweep in range(65):
        off = norm(w[np.tril_indices(n, -1)]) * math.sqrt(2)
        if off <= cutoff:
            order = np.argsort(np.diag(w))
            return np.diag(w)[order] * scale, v[:, order]
        if sweep == 64:
            raise ValueError("eigen did not converge")
        for p in range(n):
            for q in range(p+1, n):
                apq = w[p, q]
                if apq == 0:
                    continue
                delta = (w[q, q]-w[p, p]) / 2
                t = 1. if delta == 0 else apq / (delta + math.copysign(math.hypot(delta, apq), delta))
                c, s = 1/math.sqrt(1+t*t), t/math.sqrt(1+t*t)
                w[p, p] -= t*apq
                w[q, q] += t*apq
                w[p, q] = w[q, p] = 0.
                for k in range(n):
                    if k != p and k != q:
                        x, y = w[k, p], w[k, q]
                        w[k, p] = w[p, k] = c*x-s*y
                        w[k, q] = w[q, k] = s*x+c*y
                for k in range(n):
                    x, y = v[k, p], v[k, q]
                    v[k, p], v[k, q] = c*x-s*y, s*x+c*y


def pcg(a, b):
    x, r = np.zeros_like(b), b.copy()
    z, p = r/np.diag(a), r/np.diag(a)
    rho = math.fsum(r*z)
    target = 1e-10*norm(b)
    if norm(r) <= target:
        return x
    for iteration in range(1, 1001):
        ap = a@p
        curvature = math.fsum(p*ap)
        if curvature <= 0:
            raise ValueError("CG breakdown")
        alpha = rho/curvature
        x += alpha*p
        r -= alpha*ap
        replace = iteration % 32 == 0 or norm(r) <= target or iteration == 1000
        if replace:
            r = b-a@x
        if norm(r) <= target:
            return x
        z = r/np.diag(a)
        new_rho = math.fsum(r*z)
        p = z.copy() if replace else z+(new_rho/rho)*p
        rho = new_rho
    raise ValueError("CG max iterations")


X = np.array([.9914553711208126,.9491079123427585,.8648644233597691,.7415311855993945,.5860872354676911,.4058451513773972,.2077849550078985,0.])
WK = np.array([.022935322010529225,.06309209262997855,.10479001032225018,.14065325971552592,.1690047266392679,.19035057806478542,.20443294007529889,.20948214108472782])
WG = np.array([.1294849661688697,.27970539148927667,.38183005050511894,.4179591836734694])


def qrule(f, a, b):
    mid, half = a+(b-a)/2, (b-a)/2
    fc = f(mid)
    l, r = np.array([f(mid-half*x) for x in X[:7]]), np.array([f(mid+half*x) for x in X[:7]])
    k = WK[7]*fc + math.fsum(WK[:7]*l) + math.fsum(WK[:7]*r)
    g = WG[3]*fc + math.fsum(WG[:3]*l[1::2]) + math.fsum(WG[:3]*r[1::2])
    absolute = half*(WK[7]*abs(fc)+math.fsum(WK[:7]*abs(l))+math.fsum(WK[:7]*abs(r)))
    asc = half*(WK[7]*abs(fc-k/2)+math.fsum(WK[:7]*abs(l-k/2))+math.fsum(WK[:7]*abs(r-k/2)))
    error = abs(half*(k-g))
    if asc > 0 and error > 0:
        error = asc*min(1, (200*error/asc)**1.5)
    error = max(error, 50*np.finfo(float).eps*absolute)
    return half*k, error


def quadrature(f, a, b):
    if a == b:
        return 0.
    if a > b:
        return -quadrature(f,b,a)
    leaves = [(a,b,*qrule(f,a,b))]
    for _ in range(4096):
        total = math.fsum(x[2] for x in leaves)
        err = math.fsum(x[3] for x in leaves)
        if err <= max(1e-11, 1e-11*abs(total)):
            return total
        i = max(range(len(leaves)),key=lambda i:leaves[i][3])
        lo, hi, _, _ = leaves.pop(i)
        mid = lo+(hi-lo)/2
        leaves.extend([(lo,mid,*qrule(f,lo,mid)),(mid,hi,*qrule(f,mid,hi))])
    raise ValueError("quadrature budget")


C = np.array([0.,1/5,3/10,4/5,8/9,1.,1.])
A = np.zeros((7,7))
A[1,:1]=[1/5]
A[2,:2]=[3/40,9/40]
A[3,:3]=[44/45,-56/15,32/9]
A[4,:4]=[19372/6561,-25360/2187,64448/6561,-212/729]
A[5,:5]=[9017/3168,-355/33,46732/5247,49/176,-5103/18656]
A[6,:6]=[35/384,0.,500/1113,125/192,-2187/6784,11/84]
B4=np.array([5179/57600,0.,7571/16695,393/640,-92097/339200,187/2100,1/40])


def rk45(fun, t0, tf, y0):
    y=np.array(y0,dtype=float)
    direction=1 if tf>t0 else -1
    t=t0;h_abs=abs(tf-t0)*.01;k=np.zeros((7,len(y)));k[0]=fun(t,y)
    rejected=False
    for _ in range(100000):
        remaining=abs(tf-t);h_abs=min(h_abs,remaining)
        tn=tf if h_abs==remaining else t+direction*h_abs
        h=tn-t
        for stage in range(1,7):
            temp=y+h*(A[stage,:stage]@k[:stage])
            k[stage]=fun(tn if stage>=5 else t+h*C[stage],temp)
        candidate=temp.copy()
        error=h*((A[6]-B4)@k)
        error_norm=np.max(abs(error)/(1e-9+1e-7*np.maximum(abs(y),abs(candidate))))
        if error_norm<=1:
            t,y=tn,candidate
            if t==tf:
                return y
            k[0]=k[6]
            factor=5 if error_norm==0 else np.clip(.9*error_norm**(-.2),.2,5)
            if rejected:
                factor=min(factor,1)
            h_abs=abs(h)*factor;rejected=False
        else:
            h_abs=abs(h)*np.clip(.9*error_norm**(-.2),.2,.9);rejected=True
    raise ValueError("ODE budget")


def cuda_model(a,b,shift=0.):
    """Model per-thread FP32 recurrence, NOT a GPU run or compiler equivalence proof."""
    a,b=np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32)
    n=len(a);l=np.zeros_like(a);x=np.zeros_like(b)
    with np.errstate(all='ignore'):
        for j in range(n):
            diagonal=np.float32(a[j,j]+np.float32(shift))
            for k in range(j):
                diagonal=np.float32(diagonal-np.float32(l[j,k]*l[j,k]))
            if not np.isfinite(diagonal) or diagonal<=0:
                raise ValueError("non-SPD or overflow")
            l[j,j]=np.sqrt(diagonal)
            for i in range(j+1,n):
                s=a[i,j]
                for k in range(j):
                    s=np.float32(s-np.float32(l[i,k]*l[j,k]))
                l[i,j]=np.float32(s/l[j,j])
        for c in range(b.shape[1]):
            for i in range(n):
                v=b[i,c]
                for k in range(i):
                    v=np.float32(v-np.float32(l[i,k]*x[k,c]))
                x[i,c]=np.float32(v/l[i,i])
            for i in reversed(range(n)):
                v=x[i,c]
                for k in range(i+1,n):
                    v=np.float32(v-np.float32(l[k,i]*x[k,c]))
                x[i,c]=np.float32(v/l[i,i])
    return x


def fixtures(rng):
    result=[]
    for n in [1,2,3,5,8]:
        for kind in ['lu','cholesky','qr','eigen']:
            m=n+3 if kind=='qr' else n
            a=rng.normal(size=(m,n))
            if kind=='cholesky':
                a=a@a.T+n*np.eye(n)
            elif kind=='lu':
                a+=2*n*np.eye(n);a=a[::-1].copy()
            elif kind=='eigen':
                a=(a+a.T)/2
            if kind=='eigen':
                expected=np.linalg.eigvalsh(a);b=None
            else:
                b=rng.normal(size=(m,2))
                expected=np.linalg.lstsq(a,b,rcond=None)[0] if kind=='qr' else np.linalg.solve(a,b)
            result.append(dict(kind=kind,rows=m,columns=n,a=a.tolist(),b=None if b is None else b.tolist(),expected=expected.tolist()))
    return result


def rust_vec(x):
    return '&[' + ','.join(repr(float(v)) for v in np.asarray(x).flat) + ']'


def rust_fixtures(cases):
    text=['// SPDX-License-Identifier: Apache-2.0',
          '// GENERATED by tools/science/oracle.py --generate-fixtures.',
          '// Expected numbers come from NumPy linalg; tests still require Rust execution.',
          'use super::*;']
    for i,c in enumerate(cases):
        text.append(f'#[test] fn numpy_fixture_{i:02}_{c["kind"]}() {{')
        text.append(f'let a=matrix({c["rows"]},{c["columns"]},{rust_vec(c["a"])});')
        if c['kind']=='eigen':
            text.append('let result=symmetric_eigen(a.view(),Default::default()).unwrap();')
            text.append(f'close(&result.values,{rust_vec(c["expected"])},2e-10);')
            text.append('let av=multiply(&a,&result.vectors); let n=a.rows(); for i in 0..n{for j in 0..n{assert!((av.data[i*n+j]-result.vectors.data[i*n+j]*result.values[j]).abs()<1e-9);}}')
        else:
            text.append(f'let b=matrix({c["rows"]},2,{rust_vec(c["b"])});')
            if c['kind']=='lu':
                text.append('let x=Lu::factor(a.view(),Default::default()).unwrap().solve(b.view()).unwrap();')
            elif c['kind']=='cholesky':
                text.append('let x=Cholesky::factor(a.view(),Default::default()).unwrap().solve(b.view()).unwrap();')
            else:
                text.append('let x=Qr::factor(a.view(),Default::default()).unwrap().least_squares(b.view()).unwrap().solution;')
            text.append(f'close(x.values(),{rust_vec(c["expected"])},2e-10);')
        text.append('}')
    return '\n'.join(text)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generate-fixtures',action='store_true')
    args=parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite existing evidence')
    rng=np.random.default_rng(20260910)
    results={}; comparisons=0
    def check(kind,got,want,rtol=2e-9,atol=2e-10):
        nonlocal comparisons
        np.testing.assert_allclose(got,want,rtol=rtol,atol=atol)
        count=results.setdefault(kind,{'comparisons':0,'max_scaled_error':0.0})
        error=float(np.max(abs(np.asarray(got)-want)/(1+abs(np.asarray(want)))))
        count['comparisons']+=1;count['max_scaled_error']=max(count['max_scaled_error'],error)
        comparisons+=1
    for n in [1,2,3,5,8,16,32]:
        for scale in [1e-100,1.,1e100]:
            for repeat in range(2):
                a=rng.normal(size=(n,n));a+=2*n*np.eye(n);a=a[::-1].copy()*scale
                b=rng.normal(size=(n,3))*scale
                check('lu',lu_solve(a,b),np.linalg.solve(a,b))
                a=rng.normal(size=(n,n));a=(a@a.T+n*np.eye(n))*scale
                l,x=cholesky_solve(a,b)
                check('cholesky_solve',x,np.linalg.solve(a,b))
                check('cholesky_factor_scaled',l/math.sqrt(scale),np.linalg.cholesky(a)/math.sqrt(scale))
                a=rng.normal(size=(n+5,n))*scale;b=rng.normal(size=(n+5,2))*scale
                x,res=qr_solve(a,b)
                want=np.linalg.lstsq(a,b,rcond=None)[0]
                check('qr_least_squares',x,want)
                check('qr_residual_scaled',res/scale,np.linalg.norm(a@want-b,axis=0)/scale)
    for n in [1,2,5,9,16]:
        for scale in [1e-200,1.,1e200]:
            a=rng.normal(size=(n,n));a=(a+a.T)/2*scale
            values,vectors=jacobi(a)
            check('eigenvalues_scaled',values/scale,np.linalg.eigvalsh(a)/scale)
            check('eigenvector_residual',(a/scale)@vectors,vectors*(values/scale))
            check('eigenvector_orthogonality',vectors.T@vectors,np.eye(n))
    for n in [2,8,20]:
        for condition in [1.,10.,100.]:
            q,_=np.linalg.qr(rng.normal(size=(n,n)));a=(q*np.geomspace(1,condition,n))@q.T;b=rng.normal(size=n)
            check('preconditioned_cg',pcg(a,b),np.linalg.solve(a,b),rtol=1e-8,atol=2e-9)
    functions=[('polynomial',lambda x:x**4,0.,1.),('exp',math.exp,0.,1.),
               ('sin',math.sin,0.,math.pi),('gaussian',lambda x:math.exp(-x*x),-2.,2.),
               ('cusp',lambda x:abs(x-.13),0.,1.),('rational',lambda x:1/(1+x*x),-5.,5.)]
    for _,f,a,b in functions:
        expected=quad(f,a,b,epsabs=1e-12,epsrel=1e-12)[0]
        check('quadrature',np.array([quadrature(f,a,b),quadrature(f,b,a)]),np.array([expected,-expected]),rtol=1e-9,atol=1e-10)
    ode_cases=[(lambda t,y:-y,0.,5.,[1.]),(lambda t,y:y,1.,0.,[math.e]),
               (lambda t,y:np.array([y[1],-y[0]]),0.,2*math.pi,[1.,0.]),
               (lambda t,y:2*y*(1-y),0.,4.,[.1]),
               (lambda t,y:np.array([-y[0],-2*y[1],-3*y[2]]),0.,3.,[1.,2.,3.])]
    for f,a,b,y in ode_cases:
        reference=scipy_ivp(f,[a,b],y,method='DOP853',rtol=2e-12,atol=1e-14)
        if not reference.success:
            raise AssertionError(reference.message)
        check('rk45_final_state',rk45(f,a,b,y),reference.y[:,-1],rtol=2e-6,atol=2e-7)
    for n in [1,2,7,16,32]:
        for nrhs in [1,3,8]:
            r=rng.normal(size=(n,n));a=(r@r.T+n*np.eye(n)).astype(np.float32);b=rng.normal(size=(n,nrhs)).astype(np.float32)
            check('fp32_device_recurrence_model',cuda_model(a,b),np.linalg.solve(a.astype(float),b.astype(float)),rtol=3e-5,atol=3e-6)
    cases=fixtures(np.random.default_rng(2077))
    if args.generate_fixtures:
        (ROOT/'tools/science/fixtures.json').write_text(json.dumps(cases,indent=2)+'\n')
        (ROOT/'ruSOLVER/src/fixtures.rs').write_text(rust_fixtures(cases))
    report={'schema':'ruda.science.python_oracle.v1','status':'passed','seed':20260910,
        'scope':'Python mathematical models vs NumPy/SciPy; NOT Rust compilation or GPU execution',
        'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__},
        'comparisons':comparisons,'results':results,'independent_rust_fixture_cases':len(cases),
        'rust_executed':False,'gpu_executed':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report,output,indent=2);output.write('\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
