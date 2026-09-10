#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent numerical experiments, NOT a RUDA Rust/GPU execution test.

Compare a NumPy FP32 transcription to separately expressed PyTorch FP64 math.
Also run unmodified torch.optim.Muon (whose NS is BF16): check its EMA buffer,
report, but do not equate, its update difference from the FP32 variant.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = {'schema': 'ruda.muon.oracle.v1', 'status': 'running',
              'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'ruda_rust_executed': False, 'gpu_executed': False,
              'benchmark_measured': False}
    started = time.monotonic()
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.report.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        tmp.replace(args.report)
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        report.update(status='blocked', reason=str(exc)); save(); return 2
    torch.set_num_threads(1)
    report.update(numpy=np.__version__, torch=torch.__version__)
    comparisons = 0
    cases = 0
    maximum_weight_error = 0.0
    maximum_buffer_error = 0.0

    def np_step(w, g, prior, *, ema, nesterov, stable, rms, io=False):
        t = w.dtype.type
        beta = t(0.95)
        # RUDA converts host 1-beta to the tensor dtype independently, rather
        # than subtracting two already-rounded FP32 values.
        one_minus = t(1.0-0.95)
        if ema:
            m = g*one_minus if prior is None else prior*beta + g*one_minus
        else:
            m = g.copy() if prior is None else prior*beta + g
        direction = m if not nesterov else m*beta + g*(one_minus if ema else t(1))
        transposed = w.shape[0] > w.shape[1]
        x = (direction.T if transposed else direction).copy()
        if stable:
            scale = t(max(np.max(np.abs(x)), np.finfo(t).tiny))
            x = x/scale
            x = x/max(np.sqrt(np.sum(x*x, dtype=t)), t(1e-7)/scale)
        else:
            x = x/max(np.sqrt(np.sum(x*x, dtype=t)), t(1e-7))
        for _ in range(5):
            gram = x@x.T
            polynomial = t(-4.775)*gram + t(2.0315)*(gram@gram)
            x = t(3.4445)*x + polynomial@x
        if transposed: x = x.T
        rows, cols = w.shape[::-1] if io else w.shape
        ratio = .2*np.sqrt(max(rows,cols)) if rms else np.sqrt(max(1.,rows/cols))
        return w*t(1-.02*.01) - x*t(.02*ratio), m

    def torch64_step(w, g, prior, *, ema, nesterov, rms, io=False):
        beta = .95
        m = ((torch.zeros_like(g) if prior is None else prior)*beta + g*(1-beta)) if ema else (g.clone() if prior is None else prior*beta+g)
        direction = m if not nesterov else g*((1-beta) if ema else 1)+beta*m
        x = direction.T if w.shape[0] > w.shape[1] else direction
        x = x/torch.linalg.vector_norm(x).clamp_min(1e-7)
        for _ in range(5):
            gram = x@x.T
            # Compute the polynomial in a different association in FP64.
            transform = gram@gram*2.0315-gram*4.775
            x = torch.add(x*3.4445, transform@x)
        if w.shape[0] > w.shape[1]: x=x.T
        rows, cols = w.shape[::-1] if io else w.shape
        ratio = .2*max(rows,cols)**.5 if rms else max(1., rows/cols)**.5
        return (1-.02*.01)*w-.02*ratio*x, m

    try:
        for rows, cols in [(1, 1), (1, 5), (5, 1), (3, 5), (5, 3), (3, 3)]:
            for ema, nesterov, stable, rms, io in itertools.product([False, True], repeat=5):
                w = ((np.arange(rows*cols, dtype=np.float32)-6)*np.float32(.03)).reshape(rows,cols)
                p = torch.tensor(w.copy(), dtype=torch.float64)
                m = pm = None
                for step_index in range(4):
                    g = (((np.arange(rows*cols)*7+step_index*3)%17-8)*.125).astype(np.float32).reshape(rows,cols)
                    w,m=np_step(w,g,m,ema=ema,nesterov=nesterov,stable=stable,rms=rms,io=io)
                    p,pm=torch64_step(p,torch.tensor(g,dtype=torch.float64),pm,ema=ema,nesterov=nesterov,rms=rms,io=io)
                    np.testing.assert_allclose(w,p.numpy(),rtol=2e-4,atol=2e-4)
                    np.testing.assert_allclose(m,pm.numpy(),rtol=2e-5,atol=2e-5)
                    maximum_weight_error=max(maximum_weight_error,float(np.max(np.abs(w-p.numpy()))))
                    maximum_buffer_error=max(maximum_buffer_error,float(np.max(np.abs(m-pm.numpy()))))
                    comparisons+=2
                cases+=1
        report['fp32_vs_fp64']={'cases':cases,'steps_per_case':4,'array_comparisons':comparisons,
                               'max_weight_absolute_error':maximum_weight_error,
                               'max_momentum_absolute_error':maximum_buffer_error}
        extreme_cases=0
        for scale in [0.,1e-30,1e30]:
            w=np.ones((2,2),dtype=np.float32)
            g=(np.array([[1.,-.5],[.25,.75]])*scale).astype(np.float32)
            nw,_=np_step(w,g,None,ema=True,nesterov=True,stable=True,rms=False)
            pw,_=torch64_step(torch.tensor(w,dtype=torch.float64),torch.tensor(g,dtype=torch.float64),None,ema=True,nesterov=True,rms=False)
            np.testing.assert_allclose(nw,pw.numpy(),atol=2e-4,rtol=2e-4)
            extreme_cases+=1
        report['extreme_cases']=extreme_cases
        if hasattr(torch.optim,'Muon'):
            native_max=0.0; native_cases=0; native_buffer_comparisons=0
            for shape in [(3,5),(5,3),(3,3)]:
                for nesterov,rms in itertools.product([False,True],repeat=2):
                    w=(np.arange(np.prod(shape),dtype=np.float32)*.03).reshape(shape)
                    parameter=torch.nn.Parameter(torch.tensor(w.copy()))
                    optimizer=torch.optim.Muon([parameter],lr=.02,weight_decay=.01,momentum=.95,nesterov=nesterov,
                        adjust_lr_fn='match_rms_adamw' if rms else 'original')
                    m=None
                    for step_index in range(4):
                        g=(((np.arange(w.size)*7+step_index*3)%17-8)*.125).astype(np.float32).reshape(shape)
                        parameter.grad=torch.tensor(g.copy())
                        optimizer.step()
                        w,m=np_step(w,g,m,ema=True,nesterov=nesterov,stable=False,rms=rms)
                        buffer=optimizer.state[parameter]['momentum_buffer'].numpy()
                        np.testing.assert_allclose(buffer,m,atol=2e-6,rtol=2e-6)
                        assert torch.isfinite(parameter).all()
                        native_max=max(native_max,float(np.max(np.abs(parameter.detach().numpy()-w))))
                        native_buffer_comparisons+=1
                    native_cases+=1
            import torch.optim._muon as native_source
            report['unmodified_torch_muon']={'cases':native_cases,'steps_per_case':4,
                'momentum_comparisons_passed':native_buffer_comparisons,
                'native_ns_dtype':'BF16', 'ruda_oracle_ns_dtype':'FP32',
                'weights_claimed_equal':False,'max_observed_weight_difference':native_max,
                'source_sha256':hashlib.sha256(inspect.getsource(native_source).encode()).hexdigest()}
        else:
            report['unmodified_torch_muon']={'status':'blocked','reason':'installed torch has no optim.Muon'}
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',exception_type=type(exc).__name__,reason=str(exc))
    report.update(elapsed_seconds=time.monotonic()-started,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save()
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='passed' else 1

if __name__ == '__main__':
    sys.exit(main())
