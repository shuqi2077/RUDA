#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only numerical experiment, NOT execution of RUDA/Rust/CUDA kernels.

Compare a NumPy FP32 model of the device reduction tree to an independent FP64
sum-of-squares; compare clipping and AdamW semantics to real PyTorch CPU.
Large/small magnitude cases use the FP64 oracle (PyTorch's FP32 norm can overflow).
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('refusing to overwrite report')
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        print(f'BLOCKED: {exc}', file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    rng = np.random.default_rng(18472)
    checks = 0
    maximum_relative_error = 0.0
    max_torch_abs_error = 0.0
    report = {
        'schema': 'ruda.gradient_guard.cpu_oracle.v1', 'status': 'running',
        'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'numpy': np.__version__, 'torch': torch.__version__,
        'rust_executed': False, 'cuda_executed': False, 'performance_measured': False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.report.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(args.report)

    def merge(a, sa, b, sb):
        # FP32 operations model the source expression and do not use fast-math.
        big = b > a
        ratio = np.zeros_like(a)
        np.divide(a, b, out=ratio, where=big)
        np.divide(b, a, out=ratio, where=(~big) & (b > 0))
        ss = np.where(big, sb + sa * (ratio * ratio), sa + sb * (ratio * ratio))
        return np.maximum(a, b), ss.astype(np.float32)

    def stage(raw, blocks, partial_input, inverse):
        lanes = blocks * 256
        scale = np.zeros(lanes, np.float32)
        sums = np.zeros(lanes, np.float32)
        bad = np.zeros(lanes, bool)
        count = len(raw)
        for start in range(0, count, lanes):
            chunk = raw[start:start+lanes]
            n = len(chunk)
            if partial_input:
                b = chunk[:, 0]; sb = chunk[:, 1]; next_bad = chunk[:, 2] != 0
            else:
                with np.errstate(over='ignore', invalid='ignore', under='ignore'):
                    value = chunk * inverse
                next_bad = (~np.isfinite(chunk)) | (~np.isfinite(value))
                b = np.where(next_bad, 0.0, np.abs(value)).astype(np.float32)
                sb = (b > 0).astype(np.float32)
            scale[:n], sums[:n] = merge(scale[:n], sums[:n], b, sb)
            bad[:n] |= next_bad
        scale = scale.reshape(blocks, 256)
        sums = sums.reshape(blocks, 256)
        bad = bad.reshape(blocks, 256)
        offset = 128
        while offset:
            scale[:, :offset], sums[:, :offset] = merge(
                scale[:, :offset], sums[:, :offset],
                scale[:, offset:2*offset], sums[:, offset:2*offset])
            bad[:, :offset] |= bad[:, offset:2*offset]
            offset //= 2
        return np.stack([scale[:, 0], sums[:, 0], bad[:, 0].astype(np.float32)], axis=1)

    def model_summary(values, loss_scale):
        if not len(values): return np.array([0.0, 0.0, 0.0], np.float32)
        blocks = min((len(values)+1023)//1024, 1024)
        x = stage(values, blocks, False, np.float32(1.0)/np.float32(loss_scale))
        if blocks > 1: x = stage(x, 1, True, np.float32(1.0))
        return x[0]

    def check_summary(values, loss_scale=1.0):
        nonlocal checks, maximum_relative_error
        actual = model_summary(values, loss_scale)
        with np.errstate(over='ignore', invalid='ignore'):
            unscaled = values * (np.float32(1.0)/np.float32(loss_scale))
        invalid = (~np.isfinite(values)) | (~np.isfinite(unscaled))
        assert bool(actual[2]) == bool(invalid.any())
        finite = unscaled[~invalid].astype(np.float64)
        expected = float(np.sqrt(np.dot(finite, finite)))
        obtained = float(actual[0]) * float(np.sqrt(float(actual[1])))
        if expected:
            error = abs(obtained-expected)/expected
            assert error < 2e-5, (len(values), expected, obtained)
            maximum_relative_error = max(error, maximum_relative_error)
        else:
            assert obtained == 0.0
        checks += 1

    try:
        with np.errstate(under='ignore'):
            for n in [0, 1, 31, 255, 256, 257, 1023, 1024, 1025, 4097, 65537, 1_048_577, 2_100_123]:
                x = rng.normal(0, 2.0, n).astype(np.float32)
                for scale in [1.0, 128.0, 0.5]: check_summary(x, scale)
            for values in [[0.0, -0.0], [1e30, -1e30], [1e-30, -1e-30], [3e38, 3e38],
                           [np.nan], [np.inf], [-np.inf], [0, 3, np.nan, 4], [1e30, 1e-30, -1e20]]:
                check_summary(np.array(values, np.float32))
            check_summary(np.array([np.finfo(np.float32).max], np.float32), 0.5)

        training_configurations = 0
        tensor_comparisons = 0
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for amsgrad in [False, True]:
                for maximize in [False, True]:
                    training_configurations += 1
                    original = [rng.normal(0, 0.25, n).astype(np.float32) for n in [17, 1025, 31]]
                    params = [torch.nn.Parameter(torch.tensor(x.copy())) for x in original]
                    numpy_params = [x.copy() for x in original]
                    first = [np.zeros_like(x) for x in original]
                    second = [np.zeros_like(x) for x in original]
                    vmax = [np.zeros_like(x) for x in original]
                    o = torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-5,
                                         weight_decay=0.01, amsgrad=amsgrad, maximize=maximize,
                                         foreach=False, fused=False)
                    for step in range(1, 6):
                        source = [torch.tensor(rng.normal(0, 4.0, len(x)).astype(np.float32)).to(dtype).float().numpy() for x in original]
                        summaries = [model_summary(g, 128.0) for g in source]
                        assert all(s[2] == 0 for s in summaries)
                        total = sum(float(s[0])**2*float(s[1]) for s in summaries)**0.5
                        clip = np.float32(min(0.05/(total+float(np.float32(1e-6))), 1.0))
                        for p, g in zip(params, source): p.grad = torch.tensor(g.copy()) * (1.0/128.0)
                        torch_norm = float(torch.nn.utils.clip_grad_norm_(params, 0.05))
                        assert abs(torch_norm-total) < 2e-5*max(total, 1.0)
                        o.step()
                        for i, g in enumerate(source):
                            gradient = (g*np.float32(1.0/128.0))*clip
                            if maximize: gradient = -gradient
                            b1, b2 = np.float32(0.9), np.float32(0.999)
                            first[i] = b1*first[i]+(np.float32(1)-b1)*gradient
                            second[i] = b2*second[i]+(np.float32(1)-b2)*(gradient*gradient)
                            vmax[i] = np.maximum(vmax[i], second[i])
                            var = vmax[i] if amsgrad else second[i]
                            ib1 = np.float32(1/(1-float(b1)**step))
                            ib2 = np.float32(1/(1-float(b2)**step))
                            numpy_params[i] = numpy_params[i]*np.float32(1-np.float32(0.001)*np.float32(0.01)) - np.float32(0.001)*(first[i]*ib1)/(np.sqrt(var*ib2)+np.float32(1e-5))
                            state = o.state[params[i]]
                            pairs = [(numpy_params[i], params[i].detach().numpy()),
                                     (first[i], state['exp_avg'].numpy()), (second[i], state['exp_avg_sq'].numpy())]
                            if amsgrad: pairs.append((vmax[i], state['max_exp_avg_sq'].numpy()))
                            for actual, expected in pairs:
                                np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-7)
                                max_torch_abs_error = max(max_torch_abs_error, float(np.max(np.abs(actual-expected))))
                                tensor_comparisons += 1
        report.update(status='passed', reduction_cases=checks, training_configurations=training_configurations,
                      steps_per_configuration=5, tensor_comparisons=tensor_comparisons,
                      max_reduction_relative_error=maximum_relative_error, max_pytorch_absolute_error=max_torch_abs_error,
                      scope='NumPy source-level reduction model + independent FP64 norm + real PyTorch CPU optimizer; no RUDA code executed')
    except Exception as exc:
        report.update(status='failed', error=repr(exc), reduction_cases=checks)
        save()
        raise
    finally:
        report['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
    print(json.dumps(report, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
