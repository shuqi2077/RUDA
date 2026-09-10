#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU PyTorch fixtures and FP32 formula cross-check.

This executes Python/NumPy/PyTorch, NOT RUDA Rust or GPU kernels. A passing result
must never be reported as a RUDA device test. Regeneration is opt-in.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def run_oracle() -> tuple[dict, dict]:
    import numpy as np
    import torch
    torch.set_num_threads(1)
    f = np.float32

    def fp32_step(p, stored, m, v, vmax, step, o, scale):
        # Deliberately models the scalar FP32 formula, not any GPU timing/schedule.
        g = (stored * f(1.0 / scale)).astype(np.float32)
        if o["maximize"]:
            g = -g
        m = f(o["beta1"]) * m + f(1 - f(o["beta1"])) * g
        v = f(o["beta2"]) * v + f(1 - f(o["beta2"])) * (g * g)
        if o["amsgrad"]:
            vmax = np.maximum(vmax, v)
            used = vmax
        else:
            used = v
        inv1 = f(1 / (1 - float(o["beta1"]) ** step))
        inv2 = f(1 / (1 - float(o["beta2"]) ** step))
        delta = (m * inv1) / (np.sqrt(used * inv2) + f(o["epsilon"]))
        p = p * f(1 - f(o["learning_rate"]) * f(o["weight_decay"])) - f(o["learning_rate"]) * delta
        return p, m, v, vmax

    cases = []
    max_error = 0.0
    comparisons = 0
    # Values written to JSON are exactly the FP32 hyperparameters accepted by RUDA.
    for name, dtype in [("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)]:
        for amsgrad in [False, True]:
            for maximize in [False, True]:
                n = 17  # odd-sized tails are intentional
                initial = np.linspace(-0.8, 0.7, n, dtype=np.float32)
                o = {k: float(f(v)) for k, v in dict(learning_rate=0.003, beta1=0.9, beta2=0.98, epsilon=1e-5, weight_decay=0.07).items()}
                o.update(amsgrad=amsgrad, maximize=maximize)
                parameter = torch.nn.Parameter(torch.from_numpy(initial.copy()))
                optimizer = torch.optim.AdamW([parameter], lr=o["learning_rate"], betas=(o["beta1"], o["beta2"]),
                    eps=o["epsilon"], weight_decay=o["weight_decay"], amsgrad=amsgrad, maximize=maximize,
                    foreach=False, fused=False)
                scale = 128.0
                simulated = (initial.copy(), np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32))
                case = {"dtype": name, "options": o, "gradient_scale": scale, "initial": initial.tolist(), "steps": []}
                for step in range(1, 6):
                    unrounded = np.array([((i * 7 + step * 3) % 23 - 11) * 0.017 * scale for i in range(n)], np.float32)
                    stored = torch.from_numpy(unrounded).to(dtype).to(torch.float32)
                    parameter.grad = stored / scale
                    optimizer.step()
                    state = optimizer.state[parameter]
                    simulated = fp32_step(simulated[0], stored.numpy(), *simulated[1:], step, o, scale)
                    expected_arrays = [parameter.detach().numpy(), state["exp_avg"].numpy(), state["exp_avg_sq"].numpy()]
                    if amsgrad:
                        expected_arrays.append(state["max_exp_avg_sq"].numpy())
                    for actual, expected in zip(simulated, expected_arrays):
                        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                        max_error = max(max_error, float(np.max(np.abs(actual - expected))))
                        comparisons += 1
                    case["steps"].append({"step": step, "stored_gradient": stored.tolist(),
                        "parameters": parameter.detach().tolist(), "first": state["exp_avg"].tolist(),
                        "second": state["exp_avg_sq"].tolist(),
                        "maximum": state["max_exp_avg_sq"].tolist() if amsgrad else None})
                cases.append(case)
    metadata = {
        "schema": "ruda.fused_adamw.pytorch_reference.v1", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "torch_version": torch.__version__, "numpy_version": np.__version__, "device": "cpu",
        "optimizer": "torch.optim.AdamW(foreach=False, fused=False)",
        "note": "FP32 master/state; gradients rounded to requested storage type before loss-scale division. Not a RUDA execution result.",
    }
    report = {**metadata, "status": "passed", "cases": len(cases), "steps_per_case": 5,
        "array_comparisons": comparisons, "max_absolute_difference_numpy_vs_pytorch": max_error,
        "ruda_rust_compiled": False, "ruda_gpu_executed": False, "performance_measured": False}
    return {**metadata, "cases": cases}, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-fixtures", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        document, report = run_oracle()
    except ImportError as exc:
        report = {"status": "blocked", "reason": str(exc), "ruda_gpu_executed": False}
    except Exception as exc:
        report = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "ruda_gpu_executed": False}
    else:
        if args.write_fixtures:
            path = ROOT / "ruda-optim/tests/fused_adamw/pytorch_fixtures.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            report["fixture_path"] = str(path.relative_to(ROOT))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        if args.report.exists():
            parser.error("report already exists; choose a new path")
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return {"passed": 0, "blocked": 2, "failed": 1}[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
