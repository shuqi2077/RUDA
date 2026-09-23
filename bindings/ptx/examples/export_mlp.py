"""Export a small PyTorch inference graph directly to PTX; GPU execution is opt-in."""
from __future__ import annotations
import argparse
import json
import torch
import torch.nn.functional as F
from ruda_ptx import Executor, compile_exported


class TinyMLP(torch.nn.Module):
    def __init__(self, width, hidden, dtype):
        super().__init__()
        self.norm = torch.nn.RMSNorm(width, eps=1e-6, dtype=dtype)
        self.gate = torch.nn.Linear(width, hidden, dtype=dtype)
        self.up = torch.nn.Linear(width, hidden, dtype=dtype)
        self.down = torch.nn.Linear(hidden, width, dtype=dtype)
    def forward(self, x):
        y = self.norm(x)
        return self.down(F.silu(self.gate(y)) * self.up(y)) + x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-dir", default=None, help="New directory for PTX/metadata, no weights")
    parser.add_argument("--run-driver", action="store_true", help="Explicitly select NVIDIA Driver reference executor (not RUDA Rust)")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not args.emit_dir and not args.run_driver:
        parser.error("Choose --emit-dir and/or --run-driver")
    torch.manual_seed(17)
    dtype = getattr(torch, args.dtype)
    model = TinyMLP(17, 33, dtype).eval()
    x = torch.randn(3, 17, dtype=dtype)
    plan = compile_exported(torch.export.export(model, (x,)))
    if args.emit_dir:
        plan.write(args.emit_dir)
    print(json.dumps(plan.report(), indent=2))
    if args.run_driver:
        from ruda_ptx.nvidia_driver import NvidiaDriverRuntime
        with NvidiaDriverRuntime(args.device) as runtime, Executor(plan, runtime) as run, torch.inference_mode():
            y = run(x)
            tolerance = {"float32": 5e-4, "float16": 4e-3, "bfloat16": 3e-2}[args.dtype]
            torch.testing.assert_close(y, model(x), rtol=tolerance, atol=tolerance)
            print(json.dumps({"provider": runtime.name, "correctness_check": "passed_this_run", "execution": run.stats}, indent=2))


if __name__ == "__main__":
    main()
