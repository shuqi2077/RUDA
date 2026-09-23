"""Actual-GPU A/B benchmark; no host interpreter or automatic CPU fallback.

Reports synchronized steady-state wall latency (including Python dispatch), NOT
pure kernel duration. JIT, allocation, uploads and readback are outside timing.
Optional PyTorch GPU comparison uses the same shapes/precision on device 0.
"""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from ruda_ptx import TensorSpec, Executor
from ruda_ptx.emitter import rms_norm_reference, matmul
from ruda_ptx.reductions import rms_norm
from ruda_ptx.decode_linear import linear_decode
from ruda_ptx.attention import attention_program
from ruda_ptx.nvidia_driver import NvidiaDriverRuntime


def measure(call, sync, warmup, iterations):
    for _ in range(warmup):
        call()
    sync()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(iterations):
            call()
        sync()
        samples.append((time.perf_counter()-start)*1000/iterations)
    return {"median_ms": statistics.median(samples), "samples_ms": samples,
            "metric": "synchronized_wall_ms_per_iteration_including_dispatch"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["norm", "linear", "attention"], default="norm")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--width", type=int, default=4096)
    p.add_argument("--out-width", type=int, default=4096)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--q-heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--partitions", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--pytorch-baseline", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if min(a.rows, a.width, a.out_width, a.context, a.iterations) < 1 or a.warmup < 0:
        p.error("Shapes/iterations must be positive; warmup must be nonnegative")
    dtype = getattr(torch, a.dtype)
    torch.manual_seed(8401)
    variants, specs = [], []
    if a.case == "norm":
        inputs = (torch.randn(a.rows, a.width, dtype=dtype)*0.3, torch.randn(a.width, dtype=dtype))
        s = TensorSpec(tuple(inputs[0].shape), a.dtype)
        variants = [("v7_block_reduce", (rms_norm_reference("v7_norm", s, 1e-6),), s, 0),
                    ("v8_warp_reduce", (rms_norm("v8_norm", s, 1e-6),), s, 0)]
        fn = lambda x, w: F.rms_norm(x, (a.width,), w, 1e-6)
        expected = (inputs[0].float()/torch.sqrt(inputs[0].float().square().mean(-1, keepdim=True)+1e-6)*inputs[1].float()).to(dtype)
    elif a.case == "linear":
        inputs = (torch.randn(a.rows, a.width, dtype=dtype)*0.1, torch.randn(a.out_width, a.width, dtype=dtype)*0.1)
        sa, sw = [TensorSpec(tuple(x.shape), a.dtype) for x in inputs]
        so = TensorSpec((a.rows, a.out_width), a.dtype)
        variants = [("v7_tiled_gemm", (matmul("v7_linear", sa, sw, transpose_b=True),), so, 0),
                    ("v8_decode_linear", (linear_decode("v8_linear", sa, sw),), so, 0)]
        fn = lambda x, w: F.linear(x, w)
        expected = F.linear(inputs[0].float(), inputs[1].float()).to(dtype)
    else:
        inputs = (torch.randn(1, a.q_heads, 1, a.head_dim, dtype=dtype)*0.3,
                  torch.randn(1, a.kv_heads, a.context, a.head_dim, dtype=dtype)*0.3,
                  torch.randn(1, a.kv_heads, a.context, a.head_dim, dtype=dtype))
        specs = [TensorSpec(tuple(x.shape), a.dtype) for x in inputs]
        for parts in a.partitions:
            pr = attention_program(f"v8_decode_p{parts}", *specs, partitions=parts)
            variants.append((f"v8_decode_partitions_{parts}", pr.kernels, pr.output, pr.workspace_bytes))
        fn = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        expected = fn(*(x.float() for x in inputs)).to(dtype)
    tolerance = 8e-4 if dtype == torch.float32 else (7e-3 if dtype == torch.float16 else 4e-2)
    result = {"case": a.case, "dtype": a.dtype, "input_shapes": [list(x.shape) for x in inputs],
              "iterations_per_batch": a.iterations, "batches": 5, "warmup": a.warmup,
              "results": [], "measured_model_peak_vram": None,
              "warnings": ["Microbenchmark, not full-model tokens/sec", "No CPU performance fallback"]}
    with NvidiaDriverRuntime() as rt:
        result.update(runtime=rt.name, target_device_sm=rt.sm)
        owned = []
        try:
            for x in inputs:
                b = rt.allocate(x.numel()*x.element_size())
                owned.append(b)
                rt.write(b, Executor._tensor_bytes(x))
            for label, kernels, outspec, workspace in variants:
                local = []
                try:
                    out = rt.allocate(outspec.nbytes); local.append(out)
                    temporary = rt.allocate(workspace) if workspace else out
                    if workspace:
                        local.append(temporary)
                    loaded = tuple(rt.load(k) for k in kernels)
                    calls = [(loaded[0], kernels[0], tuple(owned)+(temporary,))]
                    if len(kernels) == 2:
                        calls.append((loaded[1], kernels[1], (temporary, out)))
                    rt.launch_many(calls)
                    rt.synchronize()
                    actual = torch.frombuffer(bytearray(rt.read(out, outspec.nbytes)), dtype=dtype).reshape(outspec.shape)
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    timing = measure(lambda: rt.launch_many(calls), rt.synchronize, a.warmup, a.iterations)
                    result["results"].append({"variant": label, "correctness": "passed",
                        "workspace_bytes": workspace, "explicit_io_bytes": sum(b.nbytes for b in owned)+outspec.nbytes,
                        **timing})
                finally:
                    rt.synchronize()
                    for b in reversed(local):
                        rt.free(b)
        finally:
            for b in reversed(owned):
                rt.free(b)
    if a.pytorch_baseline:
        # Explicit reference-only path; not part of PTX compilation/execution.
        device_inputs = tuple(x.to("cuda:0") for x in inputs)
        with torch.inference_mode():
            actual = fn(*device_inputs).to("cpu")
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            timing = measure(lambda: fn(*device_inputs), lambda: torch.cuda.synchronize(0), a.warmup, a.iterations)
        result["results"].append({"variant": "pytorch_gpu_default_kernels", "correctness": "passed", **timing})
    text = json.dumps(result, indent=2)
    if a.output:
        a.output.write_text(text+"\n")
    print(text)


if __name__ == "__main__":
    main()
