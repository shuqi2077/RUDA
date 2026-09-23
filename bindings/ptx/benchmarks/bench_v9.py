#!/usr/bin/env python3
"""Compare retained v8 vs candidate v9 PTX on an actual GPU, no CUDA compiler.

Inputs/weights are uploaded before timing. Timings include Python/driver launch
submission + stream completion; these are NOT event-only device timings.
Workspace sizes are explicit buffer accounting, NOT measured peak VRAM.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import time
import torch
from ruda_ptx import Executor, TensorSpec
from ruda_ptx.emitter import elementwise
from ruda_ptx.decode_linear import linear_decode, gated_linear
from ruda_ptx.decode_linear_v8 import linear_decode as old_linear
from ruda_ptx.partitioned_selection import partitioned_topk
from ruda_ptx.selection import topk


def tensor_spec(x):
    return TensorSpec(tuple(x.shape), str(x.dtype).removeprefix('torch.'))


@contextmanager
def allocations(rt):
    owned = []
    def alloc(n):
        b = rt.allocate(n); owned.append(b); return b
    def upload(x):
        b = alloc(tensor_spec(x).nbytes)
        rt.write(b, Executor._tensor_bytes(x.contiguous()))
        return b
    try:
        yield alloc, upload
    finally:
        rt.synchronize()
        for b in reversed(owned):
            rt.free(b)


def measure(rt, calls, args):
    for _ in range(args.warmup):
        rt.launch_many(calls)
    rt.synchronize()
    trials = []
    for _ in range(args.trials):
        start = time.perf_counter_ns()
        for _ in range(args.iterations):
            rt.launch_many(calls)
        rt.synchronize()
        trials.append((time.perf_counter_ns()-start)/args.iterations/1000)
    return {'median_us': statistics.median(trials), 'trial_us': trials,
            'launches_per_iteration': len(calls)}


def fp_read(rt, b, shape, dtype):
    count = TensorSpec(shape, str(dtype).removeprefix('torch.')).nbytes
    return torch.frombuffer(bytearray(rt.read(b, count)), dtype=dtype).reshape(shape)


def check_float(actual, expected, dtype):
    tolerance = 1e-3 if dtype == torch.float32 else (1e-2 if dtype == torch.float16 else 6e-2)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def bench_linear(rt, a, dtype):
    x = torch.randn(a.rows, a.width, dtype=dtype)*0.1
    w = torch.randn(a.hidden, a.width, dtype=dtype)*0.1
    expected = torch.nn.functional.linear(x.float(), w.float()).to(dtype)
    with allocations(rt) as (alloc, upload):
        dx, dw = upload(x), upload(w)
        output = alloc(a.rows*a.hidden*x.element_size())
        kernels = [('v8_warp_one_output', old_linear('v8_linear', tensor_spec(x), tensor_spec(w)))]
        kernels += [(f'v9_outputs_per_warp_{tile}', linear_decode(f'v9_linear_{tile}', tensor_spec(x), tensor_spec(w), outputs_per_warp=tile)) for tile in (1,2,4)]
        result = []
        for label, kernel in kernels:
            calls = [(rt.load(kernel), kernel, (dx,dw,output))]
            rt.launch_many(calls); rt.synchronize()
            check_float(fp_read(rt, output, (a.rows,a.hidden), dtype), expected, dtype)
            result.append({'variant': label, 'output_bytes': output.nbytes,
                           'extra_workspace_bytes': 0, **measure(rt,calls,a)})
        return result


def bench_gated(rt, a, dtype):
    x = torch.randn(a.rows, a.width, dtype=dtype)*0.1
    wg, wu = torch.randn(a.hidden,a.width,dtype=dtype)*0.1, torch.randn(a.hidden,a.width,dtype=dtype)*0.1
    with allocations(rt) as (alloc, upload):
        dx, dg, du = upload(x), upload(wg), upload(wu)
        shape = TensorSpec((a.rows,a.hidden), str(dtype).removeprefix('torch.'))
        g, u, output, fused_output = (alloc(shape.nbytes) for _ in range(4))
        kg, ku = old_linear('old_gate',tensor_spec(x),tensor_spec(wg)), old_linear('old_up',tensor_spec(x),tensor_spec(wu))
        km = elementwise('old_activation','silu_mul',shape)
        old = [(rt.load(kg),kg,(dx,dg,g)),(rt.load(ku),ku,(dx,du,u)),(rt.load(km),km,(g,u,output))]
        kf = gated_linear('fused_gate',tensor_spec(x),tensor_spec(wg))
        new = [(rt.load(kf),kf,(dx,dg,du,fused_output))]
        rt.launch_many(old); rt.launch_many(new); rt.synchronize()
        expected = fp_read(rt,output,shape.shape,dtype)
        check_float(fp_read(rt,fused_output,shape.shape,dtype),expected,dtype)
        return [{'variant':'v8_two_projections_plus_activation','extra_workspace_bytes':2*shape.nbytes,**measure(rt,old,a)},
                {'variant':'v9_fused_gated_decode','extra_workspace_bytes':0,**measure(rt,new,a)}]


def bench_selection(rt, a, dtype):
    x = torch.randn(a.rows,a.vocab,dtype=dtype)
    expected = torch.argsort(x,dim=-1,descending=True,stable=True)[...,:a.topk]
    with allocations(rt) as (alloc, upload):
        dx = upload(x)
        out_i, out_v = alloc(a.rows*a.topk*4), alloc(a.rows*a.topk*4)
        old = topk('old_topk',tensor_spec(x),a.topk)
        program = partitioned_topk('new_topk',tensor_spec(x),a.topk,partitions=a.partitions)
        pi,pv = alloc(program.partial_nbytes),alloc(program.partial_nbytes)
        variants = [('v8_single_block_row',[(rt.load(old.kernel),old.kernel,(dx,out_i,out_v))],0),
                    ('v9_partitioned',[(rt.load(program.first),program.first,(dx,pi,pv)),
                                       (rt.load(program.merge),program.merge,(pv,pi,out_i,out_v))],program.workspace_nbytes)]
        result = []
        for label,calls,workspace in variants:
            rt.launch_many(calls); rt.synchronize()
            indices = torch.frombuffer(bytearray(rt.read(out_i,out_i.nbytes)),dtype=torch.int32).reshape(a.rows,a.topk)
            assert torch.equal(indices.long(),expected), f'{label}: incorrect indices'
            values = fp_read(rt,out_v,(a.rows,a.topk),torch.float32)
            torch.testing.assert_close(values,x.gather(-1,expected).float(),rtol=0,atol=0)
            result.append({'variant':label,'extra_workspace_bytes':workspace,**measure(rt,calls,a)})
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['linear','gated','topk','all'],default='all')
    p.add_argument('--dtype',choices=['float32','float16','bfloat16'],default='float16')
    p.add_argument('--rows',type=int,default=1)
    p.add_argument('--width',type=int,default=4096)
    p.add_argument('--hidden',type=int,default=8192)
    p.add_argument('--vocab',type=int,default=65536)
    p.add_argument('--topk',type=int,default=1)
    p.add_argument('--partitions',type=int,default=8)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--trials',type=int,default=5)
    p.add_argument('--output',type=Path)
    a = p.parse_args()
    if not 1 <= a.rows <= 4 or any(v <= 0 for v in (a.width,a.hidden,a.vocab,a.iterations,a.trials)) or a.warmup < 0:
        p.error('rows must be 1..4; sizes/iterations/trials positive; warmup nonnegative')
    if not 1 <= a.topk <= min(8,a.vocab) or not 1 <= a.partitions <= 64:
        p.error('topk must be 1..min(8,vocab); partitions must be 1..64')
    if a.output and a.output.exists():
        p.error('Refusing to overwrite existing output')
    from ruda_ptx.nvidia_driver import NvidiaDriverRuntime
    torch.manual_seed(2026)
    with NvidiaDriverRuntime() as rt:
        functions = {'linear':bench_linear,'gated':bench_gated,'topk':bench_selection}
        selected = functions if a.case == 'all' else {a.case:functions[a.case]}
        results = {name:fn(rt,a,getattr(torch,a.dtype)) for name,fn in selected.items()}
        report = {'package':'0.9.0','runtime':rt.name,'device_sm':rt.sm,
                  'timing_scope':'host submission + GPU completion, preallocated and preloaded',
                  'peak_vram_measured':False,'rust_runtime_connected':False,
                  'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'results':results}
    text = json.dumps(report,indent=2)
    if a.output:
        with a.output.open('x') as out:
            out.write(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
