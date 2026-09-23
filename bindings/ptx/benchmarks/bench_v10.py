"""GPU-required scalar/vector and prepared-launch comparisons, no CPU substitute.
Times are synchronized HOST wall time (include submission overhead), not pure
GPU event times. No claimed winner: compare sizes on your actual device.
"""
import argparse
import json
from pathlib import Path
import time
import torch
from ruda_ptx import TensorSpec
from ruda_ptx.emitter import elementwise
from ruda_ptx.vectorized import elementwise4
from ruda_ptx.nvidia_isa import NvidiaIsaRuntime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--elements",type=int,default=1048576)
    p.add_argument("--repeats",type=int,default=100)
    p.add_argument("--cache-dir",required=True,type=Path)
    args=p.parse_args()
    if args.repeats<1:raise ValueError("repeats must be positive")
    spec=TensorSpec((args.elements,))
    x,y=torch.randn(args.elements),torch.randn(args.elements)
    report={"elements":args.elements,"repeats":args.repeats,"units":"synchronized_host_wall_ms_per_launch","cases":[]}
    with NvidiaIsaRuntime(cache_dir=args.cache_dir) as rt:
        a,b,out=(rt.allocate(spec.nbytes) for _ in range(3))
        rt.write(a,x.numpy().tobytes());rt.write(b,y.numpy().tobytes())
        for label,emitter in (("scalar",elementwise),("vector4",elementwise4)):
            kernel=emitter("bench_"+label,"silu_mul",spec)
            start=time.perf_counter();loaded=rt.load(kernel);load_ms=(time.perf_counter()-start)*1000
            rt.launch(loaded,kernel,(a,b,out));rt.synchronize()
            actual=torch.frombuffer(bytearray(rt.read(out,spec.nbytes)),dtype=torch.float32)
            torch.testing.assert_close(actual,torch.nn.functional.silu(x)*y,atol=2e-5,rtol=2e-4)
            prepared=rt.prepare_launches([(loaded,kernel,(a,b,out))])
            for mode in ("ordinary","prepared"):
                launch=(lambda:rt.launch(loaded,kernel,(a,b,out))) if mode=="ordinary" else prepared.replay
                for _ in range(5):launch()
                rt.synchronize();start=time.perf_counter()
                for _ in range(args.repeats):launch()
                rt.synchronize()
                report["cases"].append({"kernel":label,"submission":mode,"load_wall_ms":load_ms,
                    "mean_wall_ms":(time.perf_counter()-start)*1000/args.repeats,"grid":kernel.grid})
            prepared.close()
        report.update(target=f"sm_{rt.sm}",cache_stats=rt.native_cache.stats,isa_stats=rt.isa_stats)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
