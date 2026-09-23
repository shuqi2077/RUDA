"""Export a SMALL supported PyTorch graph to actual AMD ISA; execution opt-in.
Not a transformer, full MLP, or arbitrary-model compatibility demonstration.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from ruda_ptx import Executor, NativePlanRuntime, compile_exported
from ruda_ptx.amdgpu_isa import AmdGpuCompiler, lower_plan_amdgpu


class Activation(torch.nn.Module):
    def forward(self,x,y):
        return F.silu(x+y)*y


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target",required=True,choices=["gfx90a","gfx942","gfx1100"])
    parser.add_argument("--emit-dir",required=True,type=Path)
    parser.add_argument("--cache-dir",type=Path)
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    if args.emit_dir.exists() and any(args.emit_dir.iterdir()):raise FileExistsError("Output directory is nonempty")
    torch.manual_seed(10)
    x,y=torch.randn(1,1025),torch.randn(1,1025)
    model=Activation().eval()
    plan=compile_exported(torch.export.export(model,(x,y)))
    compiler=AmdGpuCompiler(args.target,cache_dir=args.cache_dir)
    images=lower_plan_amdgpu(plan,compiler)
    args.emit_dir.mkdir(parents=True,exist_ok=True)
    for image in images.values():
        image.write(args.emit_dir/(image.entry+".risa"))
        (args.emit_dir/(image.entry+".hsaco")).write_bytes(image.code)
    report={"source":"torch.export","isa":"amdgcn","target":args.target,
            "compiled_kernels":len(images),"runtime":"not_invoked","rust_runtime_connected":False,
            "full_model_supported":False,"gpu_execution_verified":False}
    if args.run:
        from ruda_ptx.hip_isa import HipIsaRuntime
        with HipIsaRuntime(args.target) as native:
            with Executor(plan,NativePlanRuntime(native,images)) as executable,torch.inference_mode():
                out=executable(x,y)
                torch.testing.assert_close(out,model(x,y),rtol=2e-4,atol=2e-5)
                report.update(runtime=native.name,gpu_execution_verified=True,
                              scope="only this two-kernel FP32 example",executor_stats=executable.stats)
    (args.emit_dir/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
