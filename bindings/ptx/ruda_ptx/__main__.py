"""Explicit PTX/native ISA diagnostics and code emission; no hidden backend."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import struct
from . import __version__


def capabilities():
    return {"package_version":__version__, "rust_runtime_connected":False,
            "cpu_compute_fallback":False, "gpu_execution_verified_by_package":False,
            "backends":{
                "ptx":{"representation":"virtual ISA text", "scope":"retained v9 operators plus opt-in FP32 vectors"},
                "nvidia-sass":{"representation":"native cubin", "compiler":"NVIDIA driver JIT linker",
                               "scope":"existing supported PTX kernels; explicit exact-SM image loading/cache"},
                "amdgcn":{"representation":"LLVM native assembly + linked AMDHSA .hsaco",
                           "targets":["gfx90a","gfx942","gfx1100"], "dtypes":["float32"],
                           "operations":["add","mul","silu","silu_mul"], "loader":"explicit HIP module reference"}},
            "not_implemented":["arbitrary PTX-to-AMDGPU translation", "Intel native ISA",
                               "AMD full attention/GEMM/MLA/MoE", "full-model validation"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=["doctor","isa-info","emit-isa"])
    p.add_argument("--backend",choices=["ptx","nvidia-sass","amdgcn"],default="ptx")
    p.add_argument("--require-gpu",action="store_true")
    p.add_argument("--target",help="Explicit AMD gfx target; NVIDIA native target is the actual device SM")
    p.add_argument("--cache-dir",type=Path)
    p.add_argument("--emit-dir",type=Path)
    p.add_argument("--op",choices=["add","mul","silu","silu_mul"],default="silu_mul")
    p.add_argument("--elements",type=int,default=1025)
    p.add_argument("--vector-width",type=int,choices=[1,4],default=4)
    args=p.parse_args()
    if args.command=="isa-info":
        print(json.dumps(capabilities(),indent=2));return 0
    result={"package_version":__version__,"torch_version":importlib.metadata.version("torch"),
            "kernel_format":"PTX" if args.backend=="ptx" else args.backend,
            "rust_runtime_connected":False,"cpu_compute_fallback":False,"gpu_test":"not_requested"}
    try:
        from .emitter import elementwise,TensorSpec
        from .vectorized import elementwise4
        if args.command=="emit-isa":
            if args.emit_dir is None:raise ValueError("--emit-dir is required")
            if args.emit_dir.exists() and any(args.emit_dir.iterdir()):raise FileExistsError("Output directory is not empty")
            spec=TensorSpec((args.elements,))
            name="ruda_"+args.op
            if args.backend=="amdgcn":
                if args.target is None:raise ValueError("AMDGPU emission requires --target gfx90a/gfx942/gfx1100")
                from .amdgpu_isa import AmdGpuCompiler
                build=AmdGpuCompiler(args.target).build(name,args.op,spec,vector_width=args.vector_width)
                build.write(args.emit_dir)
                result.update(native_object_emitted=True,hsaco_linked=True,target=args.target)
            else:
                kernel=(elementwise4 if args.vector_width==4 else elementwise)(name,args.op,spec)
                args.emit_dir.mkdir(parents=True,exist_ok=True)
                (args.emit_dir/(name+".ptx")).write_text(kernel.ptx)
                if args.backend=="nvidia-sass":
                    from .nvidia_isa import NvidiaIsaRuntime
                    with NvidiaIsaRuntime(cache_dir=args.cache_dir) as rt:
                        image=rt.compile_native(kernel)
                        image.write(args.emit_dir/(name+".risa"))
                        (args.emit_dir/(name+".cubin")).write_bytes(image.code)
                        result.update(native_object_emitted=True,target=image.target)
                else:result.update(ptx_emitted=True,native_object_emitted=False)
                (args.emit_dir/"metadata.json").write_text(json.dumps(result,indent=2))
            result["output_directory"]=str(args.emit_dir)
        elif args.require_gpu:
            if args.backend=="amdgcn":
                if args.target is None:raise ValueError("AMD GPU execution requires an explicit --target")
                from .hip_isa import HipIsaRuntime
                from .amdgpu_isa import AmdGpuCompiler
                image=AmdGpuCompiler(args.target,cache_dir=args.cache_dir).compile("ruda_doctor_add","add",TensorSpec((4,)))
                runtime=HipIsaRuntime(args.target)
            elif args.backend=="nvidia-sass":
                from .nvidia_isa import NvidiaIsaRuntime
                runtime=NvidiaIsaRuntime(cache_dir=args.cache_dir)
            else:
                from .nvidia_driver import NvidiaDriverRuntime
                runtime=NvidiaDriverRuntime()
            with runtime as rt:
                result["runtime"]=rt.name
                if hasattr(rt,"sm"):result["device_sm"]=rt.sm
                if hasattr(rt,"target_probe"):result["target_probe"]=rt.target_probe
                kernel=elementwise("ruda_doctor_add","add",TensorSpec((4,)))
                x,y=rt.allocate(16),rt.allocate(16)
                try:
                    rt.write(x,struct.pack("<4f",1,2,3,4))
                    if args.backend=="amdgcn":rt.launch_image(rt.load_image(image),image,(x,x,y))
                    else:rt.launch(rt.load(kernel),kernel,(x,x,y))
                    output=struct.unpack("<4f",rt.read(y,16))
                    if output!=(2,4,6,8):raise RuntimeError(f"GPU add smoke test failed: {output}")
                    result["gpu_test"]="four_element_add_passed_only_not_full_model_validation"
                finally:
                    rt.free(y);rt.free(x)
        print(json.dumps(result,indent=2));return 0
    except Exception as exc:
        result.update(error=str(exc),gpu_test="failed" if args.require_gpu else "not_executed")
        print(json.dumps(result,indent=2));return 1


if __name__=="__main__":
    raise SystemExit(main())
