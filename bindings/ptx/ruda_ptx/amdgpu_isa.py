"""Small native AMDGPU ISA backend, lowered from operators, NOT from PTX text.

Implemented: contiguous FP32 add, mul, SiLU, SiLU*up, scalar or 4-wide memory.
LLVM emits native AMD assembly and ELF; LLD links AMDHSA ET_DYN (.hsaco).
Unsupported dtypes/operators/targets fail before any GPU work. No HIP C++ source.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import sys
from .emitter import TensorSpec
from .isa import AMD_TARGETS, NativeCache, NativeImage, digest_json

_LLVM_LOCK = threading.RLock()
OPS = ("add", "mul", "silu", "silu_mul")


def elementwise_ir(name: str, operation: str, spec: TensorSpec, *, vector_width: int = 4):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError("Invalid native kernel name")
    if not isinstance(spec, TensorSpec) or spec.dtype != "float32":
        raise ValueError("AMDGPU v10 lowering currently supports FP32 only")
    if operation not in OPS:
        raise ValueError(f"AMDGPU operation not implemented: {operation}")
    if type(vector_width) is not int or vector_width not in (1, 4):
        raise ValueError("AMDGPU vector_width must be 1 or 4")
    params = ("x", "out") if operation == "silu" else ("x", "y", "out")
    arguments = ", ".join("ptr addrspace(1) %" + p for p in params)
    # Hex floats in LLVM IR encode the exact F32 value in F64 representation.
    scale = struct.unpack("<f", struct.pack("<f", 1.4426950408889634))[0]
    scale_hex = "0x" + struct.pack(">d", scale).hex().upper()
    lines = [
        'target triple = "amdgcn-amd-amdhsa"',
        f"define amdgpu_kernel void @{name}({arguments}) #0 !reqd_work_group_size !0 {{",
        "entry:",
        "  %tid = call i32 @llvm.amdgcn.workitem.id.x()",
        "  %bid = call i32 @llvm.amdgcn.workgroup.id.x()",
        "  %bid64 = zext i32 %bid to i64",
        "  %tid64 = zext i32 %tid to i64",
        "  %groupbase = mul i64 %bid64, 256",
        "  %thread = add i64 %groupbase, %tid64",
        f"  %start = mul i64 %thread, {vector_width}",
    ]

    def calculate(prefix: str, index: str, width: int):
        typ = "float" if width == 1 else "<4 x float>"
        const = lambda x: x if width == 1 else "<" + ", ".join("float " + x for _ in range(4)) + ">"
        for p in params:
            lines.append(f"  %{prefix}_{p}p = getelementptr float, ptr addrspace(1) %{p}, i64 {index}")
        for p in params[:-1]:
            lines.append(f"  %{prefix}_{p} = load {typ}, ptr addrspace(1) %{prefix}_{p}p, align {4*width}")
        val = f"%{prefix}_x"
        if operation in ("silu", "silu_mul"):
            lines.extend([
                f"  %{prefix}_neg = fneg {typ} {val}",
                f"  %{prefix}_scaled = fmul {typ} %{prefix}_neg, {const(scale_hex)}",
                f"  %{prefix}_exp = call {typ} @llvm.exp2.{'f32' if width == 1 else 'v4f32'}({typ} %{prefix}_scaled)",
                f"  %{prefix}_denom = fadd {typ} %{prefix}_exp, {const('1.0')}",
                f"  %{prefix}_silu = fdiv {typ} {val}, %{prefix}_denom",
            ])
            val = f"%{prefix}_silu"
        if operation != "silu":
            instruction = "fadd" if operation == "add" else "fmul"
            lines.append(f"  %{prefix}_result = {instruction} {typ} {val}, %{prefix}_y")
            val = f"%{prefix}_result"
        lines.append(f"  store {typ} {val}, ptr addrspace(1) %{prefix}_outp, align {4*width}")

    if vector_width == 4:
        lines.extend(["  %end = add i64 %start, 4", f"  %full = icmp ule i64 %end, {spec.numel}",
                      "  br i1 %full, label %vector, label %tail0", "vector:"])
        calculate("v", "%start", 4)
        lines.append("  ret void")
    else:
        lines.append("  br label %tail0")
    # Incomplete final chunk: each scalar access is guarded; no padded buffers
    # or out-of-range vector loads. No barrier crosses these divergent branches.
    for j in range(vector_width):
        next_label = f"tail{j+1}" if j+1 < vector_width else "done"
        lines.extend([f"tail{j}:", f"  %index{j} = add i64 %start, {j}",
                      f"  %valid{j} = icmp ult i64 %index{j}, {spec.numel}",
                      f"  br i1 %valid{j}, label %scalar{j}, label %done", f"scalar{j}:"])
        calculate(f"s{j}", f"%index{j}", 1)
        lines.append(f"  br label %{next_label}")
    lines.extend(["done:", "  ret void", "}",
                  "declare i32 @llvm.amdgcn.workitem.id.x()",
                  "declare i32 @llvm.amdgcn.workgroup.id.x()",
                  "declare float @llvm.exp2.f32(float)",
                  "declare <4 x float> @llvm.exp2.v4f32(<4 x float>)",
                  'attributes #0 = { nounwind "amdgpu-flat-work-group-size"="256,256" }',
                  "!0 = !{i32 256, i32 1, i32 1}"])
    return "\n".join(lines) + "\n", params


@dataclass(frozen=True)
class AmdBuild:
    image: NativeImage
    llvm_ir: str
    assembly: str
    object_code: bytes
    linker_log: str

    def write(self, directory):
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Refusing to overwrite a nonempty ISA output directory")
        directory.mkdir(parents=True, exist_ok=True)
        name = self.image.entry
        (directory / (name + ".ll")).write_text(self.llvm_ir)
        (directory / (name + ".s")).write_text(self.assembly)
        (directory / (name + ".o")).write_bytes(self.object_code)
        (directory / (name + ".hsaco")).write_bytes(self.image.code)
        self.image.write(directory / (name + ".risa"))
        (directory / "metadata.json").write_text(json.dumps({**self.image.metadata(),
            "llvm_ir_verified": True, "native_object_emitted": True, "hsaco_linked": True,
            "gpu_execution_verified": False, "rust_runtime_connected": False}, indent=2))


class AmdGpuCompiler:
    def __init__(self, target: str, *, linker: str | None = None, cache_dir=None):
        if target not in AMD_TARGETS:
            raise ValueError(f"Unsupported AMD codegen target; choose {tuple(AMD_TARGETS)}")
        try:
            from llvmlite import binding as llvm
        except ImportError as exc:
            raise RuntimeError("AMDGPU ISA emission requires the optional llvmlite ISA dependency") from exc
        if llvm.llvm_version_info[0] != 20:
            raise RuntimeError("This AMDGPU candidate is validated with LLVM 20; refusing an unvalidated compiler major")
        self.llvm, self.target = llvm, target
        self.linker = shutil.which(linker or "ld.lld")
        if self.linker is None:
            raise RuntimeError("ld.lld is required to link a loadable .hsaco; a relocatable .o is not sufficient")
        version = subprocess.run([self.linker, "--version"], capture_output=True, text=True,
                                 timeout=10, check=True).stdout.strip()
        self.compiler_id = f"ruda-amdgpu-v1:LLVM-{'.'.join(map(str, llvm.llvm_version_info))}:{version}"
        self.cache = NativeCache(cache_dir) if cache_dir is not None else None
        with _LLVM_LOCK:
            llvm.initialize_all_targets()
            llvm.initialize_all_asmprinters()

    def build(self, name: str, operation: str, spec: TensorSpec, *, vector_width=4,
              source_digest=None) -> AmdBuild:
        ir, parameters = elementwise_ir(name, operation, spec, vector_width=vector_width)
        source_digest = source_digest or digest_json({"ir": ir})
        with tempfile.TemporaryDirectory(prefix="ruda-amdgpu-") as tmp:
            directory = Path(tmp)
            (directory / "kernel.ll").write_text(ir)
            worker = subprocess.run([sys.executable, str(Path(__file__).with_name("_llvm_worker.py")),
                                     self.target, tmp], capture_output=True, text=True, timeout=45)
            if worker.returncode:
                raise RuntimeError(f"AMDGPU compiler worker failed ({worker.returncode}):\n" + worker.stderr)
            assembly = (directory / "kernel.s").read_text()
            obj = (directory / "kernel.o").read_bytes()
            out = directory / "kernel.hsaco"
            result = subprocess.run([self.linker, "-shared", "--no-undefined", str(directory / "kernel.o"), "-o", str(out)],
                                    capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError("AMDGPU link failed:\n" + result.stderr)
            code = out.read_bytes()
        image = NativeImage("amdgcn", self.target, name, parameters,
                            ((spec.numel + 256*vector_width - 1)//(256*vector_width), 1, 1),
                            (256, 1, 1), source_digest, self.compiler_id, code,
                            (spec.nbytes,)*len(parameters), 4*vector_width)
        return AmdBuild(image, ir, assembly, obj, result.stdout + result.stderr)

    def compile(self, name, operation, spec, *, vector_width=4, source_digest=None):
        ir, _ = elementwise_ir(name, operation, spec, vector_width=vector_width)
        source_digest = source_digest or digest_json({"ir": ir})
        identity = {"isa": "amdgcn", "target": self.target, "source_digest": source_digest,
                    "compiler_id": self.compiler_id, "ir_sha256": digest_json({"ir": ir})}
        build = lambda: self.build(name, operation, spec, vector_width=vector_width,
                                   source_digest=source_digest).image
        return self.cache.get_or_compile(identity, build) if self.cache is not None else build()


def lower_plan_amdgpu(plan, compiler: AmdGpuCompiler, *, vector_width=4):
    """Lower a COMPLETE supported plan, or fail before compiling/uploading weights.

    Attention, linear, normalization, FP16/BF16 and arbitrary PTX are not
    translated by this small v10 backend. No partial execution/fallback.
    """
    for step in plan.steps:
        if step.kernel.operation not in OPS:
            raise ValueError(f"AMDGPU plan lowering not implemented: {step.kernel.operation}")
        output = plan.specs[step.output]
        if output.dtype != "float32" or any(plan.specs[n] != output for n in step.inputs):
            raise ValueError("AMDGPU plans require same-shape contiguous FP32 elementwise operations")
    return {step.kernel.digest: compiler.compile(step.kernel.name, step.kernel.operation,
            plan.specs[step.output], vector_width=vector_width, source_digest=step.kernel.digest)
            for step in plan.steps}
