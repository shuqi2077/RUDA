"""ISA contracts/compiler. Offline machine-code emission is not GPU execution."""
from dataclasses import replace
import ctypes as C
import json
import os
import shutil
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import pytest
import numpy as np
import torch
from ruda_ptx import TensorSpec, NativeImage, NativeCache, NativePlanRuntime, Executor, compile_exported
from ruda_ptx.isa import elf_info, AMD_TARGETS, digest_json
from ruda_ptx.amdgpu_isa import AmdGpuCompiler, elementwise_ir, lower_plan_amdgpu
from ruda_ptx.emitter import elementwise
from ruda_ptx.nvidia_isa import NvidiaIsaRuntime, PreparedLaunches
from ruda_ptx.nvidia_driver import DriverError
from ptx_debug import DebugRuntime


@pytest.fixture(scope="module")
def amd_build():
    pytest.importorskip("llvmlite")
    if shutil.which("ld.lld") is None:
        if os.environ.get("RUDA_REQUIRE_ISA") == "1": pytest.fail("ISA compiler required")
        pytest.skip("ld.lld unavailable")
    return AmdGpuCompiler("gfx90a").build("ruda_native", "add", TensorSpec((17,)))


@pytest.mark.isa
@pytest.mark.parametrize("target", list(AMD_TARGETS))
@pytest.mark.parametrize("operation", ["add", "mul", "silu", "silu_mul"])
@pytest.mark.parametrize("width", [1, 4])
def test_real_native_compile(target, operation, width):
    pytest.importorskip("llvmlite")
    build = AmdGpuCompiler(target).build("ruda_native", operation, TensorSpec((1025,)), vector_width=width)
    assert f'amdgcn-amd-amdhsa--{target}' in build.assembly
    info = elf_info(build.image.code)
    assert info["machine"] == 224 and info["type"] == 3
    assert build.image.entry in info["symbols"]
    assert build.image.grid == ((1025+256*width-1)//(256*width),1,1)
    if width == 4:
        assert "load <4 x float>" in build.llvm_ir
        assert "load_dwordx4" in build.assembly or "load_b128" in build.assembly
    assert NativeImage.from_bytes(build.image.to_bytes()) == build.image


@pytest.mark.isa
@pytest.mark.parametrize("operation", ["add", "mul", "silu", "silu_mul"])
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("n", [1, 3, 4, 5, 255, 257, 1025])
def test_llvm_host_semantics_only(operation, width, n):
    pytest.importorskip("llvmlite")
    from llvm_host_debug import execute_ir
    rng = np.random.default_rng(n)
    x, y = rng.uniform(-5, 5, n).astype(np.float32), rng.uniform(-2, 2, n).astype(np.float32)
    ir, _ = elementwise_ir("ruda_host_debug", operation, TensorSpec((n,)), vector_width=width)
    actual = execute_ir(ir, "ruda_host_debug", (x,) if operation == "silu" else (x,y), n, width)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    expected = {"add":tx+ty, "mul":tx*ty, "silu":torch.nn.functional.silu(tx),
                "silu_mul":torch.nn.functional.silu(tx)*ty}[operation]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("operation,dtype,width", [("linear","float32",4),("add","float16",4),
                                                  ("add","bfloat16",4),("add","float32",2)])
def test_unsupported_amd_lowering_fails(operation,dtype,width):
    with pytest.raises(ValueError): elementwise_ir("test",operation,TensorSpec((4,),dtype),vector_width=width)


def test_native_artifact_roundtrip_and_tamper(amd_build,tmp_path):
    image=amd_build.image
    path=tmp_path/"native.risa"; image.write(path)
    assert NativeImage.read(path)==image
    with pytest.raises(FileExistsError): image.write(path)
    raw=json.loads(image.to_bytes()); raw["code_sha256"]="0"*64
    with pytest.raises(ValueError,match="checksum"): NativeImage.from_bytes(json.dumps(raw).encode())
    with pytest.raises(ValueError,match="target/ELF"): replace(image,target="gfx942")
    with pytest.raises(ValueError,match="not defined"): replace(image,entry="missing")
    with pytest.raises(ValueError): replace(image,isa="intel-xe")
    with pytest.raises(ValueError): replace(image,code=amd_build.object_code)


@pytest.mark.parametrize("delta", ["class", "machine", "sections", "target"])
def test_bad_elf_rejected(amd_build,delta):
    code=bytearray(amd_build.image.code)
    if delta=="class": code[4]=1
    elif delta=="machine": struct.pack_into("<H",code,18,62)
    elif delta=="sections": struct.pack_into("<Q",code,40,2**63)
    else: struct.pack_into("<I",code,48,0)
    with pytest.raises(ValueError): replace(amd_build.image,code=bytes(code))


def test_cache_reuses_once_and_checks_identity(amd_build,tmp_path):
    cache=NativeCache(tmp_path/"cache"); image=amd_build.image
    identity={"isa":image.isa,"target":image.target,"compiler_id":image.compiler_id,"source_digest":image.source_digest}
    with ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(lambda _:cache.get_or_compile(identity,lambda:image),range(8)))
    assert all(v==image for v in result)
    assert cache.stats=={"hits":7,"misses":1,"compiles":1}
    entry=next(cache.directory.glob("*.risa")); raw=json.loads(entry.read_bytes());raw["target"]="gfx942"
    entry.write_text(json.dumps(raw))
    with pytest.raises(ValueError): cache.get_or_compile(identity,lambda:image)


def test_cache_symlink_rejected(amd_build,tmp_path):
    real=tmp_path/"real"; real.mkdir(); (tmp_path/"link").symlink_to(real,target_is_directory=True)
    with pytest.raises(ValueError,match="symlink"): NativeCache(tmp_path/"link")
    cache=NativeCache(tmp_path/"cache")
    identity={"isa":"amdgcn"}
    (cache.directory/(digest_json(identity)+".risa")).symlink_to(tmp_path/"outside")
    with pytest.raises(ValueError,match="symlink"): cache.get_or_compile(identity,lambda:amd_build.image)


def test_native_plan_rejects_missing_kernel():
    k=elementwise("missing","add",TensorSpec((4,)))
    adapter=NativePlanRuntime(DebugRuntime(),{})
    with pytest.raises(ValueError,match="Missing native"): adapter.load(k)


def test_amd_plan_rejects_unsupported_before_compilation():
    plan=compile_exported(torch.export.export(torch.nn.Linear(4,4),(torch.randn(1,4),)))
    class NoCompiler:
        def compile(self,*a,**k): pytest.fail("unsupported plan must not start compilation")
    with pytest.raises(ValueError,match="not implemented"): lower_plan_amdgpu(plan,NoCompiler())


def fake_nvidia(amd_build):
    """MOCK ONLY: rewritten ELF header exercises ABI code, NEVER executable."""
    kernel=elementwise("ruda_native","add",TensorSpec((17,)))
    raw=bytearray(amd_build.image.code);struct.pack_into("<H",raw,18,190)
    rt=object.__new__(NvidiaIsaRuntime)
    rt.sm=80;rt.optimization=4;rt.compiler_id="mock-driver-not-gpu";rt.isa_stats={"driver_compiles":0,"native_module_loads":0}
    rt._lock=threading.RLock();rt._guard=lambda:nullcontext();rt._images={};rt._modules={};rt.native_cache=None
    rt._check=lambda rc,op: (_ for _ in ()).throw(DriverError("mock "+op)) if rc else None
    rt._raw=C.create_string_buffer(bytes(raw),len(raw));rt._destroyed=False
    def create(n,opts,vals,out):
        assert list(opts)==[5,6,7,9]
        C.cast(out,C.POINTER(C.c_void_p))[0]=C.c_void_p(1)
        return 0
    def add(state,kind,source,size,name,*args):
        assert kind==1 and C.string_at(source,size).endswith(b"\0")
        assert size==len(kernel.ptx.encode())+1
        return 0
    def complete(state,pointer,size):
        C.cast(pointer,C.POINTER(C.c_void_p))[0]=C.cast(rt._raw,C.c_void_p)
        C.cast(size,C.POINTER(C.c_size_t))[0]=len(raw)
        return 0
    def destroy(state):
        rt._destroyed=True; rt._raw[0]=b"X";return 0
    rt._cuLinkCreate=create;rt._cuLinkAddData=add;rt._cuLinkComplete=complete;rt._cuLinkDestroy=destroy
    return rt,kernel


def test_driver_link_copies_before_destroy_mock_only(amd_build):
    rt,kernel=fake_nvidia(amd_build)
    image=rt.compile_native(kernel)
    assert rt._destroyed and image.code.startswith(b"\x7fELF")
    assert rt.isa_stats["driver_compiles"]==1
    assert rt.compile_native(kernel) is image
    with pytest.raises(ValueError,match="does not match"): rt.load_image(kernel,replace(image,target="sm_90"))


def test_driver_failure_destroys_link_state_mock_only(amd_build):
    rt,kernel=fake_nvidia(amd_build);rt._cuLinkAddData=lambda *a:1
    with pytest.raises(DriverError,match="cuLinkAddData"):rt.compile_native(kernel)
    assert rt._destroyed


def test_prepared_argument_storage_is_reused():
    rt=DebugRuntime();rt._lock=threading.RLock();rt._guard=lambda:nullcontext();rt._stream=None
    k=elementwise("prepared","add",TensorSpec((4,)))
    args=tuple(rt.allocate(16) for _ in range(3));rt._modules={k.digest:(None,123)}
    def validate(loaded,kernel,buffers):
        assert loaded==kernel.digest
        for b in buffers:rt.validate_buffer(b)
    rt._validate_launch=validate
    rt._check=lambda code,op:None
    seen=[]
    rt._cuLaunchKernel=lambda *a:seen.append(C.addressof(a[-2])) or 0
    seq=PreparedLaunches(rt,[(k.digest,k,args)])
    seq.replay();seq.replay()
    assert seen[0]==seen[1] and seq.replays==2
    rt.free(args[0])
    with pytest.raises(ValueError,match="freed"):seq.replay()
    seq.close()
    with pytest.raises(RuntimeError,match="closed"):seq.replay()


@pytest.mark.gpu
def test_native_nvidia_gpu_and_persistent_cache(tmp_path):
    try:rt=NvidiaIsaRuntime(cache_dir=tmp_path/"cache")
    except DriverError as exc:
        if os.environ.get("RUDA_REQUIRE_GPU")=="1":pytest.fail(str(exc))
        pytest.skip(str(exc))
    kernel=elementwise("ruda_native_add","add",TensorSpec((4,)))
    for iteration in range(2):
        if iteration:rt=NvidiaIsaRuntime(cache_dir=tmp_path/"cache")
        with rt:
            a,b=rt.allocate(16),rt.allocate(16)
            rt.write(a,struct.pack("<4f",1,2,3,4))
            rt.launch(rt.load(kernel),kernel,(a,a,b))
            assert struct.unpack("<4f",rt.read(b,16))==(2,4,6,8)
            if iteration:assert rt.native_cache.stats["hits"]==1


@pytest.mark.amd_gpu
def test_native_amd_gpu():
    target=os.environ.get("RUDA_AMD_TARGET")
    if target is None:
        if os.environ.get("RUDA_REQUIRE_AMD_GPU")=="1":pytest.fail("Set RUDA_AMD_TARGET to the actual gfx target")
        pytest.skip("AMD execution not requested; set RUDA_AMD_TARGET")
    from ruda_ptx.hip_isa import HipIsaRuntime,HipIsaError
    try:rt=HipIsaRuntime(target)
    except HipIsaError as exc:
        if os.environ.get("RUDA_REQUIRE_AMD_GPU")=="1":pytest.fail(str(exc))
        pytest.skip(str(exc))
    class M(torch.nn.Module):
        def forward(self,x,y):return torch.nn.functional.silu(x+y)*y
    x,y=torch.randn(1,1025),torch.randn(1,1025)
    plan=compile_exported(torch.export.export(M(),(x,y)))
    images=lower_plan_amdgpu(plan,AmdGpuCompiler(target))
    with rt,Executor(plan,NativePlanRuntime(rt,images)) as executor,torch.inference_mode():
        torch.testing.assert_close(executor(x,y),M()(x,y),atol=2e-5,rtol=2e-4)
