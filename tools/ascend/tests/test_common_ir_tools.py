"""Build/validation controls: production Python tools, not a fake device backend."""
import importlib.util
import math
import sys
from pathlib import Path
import pytest

TOOLS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TOOLS))
import build_common_ir as build
import validate_common_ir as validation


def contract(n=17, name="ruda_cann_add"):
    return ("schema=ruda.ascend.common-map.v1\nsource_language=ruda-kernel-ir\n"
        "lowering=ascendc-vector\nsoc=Ascend950DT\narch=dav-c310\n"
        f"kernel_name={name}\nelements={n}\nblock_dim=32\ntile_elements=256\n"
        f"ub_bytes=4096\nbindings=3\nbinding_0=0,r,{n*4}\nbinding_1=1,r,{n*4}\nbinding_2=2,w,{n*4}\n")


def test_build_contract_exact():
    assert build.parse_contract(contract())["bindings"]=="3"


@pytest.mark.parametrize("old,new",[
    ("common-map.v1","common-map.v2"),("ruda-kernel-ir","template"),
    ("Ascend950DT","Ascend910B"),("dav-c310","dav-c220"),
    ("block_dim=32","block_dim=0"),("tile_elements=256","tile_elements=257"),
    ("ub_bytes=4096","ub_bytes=200000"),("binding_2=2,w,68","binding_2=0,w,68"),
    ("binding_2=2,w,68","binding_2=2,w,64"),("bindings=3","bindings=4"),
    ("kernel_name=ruda_cann_add","kernel_name=x();"),
    ("kernel_name=ruda_cann_add","kernel_name=_invalid"),
])
def test_bad_contracts_rejected(old,new):
    with pytest.raises((ValueError,KeyError)):
        build.parse_contract(contract().replace(old,new))


@pytest.mark.parametrize("suffix",["schema=x\n","unknown=x\n","badline\n","=missing\n"])
def test_duplicate_unknown_or_malformed(suffix):
    with pytest.raises(ValueError):build.parse_contract(contract()+suffix)


@pytest.mark.parametrize("args",[("foo",17,256,32),("add",-1,256,32),("add",2**32,256,32),("add",17,7,32),("add",17,4097,32),("add",17,256,0),("add",17,256,33)])
def test_bad_generation_options(args):
    with pytest.raises(ValueError):build.validate_args(*args)


def test_missing_rust_does_not_emit(monkeypatch,tmp_path):
    monkeypatch.setattr(build.shutil,"which",lambda name:None)
    out=tmp_path/'result'
    with pytest.raises(ValueError,match="production Rust"):build.emit(out,"silu",17)
    assert not out.exists()


def test_existing_output_not_overwritten(tmp_path):
    out=tmp_path/'out';out.mkdir();(out/'keep').write_text('unchanged')
    with pytest.raises(ValueError):build.emit(out,"add",17)
    assert (out/'keep').read_text()=='unchanged'


def test_sdk_required_before_compilation(tmp_path):
    with pytest.raises(ValueError,match="compiler/linker"):
        build.build(tmp_path/'out',tmp_path/'missing','add',17)
    assert not (tmp_path/'out').exists()


def test_preflight_fails_instead_of_skipping(monkeypatch):
    monkeypatch.setattr(validation.shutil,"which",lambda name:None)
    def fail(*args):raise OSError('absent')
    monkeypatch.setattr(validation.ctypes,"CDLL",fail)
    assert set(validation.preflight(None))=={'cargo','rustc','ASCEND_HOME_PATH/--toolkit','libascendcl.so','libopapi.so'}


def test_source_wiring_is_shared_ir_not_v38_matrix_stage():
    root=TOOLS.parents[1]
    compiler=(root/'ruda-compiler/src/ascend/mod.rs').read_text()
    assert 'impl Compiler for AscendCompiler' in compiler
    assert 'kernel: KernelDefinition' in compiler
    assert 'ruda-ascend-kernels' not in compiler
    programs=(root/'ruda-compiler/src/ascend/programs.rs').read_text()
    assert 'KernelDefinition' in programs and 'Arithmetic::' in programs
    assert 'AscendC::' not in programs and 'deep_gemm' not in programs
    runtime=(root/'ruda-driver-cann/src/tensor/common_ir/mod.rs').read_text()
    assert 'validate_ranges(&ranges)?' in runtime
    assert 'session.quiescent.set(false)' in runtime
    assert 'aclrtSynchronizeStream' in runtime


def test_build_never_publishes_fake_object():
    source=Path(build.__file__).read_text()
    assert 'run(compile_cmd' in source and 'run(link_cmd' in source
    assert source.index('run(link_cmd')<source.index('(generated / "kernel.ruda").write_text')
    assert '--features", "ascend"' in source
    assert 'from build_deepgemm import kernel_name' in source  # ELF parser only


def test_zero_lane_contract():
    assert build.parse_contract(contract(0))['elements']=='0'


# Independent schedule and arithmetic references. These do NOT execute Rust,
# generated CCE, or Ascend hardware and are labelled separately in the report.
@pytest.mark.parametrize("n",[0,1,7,8,255,256,257,1025,8195,32771])
@pytest.mark.parametrize("cores",[1,7,32])
def test_striped_tiles_cover_exact_domain(n,cores):
    tile=256;seen=set()
    for core in range(cores):
        for offset in range(core*tile,n,cores*tile):
            count=min(tile,n-offset);aligned=(count+7)//8*8
            assert 0<count<=aligned<=tile
            assert offset%8==0
            for i in range(offset,offset+count):
                assert i not in seen;seen.add(i)
    assert seen==set(range(n))


@pytest.mark.parametrize("x",[-8.,-3.,-0.1,0.,0.1,3.,8.])
def test_gated_gradient_finite_difference_reference(x):
    up=0.7;dy=-0.4;h=1e-5
    sigmoid=lambda z:1/(1+math.exp(-z))
    f=lambda z,u:dy*z*sigmoid(z)*u
    dx=dy*up*sigmoid(x)*(1+x*(1-sigmoid(x)))
    dup=dy*x*sigmoid(x)
    assert abs(dx-(f(x+h,up)-f(x-h,up))/(2*h))<1e-8
    assert abs(dup-(f(x,up+h)-f(x,up-h))/(2*h))<1e-8
