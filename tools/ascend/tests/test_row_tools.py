"""Production build/validation controls. No CANN or Rust execution is mocked as success."""
from pathlib import Path
import sys
import pytest
TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import build_common_ir as build
import validate_common_rows as validation


def contract(rows=3, width=64):
    n=rows*width
    return ("schema=ruda.ascend.common-row.v1\nsource_language=ruda-kernel-ir\n"
        "lowering=ascendc-vector\nsoc=Ascend950DT\narch=dav-c310\n"
        f"kernel_name=ruda_cann_rms_norm\nelements={n}\nblock_dim=32\ntile_elements={width}\n"
        f"row_width={width}\nlogical_plane=32\nub_bytes=8192\nbindings=4\n"
        f"binding_0=0,r,{n*4}\nbinding_1=1,r,{width*4}\nbinding_2=2,w,{n*4}\nbinding_3=3,w,{rows*4}\n")


@pytest.mark.parametrize("rows,width", [(0,32),(1,32),(3,96),(7,256),(33,4096)])
def test_row_binding_domain_contract(rows,width):
    c=build.parse_contract(contract(rows,width))
    assert c["binding_3"].endswith(f",{rows*4}")
    assert c["binding_1"].endswith(f",{width*4}")


@pytest.mark.parametrize("old,new", [
    ("logical_plane=32","logical_plane=64"),("row_width=64","row_width=65"),
    ("row_width=64","row_width=0"),("row_width=64","row_width=8192"),
    ("elements=192","elements=193"),("tile_elements=64","tile_elements=32"),
    ("binding_3=3,w,12","binding_3=3,w,16"),("binding_1=1,r,256","binding_1=1,r,128"),
    ("common-row.v1","common-row.v2"),("logical_plane=32\n",""),
    ("source_language=ruda-kernel-ir","source_language=cpp-template"),
    ("soc=Ascend950DT","soc=Ascend910B"),
])
def test_invalid_row_contracts_fail(old,new):
    with pytest.raises(ValueError):build.parse_contract(contract().replace(old,new))


@pytest.mark.parametrize("op",build.ROW_OPS)
def test_row_ops_require_explicit_domain(op):
    build.validate_args(op,192,256,32,64)
    with pytest.raises(ValueError):build.validate_args(op,192,256,32)


@pytest.mark.parametrize("width",[0,1,31,33,4097,8192])
def test_unsupported_width_not_silently_padded(width):
    with pytest.raises(ValueError):build.validate_args("softmax",192,256,32,width)


@pytest.mark.parametrize("epsilon",[0.,-1.,float('inf'),float('nan'),1e99,1e-99])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):build.validate_args("rms_norm",192,256,32,64,epsilon)


def test_missing_rust_never_generates_cce_or_manifest(monkeypatch,tmp_path):
    monkeypatch.setattr(build.shutil,"which",lambda _:None)
    out=tmp_path/'row'
    with pytest.raises(ValueError,match='production Rust'):build.emit(out,"softmax",192,row_width=64)
    assert not out.exists()


def test_map_op_does_not_gain_row_semantics():
    with pytest.raises(ValueError):build.validate_args("silu",192,256,32,64)


def complete_log():
    return '\n'.join([f'RUDA_ASCEND_ROW_CASE op={op} rows={r} width={w} passed=true'
        for op,r,w in sorted(validation.EXPECTED)]+['RUDA_ASCEND_ROWS_DEVICE_OK cases=45 launches=72'])


def test_strict_device_marker_parsing():validation.validate_device_log(complete_log())


@pytest.mark.parametrize("edit",[
    lambda t:t.replace('cases=45','cases=44'),
    lambda t:t.replace('launches=72','launches=0'),
    lambda t:'\n'.join(t.splitlines()[1:]),
    lambda t:t+'\n'+t.splitlines()[0],
    lambda t:t+'\nSKIPPED',
    lambda t:t.replace('passed=true','passed=false',1),
])
def test_incomplete_device_logs_are_not_success(edit):
    with pytest.raises(RuntimeError):validation.validate_device_log(edit(complete_log()))


def test_public_ir_is_reused_not_another_algorithm_wrapper():
    src=TOOLS.parents[1]
    algorithms=(src/'ruda-compiler/src/ascend/row_programs.rs').read_text()
    lower=(src/'ruda-compiler/src/ascend/rows.rs').read_text()
    emit=(src/'ruda-compiler/src/ascend/row_emit.rs').read_text()
    assert 'KernelDefinition' in algorithms and 'Plane::Sum' in algorithms
    assert 'AscendC::' not in algorithms and '#include' not in algorithms and 'aclnn' not in algorithms
    assert 'Operation::Plane' in lower and 'ruda_dim.x != LANES' in lower
    assert 'AscendC::ReduceSum' in emit and 'AscendC::ReduceMax' in emit
    assert 'HardEvent::V_S' in emit and 'HardEvent::S_V' in emit and 'pipe.FetchEventID(event)' in emit
    assert 'AscendC::RmsNorm' not in emit and 'AscendC::SoftMax' not in emit


def test_runtime_allocates_each_output_from_its_binding():
    text=(TOOLS.parents[1]/'ruda-driver-cann/src/tensor/common_ir/mod.rs').read_text()
    assert '[(b.bytes / 4) as i64]' in text
    assert 'validate_ranges(&ranges)?' in text
    assert 't.layout.byte_len() as u64!=b.bytes' in text


def test_no_tool_replaces_production_rust_emitter():
    text=(TOOLS/'build_common_ir.py').read_text()
    assert 'row_width is not None' in text and '"--row-width", str(row_width)' in text
    assert 'missing cargo/rustc' in text
    assert text.index('run(link_cmd') < text.index('(generated / "kernel.ruda").write_text')
