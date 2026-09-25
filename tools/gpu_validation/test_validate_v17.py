"""Acceptance log parser/failure-path unit tests; no GPU is used."""
from pathlib import Path
import importlib.util
import pytest
spec=importlib.util.spec_from_file_location("graph_acceptance",Path(__file__).with_name("validate_v17.py"))
v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)

def successful():
    names="\n".join("test "+name+" ... ok" for name in sorted(v.REQUIRED))
    return names+"\nRUDA_GRAPH_RUNTIME,CudaRuntime,direct-ptx\ntest result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out;\n"

def test_accepts_complete_real_run_identity(): assert v.require_graph_success(successful())==8

def test_accepts_nocapture_output_between_name_and_status():
    assert v.require_graph_success(successful().replace(" ... ok", " ... RUDA_GRAPH_RUNTIME,CudaRuntime,direct-ptx\nok"))==8

@pytest.mark.parametrize("bad",["", "Finished release target", "test result: ok. 0 passed; 0 failed; 0 ignored;",
    "test result: ok. 0 passed; 0 failed; 8 ignored;"])
def test_no_execution_never_passes(bad):
    with pytest.raises(ValueError): v.require_graph_success(bad)

@pytest.mark.parametrize("a,b",[("8 passed","7 passed"),("0 failed","1 failed"),("0 ignored","1 ignored"),
    ("test result: ok", "test result: FAILED"),("CudaRuntime,direct-ptx","CpuRuntime,reference")])
def test_incomplete_or_wrong_runtime_fails(a,b):
    with pytest.raises(ValueError): v.require_graph_success(successful().replace(a,b))

def test_count_without_required_case_names_fails():
    with pytest.raises(ValueError): v.require_graph_success(successful().replace(next(iter(v.REQUIRED)),"foreign_case"))
