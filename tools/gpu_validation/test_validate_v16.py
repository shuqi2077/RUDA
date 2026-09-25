"""Host checks of release-runner failure handling; never GPU acceptance."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import subprocess
import sys
import json
import pytest

SCRIPT = Path(__file__).with_name("validate_v16.py")
spec = importlib.util.spec_from_file_location("v16_validator", SCRIPT)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

@pytest.mark.parametrize("text", [
    "", "Compiling ruda-fft ... Finished", "test result: ok. 0 passed; 0 failed; 0 ignored;",
    "test result: ok. 16 passed; 0 failed; 1 ignored;",
    "test result: FAILED. 16 passed; 1 failed; 0 ignored;",
    "test result: ok. 16 passed; 0 failed; 0 ignored;\ntest result: FAILED. 0 passed; 1 failed; 0 ignored;",
])
def test_release_gate_rejects_incomplete_reports(text):
    with pytest.raises(ValueError): module.require_test_success(text,16)

def test_release_gate_accepts_complete_report():
    assert module.require_test_success("test result: ok. 16 passed; 0 failed; 0 ignored; 25 filtered out;",16)==16

def test_preflight_cannot_turn_missing_tools_into_gpu_pass(tmp_path):
    # Empty PATH is intentional. Invoke this test with the actual Python path.
    env={"PATH":"", "RUDA_PTX_VERSION":"8.0"}
    output=tmp_path/"report"
    process=subprocess.run([sys.executable,str(SCRIPT),"--preflight-only","--output",str(output)],env=env,capture_output=True,text=True)
    assert process.returncode==2
    report=json.loads((output/"result.json").read_text())
    assert not report["gpu_validated"] and not report["rust_compiled"]
    assert "missing cargo" in report["errors"] and "missing rustc" in report["errors"]
