#!/usr/bin/env python3
"""Actual production Rust -> CCE -> CANN -> device row validation; no skips."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from build_common_ir import ROOT, ROW_OPS, build, run
from validate_common_ir import preflight

SHAPES = ((0, 32), (1, 32), (3, 96), (7, 256), (33, 4096))
EXPECTED = {(op, r, w) for op in ROW_OPS for r, w in SHAPES}


def validate_device_log(text: str) -> None:
    found = re.findall(r"^RUDA_ASCEND_ROW_CASE op=(\w+) rows=(\d+) width=(\d+) passed=true$", text, re.M)
    cases = [(op, int(r), int(w)) for op, r, w in found]
    if len(cases) != len(EXPECTED) or set(cases) != EXPECTED:
        raise RuntimeError("missing, duplicate or unexpected real-device row cases")
    if len(re.findall(r"^RUDA_ASCEND_ROWS_DEVICE_OK cases=45 launches=72$", text, re.M)) != 1:
        raise RuntimeError("missing exact native execution marker")
    if re.search(r"\bpassed=false\b|\bSKIPPED\b|\bFAILED\b", text):
        raise RuntimeError("failed or skipped device work")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--toolkit", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.absolute()
    if out.exists():
        p.error("use a fresh result directory")
    out.mkdir(parents=True)
    toolkit = a.toolkit or (Path(os.environ["ASCEND_HOME_PATH"]) if "ASCEND_HOME_PATH" in os.environ else None)
    state = {"status": "failed", "rust_executed": False, "device_cases": 0,
             "device_launches": 0, "source_root": str(ROOT), "failures": preflight(toolkit)}
    try:
        if state["failures"]:
            raise RuntimeError("preflight failed: " + ", ".join(state["failures"]))
        cargo = shutil.which("cargo")
        for package, features, filt in (("ruda-compiler", "ascend,ptx", "ascend::"),
                                        ("ruda-driver-cann", "common-ir", "common_ir::")):
            text = run([cargo, "test", "--locked", "-p", package, "--no-default-features",
                        "--features", features, "--lib", filt, "--", "--nocapture"], out / f"{package}-host.log")
            m = re.search(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", text)
            if not m or int(m[1]) == 0 or int(m[2]) or int(m[3]):
                raise RuntimeError("production Rust tests missing/failed/ignored")
        state["rust_executed"] = True
        root = out / "artifacts"
        for op in ROW_OPS:
            for rows, width in SHAPES:
                build(root / f"{op}-{rows}-{width}", toolkit.resolve(), op, rows * width,
                      row_width=width)
        text = run([cargo, "run", "--locked", "--release", "-p", "ruda-driver-cann", "--features", "common-ir",
                    "--example", "common_rows_validate", "--", str(root)], out / "device.log")
        validate_device_log(text)
        state.update(status="passed", device_cases=len(EXPECTED), device_launches=72)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        state["error"] = str(error)
    (out / "summary.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
