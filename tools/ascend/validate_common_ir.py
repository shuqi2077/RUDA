#!/usr/bin/env python3
"""Strict real-Rust/common-IR/CANN validation. Missing tools are failure, not skip."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from build_common_ir import ROOT, OPS, build, run

SIZES = (1, 7, 256, 257, 1025)


def preflight(toolkit: Path | None) -> list[str]:
    missing = [x for x in ("cargo", "rustc") if shutil.which(x) is None]
    if toolkit is None:
        missing.append("ASCEND_HOME_PATH/--toolkit")
    else:
        for name in ("bisheng", "ld.lld"):
            if not (toolkit / "bin" / name).is_file():
                missing.append(f"CANN {name}")
    # Do not create a device context here; the isolated Rust runner owns it.
    for env, default in (("RUDA_CANN_LIBRARY", "libascendcl.so"), ("RUDA_CANN_OPAPI", "libopapi.so")):
        try:
            ctypes.CDLL(os.environ.get(env, default))
        except OSError:
            missing.append(default)
    return missing


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--toolkit", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.absolute()
    if out.exists():
        p.error("use a fresh results directory")
    out.mkdir(parents=True)
    toolkit = a.toolkit or (Path(os.environ["ASCEND_HOME_PATH"]) if "ASCEND_HOME_PATH" in os.environ else None)
    state = {"status": "failed", "rust_executed": False, "device_cases": 0,
             "source_root": str(ROOT), "failures": preflight(toolkit)}
    try:
        if state["failures"]:
            raise RuntimeError("preflight failed: " + ", ".join(state["failures"]))
        cargo = shutil.which("cargo")
        for package, features, filt in (("ruda-compiler", "ascend,ptx", "ascend::"),
                                        ("ruda-driver-cann", "common-ir", "common_ir::")):
            text = run([cargo, "test", "--locked", "-p", package, "--no-default-features",
                        "--features", features, "--lib", filt, "--", "--nocapture"], out / f"{package}-host.log")
            if "test result: ok." not in text or "0 passed;" in text:
                raise RuntimeError("Rust host tests did not run")
        state["rust_executed"] = True
        root = out / "artifacts"
        for op in OPS:
            for n in SIZES:
                build(root / f"{op}-{n}", toolkit.resolve(), op, n)
        text = run([cargo, "run", "--locked", "--release", "-p", "ruda-driver-cann",
                    "--features", "common-ir", "--example", "common_ir_validate", "--", str(root)], out / "device.log")
        expected = f"RUDA_ASCEND_COMMON_IR_DEVICE_OK cases={len(OPS)*len(SIZES)} launches={2*len(OPS)*len(SIZES)}"
        if expected not in text:
            raise RuntimeError("missing exact real-device test marker")
        state.update(status="passed", device_cases=len(OPS)*len(SIZES))
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
        state["error"] = str(e)
    (out / "summary.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
