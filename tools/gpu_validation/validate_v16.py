#!/usr/bin/env python3
"""Fail-closed validation of the real Rust/PTX FFT path. No CPU fallback.

Running this script is not equivalent to GPU acceptance unless result.json
says gpu_validated=true. The CPU-only mathematical tests are deliberately a
separate command. Does not change any async default or public ABI.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def require_test_success(text: str, minimum: int) -> int:
    reports = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;", text)
    if not reports:
        raise ValueError("no Rust test summary: build-only/zero execution is not acceptance")
    if any(state != "ok" or int(failed) or int(ignored) for state, _, failed, ignored in reports):
        raise ValueError("failed or ignored required Rust cases")
    count = sum(int(passed) for _, passed, _, _ in reports)
    if count < minimum:
        raise ValueError(f"only {count} executed cases; {minimum} required")
    return count


def probe_driver() -> dict[str, Any]:
    driver = ctypes.CDLL("libcuda.so.1" if os.name != "nt" else "nvcuda.dll")
    for name, arguments in [
        ("cuInit", [ctypes.c_uint]),
        ("cuDriverGetVersion", [ctypes.POINTER(ctypes.c_int)]),
        ("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)]),
    ]:
        function = getattr(driver, name)
        function.argtypes = arguments
        function.restype = ctypes.c_int
    status = driver.cuInit(0)
    if status != 0:
        raise RuntimeError(f"cuInit failed ({status})")
    count, version = ctypes.c_int(), ctypes.c_int()
    if driver.cuDeviceGetCount(ctypes.byref(count)) or count.value < 1:
        raise RuntimeError("no usable NVIDIA device")
    if driver.cuDriverGetVersion(ctypes.byref(version)):
        raise RuntimeError("cannot identify NVIDIA driver version")
    name = ctypes.create_string_buffer(256)
    driver.cuDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    driver.cuDeviceGetName.restype = ctypes.c_int
    if driver.cuDeviceGetName(name, len(name), 0):
        raise RuntimeError("cannot identify device zero")
    return {"device_count": count.value, "driver_api_version": version.value,
            "device_0": name.value.decode(errors="replace")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("v16-hardware-validation"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sanitizer", choices=["memcheck", "racecheck", "synccheck", "initcheck"])
    parser.add_argument("--timeout", type=int, default=1200, help="per-command timeout in seconds")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"version": "v16", "source_baseline": "4cf754c799cc535927b68f2202c307f71b848fb7",
        "gpu_validated": False, "rust_compiled": False, "default_async_changed": False,
        "platform": platform.platform(), "commands": [], "errors": []}
    def save() -> None:
        (output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name in ["cargo", "rustc"]:
        if not shutil.which(name):
            report["errors"].append(f"missing {name}")
    if not re.fullmatch(r"\d+\.\d+", os.environ.get("RUDA_PTX_VERSION", "")):
        report["errors"].append("set RUDA_PTX_VERSION=major.minor for the actual driver")
    if args.sanitizer and not shutil.which("compute-sanitizer"):
        report["errors"].append("requested compute-sanitizer was not found")
    try:
        report["driver"] = probe_driver()
    except (OSError, AttributeError, RuntimeError) as error:
        report["errors"].append(str(error))
    save()
    if report["errors"]:
        print("Preflight failed: " + "; ".join(report["errors"]), file=sys.stderr)
        return 2
    if args.preflight_only:
        print("Preflight passed; no kernels compiled or validated.")
        return 0
    env = dict(os.environ, RUDA_CUDA_COMPILER="ptx", RUDA_FFT_REQUIRE_CUDA="1")
    def run(command: list[str]) -> str:
        log = output / f'{len(report["commands"]):02d}.log'
        record: dict[str, Any] = {"command": command, "log": log.name}
        report["commands"].append(record); save()
        with log.open("w") as stream:
            try:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
                record["returncode"] = process.returncode
            except subprocess.TimeoutExpired:
                record["timed_out"] = True; save()
                raise RuntimeError(f"required command timed out: {log.name}")
        save()
        if process.returncode:
            raise RuntimeError(f"required command failed: {log.name}")
        return log.read_text(errors="replace")
    try:
        report["rustc_version"] = run(["rustc", "--version"])
        unit = run(["cargo", "test", "--release", "--locked", "-p", "ruda-driver-cuda",
                    "--no-default-features", "--features", "std,direct-ptx",
                    "module_config::tests", "--", "--test-threads=1"])
        report["driver_host_tests"] = require_test_success(unit, 2)
        fft_command = ["cargo", "test", "--release", "--locked", "-p", "ruda-fft",
            "--no-default-features", "--features", "std,tensor,cpu-reference,ruda-test-runtime/cuda",
            "--test", "lib", "exact_fft_", "--", "--test-threads=1", "--nocapture",
            "--skip", "exact_fft_plan_reuse_benchmark"]
        text = run(fft_command)
        report["gpu_tests"] = require_test_success(text, 16)
        if not re.search(r"EXACT_FFT_RUNTIME,\S*CudaRuntime", text):
            raise RuntimeError("missing explicit CUDA-runtime test identity")
        report["rust_compiled"] = True
        if args.sanitizer:
            text = run(["compute-sanitizer", "--tool", args.sanitizer, "--target-processes", "all",
                        "--error-exitcode", "86", *fft_command])
            require_test_success(text, 16)
            if not re.search(r"ERROR SUMMARY: 0 errors|RACECHECK SUMMARY: 0 hazards", text):
                raise RuntimeError("sanitizer did not report a clean instrumented run")
            report["sanitizer"] = args.sanitizer
        if args.benchmark:
            benchmark_command = fft_command[:]
            selector = benchmark_command.index("exact_fft_")
            benchmark_command[selector] = "exact_fft_plan_reuse_benchmark"
            benchmark_command = benchmark_command[:benchmark_command.index("--skip")] + ["--ignored"]
            text = run(benchmark_command)
            require_test_success(text, 1)
            report["benchmark_lines"] = [line for line in text.splitlines() if line.startswith("EXACT_FFT_TIMING,")]
            if len(report["benchmark_lines"]) != 6:
                raise RuntimeError("incomplete exact-FFT benchmark cases")
        report["gpu_validated"] = True
        save()
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        report["errors"].append(str(error)); save()
        print(str(error), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
