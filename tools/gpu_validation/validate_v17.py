#!/usr/bin/env python3
"""Strict native Rust/PTX graph acceptance. A host-only test is never GPU acceptance."""
from __future__ import annotations
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("v17_probe_base", Path(__file__).with_name("validate_v16.py"))
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
REQUIRED = {
    "graph_replay_two_nodes_and_tail_guards", "graph_replay_retains_input_and_intermediate_handles",
    "graph_replay_arguments_survive_later_host_scratch_reuse", "graph_replay_in_place_updates_and_queue_ordering",
    "graph_replay_close_is_idempotent_and_replay_after_close_fails", "graph_replay_empty_and_zero_grids_are_rejected",
    "graph_replay_dynamic_grid_is_not_read_back", "graph_replay_wrong_queue_rejected_before_graph_build",
}


def require_graph_success(text: str) -> int:
    count = BASE.require_test_success(text, len(REQUIRED))
    # With --nocapture Rust prints application output between the test label and
    # trailing 'ok'. Count + exact name coverage + runtime identity are required.
    names = set(re.findall(r"test (graph_replay_\w+)\s+\.\.\.", text))
    missing = REQUIRED - names
    if missing:
        raise ValueError("missing required native graph cases: " + ", ".join(sorted(missing)))
    if "RUDA_GRAPH_RUNTIME,CudaRuntime,direct-ptx" not in text:
        raise ValueError("no explicit native CUDA/direct-PTX runtime identity")
    return count


def probe_graph_symbols() -> None:
    driver = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    for name in ["cuGraphCreate", "cuGraphAddKernelNode_v2", "cuGraphInstantiateWithFlags",
                 "cuGraphUpload", "cuGraphLaunch", "cuGraphExecDestroy", "cuGraphDestroy"]:
        getattr(driver, name)  # No raw graph is created by the preflight.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("v17-hardware-validation"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sanitizer", choices=["memcheck", "racecheck", "synccheck", "initcheck"])
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    report = {"version": "v17", "gpu_validated": False, "rust_compiled": False,
              "default_async_changed": False, "commands": [], "errors": []}
    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ["cargo", "rustc"]:
        if not shutil.which(name): report["errors"].append("missing " + name)
    if not re.fullmatch(r"\d+\.\d+", os.environ.get("RUDA_PTX_VERSION", "")):
        report["errors"].append("set RUDA_PTX_VERSION=major.minor for this device/driver")
    if args.sanitizer and not shutil.which("compute-sanitizer"):
        report["errors"].append("requested Compute Sanitizer is missing")
    try:
        report["driver"] = BASE.probe_driver()
        if report["driver"]["driver_api_version"] < 12000:
            raise RuntimeError("this acceptance profile requires NVIDIA Driver API >= 12.0")
        probe_graph_symbols()
    except (OSError, RuntimeError, AttributeError) as error:
        report["errors"].append(str(error))
    save()
    if report["errors"]:
        print("Preflight failed: " + "; ".join(report["errors"]), file=sys.stderr)
        return 2
    if args.preflight_only:
        print("Preflight passed; no Rust or GPU execution has been validated.")
        return 0
    env = dict(os.environ, RUDA_CUDA_COMPILER="ptx")
    def run(command):
        log = output / f'{len(report["commands"]):02d}.log'
        record = {"command": command, "log": log.name}
        report["commands"].append(record); save()
        with log.open("w") as stream:
            try:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
                record["returncode"] = process.returncode
            except subprocess.TimeoutExpired:
                record["timed_out"] = True; save()
                raise RuntimeError("required command timed out: " + log.name)
        save()
        if process.returncode: raise RuntimeError("required command failed: " + log.name)
        return log.read_text(errors="replace")
    try:
        report["rustc_version"] = run(["rustc", "--version"])
        common = ["cargo", "test", "--release", "--locked", "-p", "ruda-driver-cuda",
                  "--no-default-features", "--features", "std,direct-ptx"]
        for selector in ["execution::context::launch::tests", "graph::tests"]:
            text = run([*common, "--lib", selector, "--", "--test-threads=1"])
            BASE.require_test_success(text, 2)
        command = [*common, "--test", "graph-replay", "--", "--test-threads=1", "--nocapture",
                   "--skip", "graph_replay_benchmark"]
        text = run(command)
        report["gpu_tests"] = require_graph_success(text)
        report["rust_compiled"] = True
        if args.sanitizer:
            text = run(["compute-sanitizer", "--tool", args.sanitizer, "--target-processes", "all",
                        "--error-exitcode", "86", *command])
            require_graph_success(text)
            if not re.search(r"ERROR SUMMARY: 0 errors|RACECHECK SUMMARY: 0 hazards", text):
                raise RuntimeError("no clean instrumented sanitizer summary")
        if args.benchmark:
            text = run([*common, "--test", "graph-replay", "graph_replay_benchmark", "--",
                        "--ignored", "--exact", "--test-threads=1", "--nocapture"])
            BASE.require_test_success(text, 1)
            lines = [line for line in text.splitlines() if line.startswith("RUDA_GRAPH_TIMING,")]
            if len(lines) != 3: raise RuntimeError("incomplete graph timing cases")
            report["benchmark_lines"] = lines
        report["gpu_validated"] = True; save(); return 0
    except (OSError, ValueError, RuntimeError) as error:
        report["errors"].append(str(error)); save()
        print(str(error), file=sys.stderr); return 1

if __name__ == "__main__":
    raise SystemExit(main())
