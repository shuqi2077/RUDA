#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run explicit fused AdamW checks/benchmarks; missing tools are BLOCKED, never PASS.

No installations, implicit CPU fallback, or whole-workspace --all-features runs.
Reuses the safety patch's process-tree cleanup and per-command logging.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from run_safety_regressions import run

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["oracle", "reference", "host", "build", "cuda", "bench"], default="reference")
    parser.add_argument("--compiler", choices=["nvrtc", "ptx", "both"], default="both")
    parser.add_argument("--timeout", type=float, default=600, help="Per-command maximum; not an estimated completion time")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--elements", type=int, default=65536)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="f32")
    parser.add_argument("--amsgrad", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0 or not 0 < args.elements <= (2**32 - 1)//4 or args.iterations <= 0 or args.samples < 3:
        parser.error("require positive timeout/iterations, a valid element count, and samples >= 3")
    now = dt.datetime.now(dt.timezone.utc)
    logs = (args.log_dir or ROOT / "target" / "fused-adamw" / now.strftime("%Y%m%dT%H%M%S.%fZ")).resolve()
    flags = ["--locked"] + (["--offline"] if args.offline else [])
    env = os.environ.copy()
    plans: list[tuple[list[str], dict[str, str]]] = []
    tools = []
    if args.suite == "oracle":
        tools = [sys.executable]
        plans = [([sys.executable, "tools/fused_adamw/oracle.py", "--report", str(logs / "oracle.json")], env)]
    elif args.suite == "reference":
        tools = ["rustc"]
        executable = logs / ("reference-tests.exe" if os.name == "nt" else "reference-tests")
        plans = [(["rustc", "--edition", "2024", "--test", "tools/fused_adamw/reference_tests.rs", "-o", str(executable)], env),
                 ([str(executable), "--test-threads=1"], env)]
    elif args.suite == "host":
        tools = ["cargo", "rustc"]
        plans = [(["cargo", "test", *flags, "-p", "ruda-optim", "--lib", "--features", "fused-adamw", "fused_adamw::", "--", "--test-threads=1"], env)]
    elif args.suite == "build":
        tools = ["cargo", "rustc"]
        plans = [(["cargo", "check", *flags, "-p", "ruda-optim", "--lib", "--features", "fused-adamw-device"], env)]
    else:
        tools = ["cargo", "rustc"]
        for backend in (["nvrtc", "ptx"] if args.compiler == "both" else [args.compiler]):
            backend_env = {**env, "RUDA_CUDA_COMPILER": backend}
            if args.suite == "cuda":
                command = ["cargo", "test", *flags, "--release", "-p", "ruda-optim", "--features", "fused-adamw-cuda", "--test", "fused-adamw-cuda", "--", "--test-threads=1"]
            else:
                command = ["cargo", "run", *flags, "--release", "-p", "ruda-optim", "--features", "fused-adamw-cuda", "--example", "fused-adamw-bench", "--",
                    "--elements", str(args.elements), "--iterations", str(args.iterations), "--samples", str(args.samples), "--dtype", args.dtype,
                    "--out", str(logs / f"benchmark-{backend}.json")]
                if args.amsgrad: command.append("--amsgrad")
            plans.append((command, backend_env))
    public_plans = [{"command": cmd, "compiler": e.get("RUDA_CUDA_COMPILER")} for cmd, e in plans]
    if args.dry_run:
        print(json.dumps({"status": "planned_only", "suite": args.suite, "plans": public_plans}, indent=2)); return 0
    logs.mkdir(parents=True, exist_ok=True)
    report_path = logs / "results.json"
    if report_path.exists(): parser.error("results.json already exists; choose another log directory")
    files = sorted((ROOT / "ruda-optim/src/fused_adamw").glob("*.rs")) + [ROOT / "Cargo.lock", ROOT / "ruda-optim/Cargo.toml"]
    report = {"suite": args.suite, "started_utc": now.isoformat(), "status": "running", "plans": public_plans, "results": [],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "performance_measured": False, "environment": {"RUDA_PTX_VERSION": env.get("RUDA_PTX_VERSION")}}
    def save():
        temp = report_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8"); temp.replace(report_path)
    save()
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        report.update(status="blocked", reason="Missing tools: " + ", ".join(missing), finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save(); print(json.dumps(report, indent=2)); return 2
    for tool in [t for t in tools if t in {"cargo", "rustc"}]:
        try:
            result = subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=15, check=False)
            report["environment"][tool] = (result.stdout + result.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["environment"][tool] = str(exc)
    for i, (command, command_env) in enumerate(plans, 1):
        print(f"[{i}/{len(plans)}] {shlex.join(command)}", flush=True)
        result = run(command, logs / f"{i:02d}.log", args.timeout, command_env)
        result["compiler"] = command_env.get("RUDA_CUDA_COMPILER")
        if args.suite == "oracle" and result.get("returncode") == 2: result["status"] = "blocked"
        report["results"].append(result); save()
        if result["status"] != "passed": break
    passed = len(report["results"]) == len(plans) and all(r["status"] == "passed" for r in report["results"])
    status = "passed" if passed else ("blocked" if report["results"][-1]["status"] == "blocked" else "incomplete_or_failed")
    report.update(status=status, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(), performance_measured=passed and args.suite == "bench")
    save(); print(f"{status}: {report_path}")
    return 0 if passed else (2 if status == "blocked" else 1)


if __name__ == "__main__":
    sys.exit(main())
