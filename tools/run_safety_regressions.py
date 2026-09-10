#!/usr/bin/env python3
"""Run scoped RUDA regression suites; preserve real results and per-command logs.

Requires Python 3.10+. No dependencies or Rust tools are installed automatically.
Missing tools are reported as BLOCKED, not as passed/skipped tests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def suites() -> dict[str, list[list[str]]]:
    ruda = ["cargo", "test", "--locked", "-p", "ruda", "--lib"]
    core = ["cargo", "test", "--locked", "-p", "ruda-core", "--lib", "--no-default-features"]
    cuda = ["cargo", "test", "--locked", "-p", "ruda-driver-cuda", "--lib"]
    miri_ruda = ["cargo", "+nightly", "miri", "test", "--locked", "-p", "ruda", "--lib"]
    miri_core = ["cargo", "+nightly", "miri", "test", "--locked", "-p", "ruda-core", "--lib", "--no-default-features"]
    return {
        "cpu": [
            ruda + ["runtime::storage::"],
            ruda + ["memory_management::memory_pool::handle::tests"],
            core + ["--features", "compilation-cache", "compilation_cache::safety_tests"],
            core + ["--features", "std,tensor-host-storage", "tensor::host::storage::"],
            ["cargo", "test", "--locked", "-p", "ruda", "--test", "runtime"],
            ["cargo", "check", "--locked", "-p", "ruda", "--lib", "--no-default-features",
             "--features", "runtime,runtime-storage-bytes"],
        ],
        "miri": [
            miri_ruda + ["runtime::storage::"],
            miri_ruda + ["memory_management::memory_pool::handle::tests"],
            miri_core + ["--features", "compilation-cache", "compilation_cache::safety_tests"],
            miri_core + ["--features", "std,tensor-host-storage", "tensor::host::storage::"],
        ],
        "cuda": [
            cuda + ["binding_safety_tests"],
            cuda + ["nvrtc_program::tests"],
            cuda + ["nvrtc_program_smoke_success_and_failure", "--", "--ignored", "--test-threads=1"],
        ],
        "cpu-driver": [
            ["cargo", "check", "--locked", "-p", "ruda-driver-cpu", "--lib"],
            ["cargo", "test", "--locked", "-p", "ruda-driver-cpu", "--lib", "memory::allocation::tests"],
        ],
    }


def stop_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()


def run(command: list[str], log: Path, timeout: float, env: dict[str, str]) -> dict:
    started = time.monotonic()
    result = {"command": command, "log": str(log), "status": "not_started"}
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with log.open("w", encoding="utf-8") as output:
        output.write("$ " + shlex.join(command) + "\n")
        output.flush()
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, **kwargs)
        except OSError as exc:
            result.update(status="blocked", error=str(exc))
        else:
            try:
                code = process.wait(timeout=timeout)
                result.update(status="passed" if code == 0 else "failed", returncode=code)
            except subprocess.TimeoutExpired:
                stop_tree(process)
                result.update(status="timeout", returncode=process.returncode)
            except KeyboardInterrupt:
                stop_tree(process)
                result.update(status="interrupted", returncode=process.returncode)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=suites(), default="cpu")
    parser.add_argument("--timeout", type=float, default=600, help="Per-command limit, not a predicted runtime")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--offline", action="store_true", help="Require already-cached Cargo dependencies")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    commands = suites()[args.suite]
    if args.offline:
        for command in commands:
            # A Cargo flag goes before the test filter/--, not after libtest flags.
            command.insert(command.index("--locked") + 1, "--offline")
    if args.dry_run:
        print(json.dumps({"status": "planned_only", "suite": args.suite, "commands": commands}, indent=2))
        return 0

    timestamp = dt.datetime.now(dt.timezone.utc)
    log_dir = args.log_dir or ROOT / "target" / "safety-regressions" / timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
    log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "results.json"
    if report_path.exists():
        parser.error("log directory already contains results.json; choose a new directory")
    report = {"suite": args.suite, "started_utc": timestamp.isoformat(), "status": "running",
              "planned_commands": commands, "results": [], "environment": {}}
    def save() -> None:
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(report_path)
    save()
    missing = [tool for tool in ["cargo", "rustc"] if shutil.which(tool) is None]
    if missing:
        report.update(status="blocked", reason="Missing tools: " + ", ".join(missing),
                      finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save()
        print(report["reason"] + "\nReport: " + str(report_path), file=sys.stderr)
        return 2
    for tool in ["cargo", "rustc"]:
        try:
            probe = subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=15, check=False)
            report["environment"][tool] = {"returncode": probe.returncode, "output": (probe.stdout + probe.stderr).strip()}
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["environment"][tool] = {"error": str(exc)}
    env = os.environ.copy()
    if args.suite == "miri":
        # These scoped tests use real temporary cache directories, but not GPUs.
        env["MIRIFLAGS"] = (env.get("MIRIFLAGS", "") + " -Zmiri-disable-isolation").strip()
        report["environment"]["MIRIFLAGS"] = env["MIRIFLAGS"]
    try:
        for index, command in enumerate(commands, start=1):
            print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
            result = run(command, log_dir / f"{index:02d}.log", args.timeout, env)
            report["results"].append(result)
            save()
            print(result["status"], flush=True)
            if result["status"] == "interrupted" or (result["status"] != "passed" and not args.keep_going):
                break
    finally:
        report["status"] = "passed" if len(report["results"]) == len(commands) and all(
            item["status"] == "passed" for item in report["results"]) else "incomplete_or_failed"
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
    print("Report: " + str(report_path))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
