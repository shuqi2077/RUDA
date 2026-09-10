#!/usr/bin/env python3
"""Run explicit validation tiers. Missing tools are NOT_RUN, never PASS.

Run from any directory. GPU tiers execute real, ignored backend tests and must
only be selected on machines with the matching compiler, driver and hardware.
No package installation, network setup, device reset or model download is done.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("reference", "core", "host", "gpu-nvidia", "gpu-amd", "feature-compile")


def execute(name: str, command: list[str], output: Path, timeout: int) -> dict:
    log = output / f"{name}.log"
    record = {"name": name, "command": command, "log": log.name}
    if shutil.which(command[0]) is None:
        record.update(status="NOT_RUN", reason=f"executable unavailable: {command[0]}")
        log.write_text(record["reason"] + "\n", encoding="utf-8")
        return record
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + shlex.join(command) + "\n")
        stream.flush()
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"))
            try:
                code = process.wait(timeout=timeout)
                record.update(status="PASS" if code == 0 else "FAIL", returncode=code)
            except subprocess.TimeoutExpired:
                # These are isolated test processes, not serving workers.
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                record.update(status="TIMEOUT", reason=f"exceeded {timeout} seconds")
        except OSError as error:
            record.update(status="FAIL", reason=str(error))
    record["elapsed_seconds"] = round(time.monotonic() - start, 3)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=TIERS, action="append", help="repeatable; default: reference and core")
    parser.add_argument("--output", type=Path, default=ROOT / "ruLLM/validation/runtime-v2")
    parser.add_argument("--timeout", type=int, default=1800, help="per-command timeout in seconds")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    suites = list(dict.fromkeys(args.suite or ["reference", "core"]))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    def run(name: str, command: list[str]) -> dict:
        result = execute(name, command, output, args.timeout)
        records.append(result)
        print(f"{name}: {result['status']}", flush=True)
        return result

    if "reference" in suites:
        run("v1-python-reference", [sys.executable, "ruLLM/tools/verify_reference.py"])
        run("v2-python-reference", [sys.executable, "ruLLM/tools/verify_runtime_reference.py"])
    if "core" in suites:
        with tempfile.TemporaryDirectory(prefix="rullm-core-") as temporary:
            binary = str(Path(temporary) / ("runtime-tests.exe" if os.name == "nt" else "runtime-tests"))
            result = run("rust-core-compile", ["rustc", "--edition=2024", "--test", "ruLLM/src/runtime/mod.rs", "-o", binary])
            if result["status"] == "PASS":
                run("rust-core-tests", [binary, "--test-threads=2"])
            else:
                records.append({"name": "rust-core-tests", "status": "NOT_RUN", "reason": "core compilation did not pass"})
    if "host" in suites:
        run("cargo-host-tests", ["cargo", "test", "--locked", "-p", "ruda-llm", "--lib", "--test", "generation_control", "--test", "api_compatibility"])
        run("cargo-host-examples", ["cargo", "check", "--locked", "-p", "ruda-llm", "--example", "sampling_bench", "--example", "runtime_replicas"])
    if "feature-compile" in suites:
        for feature in ("nvidia", "amd", "nvidia,amd"):
            run("features-" + feature.replace(",", "-"), ["cargo", "check", "--locked", "-p", "ruda-llm", "--features", feature, "--lib", "--examples"])
    for suite, feature in (("gpu-nvidia", "nvidia"), ("gpu-amd", "amd")):
        if suite in suites:
            run(suite, ["cargo", "test", "--locked", "-p", "ruda-llm", "--features", feature,
                       "--lib", "backend_tests", "--", "--ignored", "--test-threads=1"])
    report = {
        "schema": 1, "utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "suites_requested": suites, "checks": records,
        "all_requested_passed": all(r["status"] == "PASS" for r in records),
        "note": "Python checks are independent references and source inspections, not Rust compilation or GPU execution. Unrequested tiers have not been tested by this run.",
    }
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report: {output / 'validation.json'}")
    if any(r["status"] in ("FAIL", "TIMEOUT") for r in records):
        return 1
    return 0 if report["all_requested_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
