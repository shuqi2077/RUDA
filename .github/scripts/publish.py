"""Publish workspace versions in dependency order; the registry is the resume point."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import tomllib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
USER_AGENT = "RUDA-publisher (https://github.com/shuqi2077/RUDA)"
BATCH_SIZE = 5


def workspace():
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
        cwd=ROOT, text=True, encoding="utf-8",
    ))
    members = set(metadata["workspace_members"])
    packages = {p["name"]: p for p in metadata["packages"] if p["id"] in members}
    manifests = {Path(p["manifest_path"]).parent.resolve(): name
                 for name, p in packages.items()}
    root_manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    inherited = root_manifest.get("workspace", {}).get("dependencies", {})
    graph = {}
    for name, package in packages.items():
        if package["publish"] is not None and "crates-io" not in package["publish"]:
            continue
        manifest = Path(package["manifest_path"])
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        graph[name] = set()
        for scope in [data, *data.get("target", {}).values()]:
            for kind in ("dependencies", "build-dependencies", "dev-dependencies"):
                for alias, dependency in scope.get(kind, {}).items():
                    if not isinstance(dependency, dict):
                        continue
                    base = manifest.parent
                    if dependency.get("workspace"):
                        dependency = inherited[alias]
                        base = ROOT
                    if not isinstance(dependency, dict) or "path" not in dependency:
                        continue
                    if kind == "dev-dependencies" and "version" not in dependency:
                        continue
                    if "version" not in dependency:
                        raise RuntimeError(f"{name}: {alias} has no registry version")
                    target = (base / dependency["path"]).resolve()
                    if target not in manifests:
                        raise RuntimeError(f"{name}: unpublished external path dependency {alias}")
                    graph[name].add(manifests[target])
    return packages, graph


def ordered(graph):
    pending = {name: set(deps) for name, deps in graph.items()}
    result = []
    while pending:
        ready = sorted(name for name, deps in pending.items() if not deps)
        if not ready:
            raise RuntimeError(f"Publication dependency cycle or non-publishable dependency: {pending}")
        for name in ready:
            result.append(name)
            del pending[name]
        for deps in pending.values():
            deps.difference_update(ready)
    return result


def index_path(name):
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def registry(name):
    request = Request("https://index.crates.io/" + index_path(name), headers={
        "User-Agent": USER_AGENT, "Cache-Control": "no-cache",
    })
    for attempt in range(3):
        try:
            with urlopen(request, timeout=30) as response:
                return {entry["vers"]: entry for line in response.read().splitlines()
                        if (entry := json.loads(line))}
        except HTTPError as error:
            if error.code == 404:
                return {}
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            delay = error.headers.get("Retry-After", "60")
            time.sleep(int(delay) if delay.isdigit() else 60)
    raise RuntimeError(f"Cannot read registry entry for {name}")


def missing(packages, order, lookup=registry):
    return [name for name in order if packages[name]["version"] not in lookup(name)]


def matrix(count):
    batches = max(1, math.ceil(count / BATCH_SIZE))
    if batches > 256:
        raise RuntimeError("Publication exceeds GitHub's matrix limit")
    return {"batch": list(range(batches))}


def retry_time(output):
    if not re.search(r"429|rate limit|too many requests", output, re.IGNORECASE):
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00| UTC)", output)
    if not match:
        http_date = re.search(r"[A-Za-z]{3}, \d{1,2} [A-Za-z]{3} \d{4} \d{2}:\d{2}:\d{2} GMT", output)
        if http_date:
            return parsedate_to_datetime(http_date[0]).timestamp()
        raise RuntimeError("Registry rate limit did not include a retry time; rerun later")
    return datetime.fromisoformat(match[0].replace(" UTC", "+00:00").replace("Z", "+00:00")).timestamp()


def confirm(name, version, entry):
    directory = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target"))) / "package"
    filename = f"{name}-{version}.crate"
    for artifact in (directory / filename, directory / "tmp-crate" / filename):
        if artifact.is_file():
            checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if checksum == entry["cksum"]:
                return
    raise RuntimeError(f"{name}@{version}: registry checksum does not match the local archive")


def publish_one(name, version):
    for attempt in range(4):
        if version in registry(name):
            print(f"SKIP {name}@{version}: already published", flush=True)
            return
        print(f"PUBLISH {name}@{version} (Cargo archive verification enabled)", flush=True)
        process = subprocess.Popen(
            ["cargo", "publish", "--locked", "--registry", "crates-io", "-p", name, "-j", "2"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        output = []
        for line in process.stdout:
            token = os.environ.get("CARGO_REGISTRY_TOKEN")
            line = line.replace(token, "***") if token else line
            print(line, end="", flush=True)
            output.append(line)
        code = process.wait()
        # Cargo may report an index timeout after a successful upload.
        for poll in range(13):
            entry = registry(name).get(version)
            if entry:
                confirm(name, version, entry)
                print(f"CONFIRMED {name}@{version}: registry checksum matches", flush=True)
                return
            if poll < 12:
                time.sleep(5)
        retry = retry_time("".join(output))
        if retry is None or attempt == 3:
            raise RuntimeError(f"{name}@{version}: publish unconfirmed (cargo exit {code}); rerun to resume")
        until = datetime.fromtimestamp(retry, timezone.utc).isoformat()
        print(f"RATE LIMIT: retry {name}@{version} after {until}", flush=True)
        time.sleep(max(0, retry - time.time()) + 2)


def run_batch(packages, order, limit=BATCH_SIZE, lookup=registry, publish=publish_one):
    pending = missing(packages, order, lookup)
    total = len(order)
    done = total - len(pending)
    print(f"Registry checkpoint: {done}/{total} workspace versions published", flush=True)
    for name in pending[:limit]:
        publish(name, packages[name]["version"])
        done += 1
        print(f"Progress: {done}/{total}; resume uses exact registry versions", flush=True)
    return max(0, len(pending) - limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--publish", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    packages, graph = workspace()
    order = ordered(graph)
    if args.publish:
        if not os.environ.get("CARGO_REGISTRY_TOKEN"):
            raise RuntimeError("Configure the repository Actions secret CARGO_REGISTRY_TOKEN")
        run_batch(packages, order)
    else:
        pending = missing(packages, order)
        print(f"{len(order) - len(pending)}/{len(order)} versions published; {len(pending)} pending")
        for name in pending:
            print(f"  {name}@{packages[name]['version']}")
        if args.check and pending:
            raise RuntimeError("Workspace publication is incomplete; rerun the workflow to resume")
        if args.plan:
            plan = json.dumps(matrix(len(pending)))
            print(plan)
            if output := os.environ.get("GITHUB_OUTPUT"):
                with open(output, "a", encoding="utf-8") as file:
                    file.write(f"matrix={plan}\npending={len(pending)}\n")


if __name__ == "__main__":
    main()
