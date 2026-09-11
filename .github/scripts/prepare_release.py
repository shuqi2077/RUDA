import os
from pathlib import Path
import subprocess

import publish


ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def update_versions(*extra_args):
    _, graph = publish.workspace()
    order = publish.ordered(graph)
    for index, name in enumerate(order, 1):
        print(f"Prepare versions {index}/{len(order)}: {name}", flush=True)
        subprocess.run(["release-plz", "update", "--package", name, *extra_args],
                       cwd=ROOT, check=True)
        commit_versions()


def commit_versions():
    changed = git("diff", "--name-only", "-z").split("\0")
    changed = [name for name in changed if name]
    untracked = git("ls-files", "--others", "--exclude-standard")
    if untracked or any(name != "Cargo.lock" and Path(name).name != "Cargo.toml" for name in changed):
        raise RuntimeError("Release preparation changed files other than Cargo manifests/lockfile")
    subprocess.run(["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
                   cwd=ROOT, stdout=subprocess.DEVNULL, check=True)
    if changed:
        git("diff", "--check")
        git("add", "--", *changed)
        git("commit", "-m", "chore: update crate versions")


def prepare():
    if git("status", "--porcelain"):
        raise RuntimeError("Release preparation requires a clean checkout")
    update_versions()
    commit_versions()
    revision = git("rev-parse", "HEAD")
    git("push", "origin", "HEAD:refs/heads/main")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a", encoding="utf-8") as file:
            file.write(f"revision={revision}\n")
    print(f"Release source: {revision}", flush=True)
    return revision


if __name__ == "__main__":
    prepare()
