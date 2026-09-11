import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import prepare_release

UPDATE_VERSIONS = prepare_release.update_versions


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ruda-release-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        self.remote = self.base / "remote.git"
        self.root.mkdir()
        self.command("git", "init", "-b", "main")
        self.command("git", "config", "user.name", "Release test")
        self.command("git", "config", "user.email", "release-test@example.invalid")
        self.command("git", "config", "core.autocrlf", "false")
        (self.root / ".gitignore").write_text("/target/\n")
        (self.root / "Cargo.toml").write_text('[workspace]\nmembers = ["core", "consumer"]\nresolver = "2"\n')
        for name in ["core", "consumer"]:
            path = self.root / name
            (path / "src").mkdir(parents=True)
            manifest = ('[package]\nname = "ruda-release-test-'+name+'"\nversion = "0.1.0"\n'
                        'edition = "2021"\nlicense = "MIT"\ndescription = "Release test fixture"\n')
            if name == "consumer":
                manifest += '[dependencies]\nruda-release-test-core = { path = "../core", version = "0.1.0" }\n'
            (path / "Cargo.toml").write_text(manifest)
            (path / "src/lib.rs").write_text("pub fn value() -> u32 { 1 }\n")
        shutil.copyfile(prepare_release.ROOT / "release-plz.toml", self.root / "release-plz.toml")
        self.command("cargo", "generate-lockfile", "--offline")
        self.commit("feat: initial packages")
        self.command("git", "clone", "--bare", str(self.root), str(self.remote))
        self.command("git", "remote", "add", "origin", str(self.remote))
        self.baseline = self.base / "baseline"
        self.command("git", "clone", str(self.remote), str(self.baseline))
        self.mark_published()
        self.scope = patch.object(prepare_release, "ROOT", self.root)
        self.scope.start()
        self.addCleanup(self.scope.stop)
        self.publisher_scope = patch.object(prepare_release.publish, "ROOT", self.root)
        self.publisher_scope.start()
        self.addCleanup(self.publisher_scope.stop)
        self.output = self.base / "output"
        self.environment = patch.dict(os.environ, {"GITHUB_OUTPUT": str(self.output)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def command(self, *command):
        return subprocess.check_output(command, cwd=self.root, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8").strip()

    def commit(self, message):
        self.command("git", "add", ".")
        self.command("git", "commit", "-m", message)

    def invoke(self, update):
        with patch.object(prepare_release, "update_versions", update), contextlib.redirect_stdout(io.StringIO()):
            return prepare_release.prepare()

    def real_update(self):
        tool = shutil.which("release-plz")
        if not tool:
            self.skipTest("release-plz is required for real version calculation")
        UPDATE_VERSIONS("--registry-manifest-path", str(self.baseline / "Cargo.toml"),
                        "--repo-url", "https://github.com/shuqi2077/RUDA")

    def version(self, name):
        return tomllib.loads((self.root / name / "Cargo.toml").read_text())["package"]["version"]

    def mark_published(self):
        self.command("cargo", "package", "--workspace", "--no-verify", "--offline")
        for name in ["core", "consumer"]:
            archive = self.root / "target/package" / f"ruda-release-test-{name}-{self.version(name)}.crate"
            with tarfile.open(archive) as package:
                for member in package.getmembers():
                    if not member.isfile():
                        continue
                    relative = Path(*Path(member.name).parts[1:])
                    destination = self.baseline / name / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(package.extractfile(member).read())

    def test_no_changes_does_not_create_a_commit(self):
        before = self.command("git", "rev-parse", "HEAD")
        self.assertEqual(self.invoke(lambda: None), before)
        self.assertEqual(self.command("git", "rev-parse", "HEAD"), before)
        self.assertIn("revision="+before, self.output.read_text())

    def test_dirty_checkout_rejected_before_update(self):
        (self.root / "private-report.json").write_text("{}")
        with patch.object(prepare_release, "update_versions") as update, self.assertRaises(RuntimeError):
            prepare_release.prepare()
        update.assert_not_called()

    def test_untracked_report_is_never_committed(self):
        before = self.command("git", "rev-parse", "HEAD")
        with self.assertRaises(RuntimeError):
            self.invoke(lambda: (self.root / "private-report.json").write_text("{}"))
        self.assertEqual(self.command("git", "rev-parse", "HEAD"), before)
        self.assertFalse(self.output.exists())

    def test_source_edits_are_never_committed(self):
        with self.assertRaises(RuntimeError):
            self.invoke(lambda: (self.root / "core/src/lib.rs").write_text("pub fn changed() {}"))
        self.assertEqual(self.command("git", "diff", "--cached", "--name-only"), "")

    def test_real_patch_and_dependency_update(self):
        (self.root / "core/src/lib.rs").write_text("pub fn value() -> u32 { 2 }\n")
        self.commit("fix(core): correct returned value")
        revision = self.invoke(self.real_update)
        self.assertEqual(self.version("core"), "0.1.1")
        consumer = tomllib.loads((self.root / "consumer/Cargo.toml").read_text())
        self.assertIn("0.1.1", consumer["dependencies"]["ruda-release-test-core"]["version"])
        self.assertEqual(self.command("git", "ls-remote", "origin", "refs/heads/main").split()[0], revision)
        self.assertFalse(list(self.root.rglob("CHANGELOG.md")))
        self.assertTrue(all(Path(name).name in ["Cargo.toml", "Cargo.lock"]
                            for name in self.command("git", "diff", "--name-only", "HEAD^", "HEAD").splitlines()))
        self.assertEqual(self.invoke(self.real_update), revision)
        self.mark_published()
        self.assertEqual(self.invoke(self.real_update), revision)

    def test_real_breaking_change_uses_minor_in_zero_series(self):
        (self.root / "core/src/lib.rs").write_text("pub fn replacement() -> u32 { 2 }\n")
        self.commit("feat(core)!: replace public function")
        self.invoke(self.real_update)
        self.assertEqual(self.version("core"), "0.2.0")

    def test_real_feature_dev_cycle_does_not_block_preparation(self):
        core = self.root / "core/Cargo.toml"
        core.write_text(core.read_text()+'\n[features]\ntrace = ["ruda-release-test-consumer/trace"]\n'
                        '[dev-dependencies]\nruda-release-test-consumer = { path = "../consumer" }\n')
        consumer = self.root / "consumer/Cargo.toml"
        consumer.write_text(consumer.read_text()+'\n[features]\ntrace = []\n')
        self.command("cargo", "generate-lockfile", "--offline")
        self.commit("fix: update packages with a feature dev cycle")
        self.invoke(self.real_update)
        self.assertEqual(self.version("core"), "0.1.1")
        self.assertIn('path = "../consumer"', core.read_text())

    def test_concurrent_push_is_not_overwritten(self):
        other = self.base / "other"
        self.command("git", "clone", str(self.remote), str(other))
        for args in [("config", "user.name", "Other"), ("config", "user.email", "other@example.invalid")]:
            subprocess.run(["git", "-C", str(other), *args], check=True, capture_output=True)
        (other / "new-source.rs").write_text("// concurrent change\n")
        for args in [("add", "."), ("commit", "-m", "fix: concurrent change"), ("push", "origin", "main")]:
            subprocess.run(["git", "-C", str(other), *args], check=True, capture_output=True)
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(lambda: None)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
