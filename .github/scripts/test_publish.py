import contextlib
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import publish


class PublisherTests(unittest.TestCase):
    def test_dependency_order(self):
        graph = {"app": {"core", "macros"}, "core": {"macros"}, "macros": set()}
        self.assertEqual(publish.ordered(graph), ["macros", "core", "app"])

    def test_cycle_rejected(self):
        with self.assertRaises(RuntimeError):
            publish.ordered({"a": {"b"}, "b": {"a"}})

    def test_nonpublishable_dependency_rejected(self):
        with self.assertRaises(RuntimeError):
            publish.ordered({"a": {"private"}})

    def test_sparse_index_paths(self):
        expected = {"a": "1/a", "ab": "2/ab", "ABC": "3/a/abc", "ruFFT": "ru/ff/rufft"}
        for name, path in expected.items():
            self.assertEqual(publish.index_path(name), path)

    def test_exact_version_skip(self):
        packages = {"a": {"version": "0.1.1"}, "b": {"version": "0.1.0"}}
        self.assertEqual(publish.missing(packages, ["a", "b"], lambda _: {"0.1.0": {}}), ["a"])

    def test_rate_limit_utc(self):
        text = "429 Too Many Requests. Please try again after 2026-09-10T14:30:53Z"
        self.assertEqual(publish.retry_time(text), 1789050653)
        self.assertIsNone(publish.retry_time("error: compilation failed"))
        with self.assertRaises(RuntimeError):
            publish.retry_time("429 without a retry timestamp")

    def test_batch_count(self):
        self.assertEqual(publish.matrix(0), {"batch": [0]})
        self.assertEqual(len(publish.matrix(51)["batch"]), 11)

    def test_rate_limit_http_date(self):
        text = (
            "the remote server responded with an error (status 429 Too Many Requests): "
            "You have published too many new crates in a short period of time. "
            "Please try again after Thu, 10 Sep 2026 15:20:53 GMT "
            "and see https://crates.io/docs/rate-limits for more details."
        )
        self.assertEqual(publish.retry_time(text), 1789053653)

    def test_publish_retries_http_date_rate_limit(self):
        limited = Mock()
        limited.stdout = ["429 Too Many Requests: Please try again after Thu, 10 Sep 2026 15:20:53 GMT"]
        limited.wait.return_value = 1
        successful = Mock()
        successful.stdout = ["Published a v1.0.0"]
        successful.wait.return_value = 0
        entry = {"cksum": "archive-checksum"}
        with (
            patch.object(publish, "registry", side_effect=[{}] * 15 + [{"1.0.0": entry}]),
            patch.object(publish.subprocess, "Popen", side_effect=[limited, successful]) as cargo,
            patch.object(publish.time, "sleep") as sleep,
            patch.object(publish.time, "time", return_value=1789053643),
            patch.object(publish, "confirm") as confirm,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            publish.publish_one("a", "1.0.0")
        self.assertEqual(cargo.call_count, 2)
        self.assertEqual(cargo.call_args_list[0], cargo.call_args_list[1])
        sleep.assert_called_with(12)
        confirm.assert_called_once_with("a", "1.0.0", entry)

    def test_resume_does_not_republish_completed_versions(self):
        packages = {name: {"version": "1.0.0"} for name in ["a", "b", "c"]}
        registry = {}
        calls = []

        def upload(name, version):
            calls.append(name)
            registry[name] = {version: {"cksum": name}}

        lookup = lambda name: registry.get(name, {})
        with contextlib.redirect_stdout(io.StringIO()):
            left = publish.run_batch(packages, list(packages), 1, lookup, upload)
            self.assertEqual(left, 2)
            left = publish.run_batch(packages, list(packages), 5, lookup, upload)
        self.assertEqual(left, 0)
        self.assertEqual(calls, ["a", "b", "c"])

    def test_failed_upload_does_not_hide_remaining_work(self):
        packages = {name: {"version": "1.0.0"} for name in ["a", "b"]}
        calls = []

        def upload(name, version):
            calls.append(name)
            raise RuntimeError("upload failed")

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            publish.run_batch(packages, list(packages), 5, lambda _: {}, upload)
        self.assertEqual(calls, ["a"])

    def test_archive_checksum(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CARGO_TARGET_DIR": directory}):
            package = Path(directory) / "package"
            package.mkdir()
            (package / "a-1.0.0.crate").write_bytes(b"archive")
            publish.confirm("a", "1.0.0", {"cksum": hashlib.sha256(b"archive").hexdigest()})
            with self.assertRaises(RuntimeError):
                publish.confirm("a", "1.0.0", {"cksum": "wrong"})


if __name__ == "__main__":
    unittest.main()
