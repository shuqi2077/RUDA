#!/usr/bin/env python3
"""Independent mathematical + source checks. DOES NOT execute Rust or a GPU.

No third-party packages required. The checked source API snapshot is a guard
against accidental field removal, not a substitute for compiling callers.
"""
from __future__ import annotations
import itertools
import json
import math
import random
import re
import statistics
import struct
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COUNTS = {"subgroup_cases": 0, "layout_cases": 0, "kv_budget_cases": 0, "numeric_cases": 0, "recovery_cases": 0}
U64 = (1 << 64) - 1

def checked_product(dims):
    if 0 in dims:
        return 0
    result = math.prod(dims)
    if result > U64:
        raise ValueError("overflow")
    return result

def layout_span(dims, strides, offset):
    if len(dims) != len(strides):
        raise ValueError("rank mismatch")
    checked_product(dims)
    if 0 in dims:
        return 0
    result = offset + sum((d - 1) * s for d, s in zip(dims, strides)) + 1
    if result > U64:
        raise ValueError("overflow")
    return result

def fixed_subgroup(lo, hi, operations, packed, max_x, max_threads):
    valid = lo > 0 and lo == hi and lo & (lo - 1) == 0 and operations and packed and lo <= min(max_x, max_threads)
    return lo if valid else None

def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)

def rms_reference(values, gamma, epsilon):
    if not values or len(values) != len(gamma) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid norm parameters")
    if any(not math.isfinite(v) for v in values + gamma):
        raise ValueError("nonfinite")
    scale = max(math.sqrt(epsilon), max(map(abs, values)))
    scaled = [v / scale for v in values]
    denominator = math.sqrt(math.fsum(v * v for v in scaled) / len(values) + epsilon / scale / scale)
    return [v / denominator * g for v, g in zip(scaled, gamma)]

def recovery(kind, state, intact, attempts, limit, floor, portable):
    if state == "unknown" or not intact or kind in {"device_lost", "synchronization", "panic", "internal"}:
        return "quarantine", None
    if attempts >= 2:
        return "fail", None
    if kind == "oom" and 0 < floor < limit:
        return "retry", max(floor, limit // 2)
    if kind == "unsupported" and not portable:
        return "portable", None
    return "fail", None

def public_struct(source, name):
    match = re.search(r"\bpub\s+struct\s+" + re.escape(name) + r"\b[^\{;]*\{", source)
    if not match:
        raise ValueError(f"missing public struct {name}")
    start = match.end()
    depth, end = 1, start
    while depth:
        char = source[end]
        depth += (char == "{") - (char == "}")
        end += 1
    body = re.sub(r"//[^\n]*", "", source[start:end-1])
    return re.sub(r"\s+", " ", body).strip()

class RuntimeReferenceTests(unittest.TestCase):
    def test_subgroup_and_two_lane_load_matrix(self):
        for lo, hi, ops, packed, maximum, head, sequence in itertools.product(
            [0, 16, 32, 48, 64, 128], [0, 16, 32, 64, 128], [False, True], [False, True],
            [32, 64, 256], [32, 64, 80, 128, 256], [1, 7]
        ):
            width = fixed_subgroup(lo, hi, ops, packed, maximum, maximum)
            eligible = width is not None and head == 2 * width and sequence == 1
            if eligible:
                positions = [lane + slot * width for lane in range(width) for slot in [0, 1]]
                self.assertEqual(sorted(positions), list(range(head)))
                self.assertEqual(len(set(positions)), head)
            if lo != hi or not ops or not packed or maximum < lo or lo == 48:
                self.assertFalse(eligible)
            COUNTS["subgroup_cases"] += 1

    def test_storage_span_matches_enumerated_addresses(self):
        rng = random.Random(271828)
        for _ in range(3000):
            rank = rng.randrange(5)
            dims = [rng.randrange(5) for _ in range(rank)]
            strides = [rng.randrange(9) for _ in range(rank)]
            offset = rng.randrange(8)
            addresses = [offset + sum(i * s for i, s in zip(index, strides))
                         for index in itertools.product(*(range(d) for d in dims))]
            expected = max(addresses) + 1 if addresses else 0
            self.assertEqual(layout_span(dims, strides, offset), expected)
            for bits in [4, 8, 16, 32, 64]:
                self.assertEqual((expected * bits + 7) // 8,
                                 expected * bits // 8 + int(expected * bits % 8 != 0))
            COUNTS["layout_cases"] += 1

    def test_storage_overflow_and_empty_shape(self):
        self.assertEqual(checked_product([U64, 0, U64]), 0)
        for args in [([U64, 2], [2, 1], 0), ([1], [1], U64), ([2], [U64], 1)]:
            with self.assertRaises(ValueError):
                layout_span(*args)

    def test_dense_kv_payload_and_budget(self):
        rng = random.Random(314159)
        for _ in range(5000):
            layers, heads, width, page = [rng.choice([1, 2, 3, 8, 16, 32, 64, 128]) for _ in range(4)]
            bytes_per_value = rng.choice([2, 4, 8])
            size = 2 * layers * heads * width * page * bytes_per_value
            count, remainder = rng.randrange(1, 256), rng.randrange(size)
            workspace, headroom = rng.randrange(100000), rng.randrange(100000)
            available = size * count + remainder + workspace + headroom
            actual = (available - workspace - headroom) // size
            self.assertEqual(actual, count)
            self.assertLess(available - workspace - headroom - actual * size, size)
            COUNTS["kv_budget_cases"] += 1

    def test_norm_overflow_recovery_is_finite(self):
        for width, magnitude in itertools.product([1, 7, 31, 32, 33, 64, 65, 129, 896], [0., 1e-30, 1., 1e20, 1e30]):
            values = [f32(magnitude) * (-1 if i % 2 else 1) for i in range(width)]
            actual = rms_reference(values, [1.] * width, 1e-6)
            self.assertTrue(all(math.isfinite(x) for x in actual))
            expected_mag = magnitude / math.sqrt(magnitude * magnitude + 1e-6)
            self.assertTrue(all(abs(abs(x) - expected_mag) < 1e-6 for x in actual))
            if magnitude >= 1e20:
                self.assertTrue(math.isinf(f32(values[0] * values[0])))
                self.assertGreater(abs(actual[0]), .999)
            COUNTS["numeric_cases"] += 1

    def test_softmax_extreme_f32_values(self):
        rows = [[f32(3.4e38), -f32(3.4e38), 0.], [-10000., -10001.], [0., -math.inf, 0.]]
        for row in rows:
            maximum = max(row)
            exponentials = [math.exp(x - maximum) for x in row]
            total = math.fsum(exponentials)
            probabilities = [x / total for x in exponentials]
            self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities))
            self.assertAlmostEqual(math.fsum(probabilities), 1.)
            COUNTS["numeric_cases"] += 1

    def test_recovery_never_retries_unknown_or_corrupt_work(self):
        for kind, state, intact, attempts, limit, floor, portable in itertools.product(
            ["oom", "unsupported", "numerical", "driver", "device_lost", "synchronization", "panic", "internal"],
            ["not_submitted", "quiescent", "unknown"], [False, True], [0, 1, 2, 100],
            [1, 7, 8, 16], [1, 7], [False, True]
        ):
            action, new_limit = recovery(kind, state, intact, attempts, limit, floor, portable)
            if state == "unknown" or not intact:
                self.assertEqual(action, "quarantine")
            if action == "retry":
                self.assertEqual(kind, "oom")
                self.assertLess(attempts, 2)
                self.assertLess(new_limit, limit)
                self.assertGreaterEqual(new_limit, floor)
            if kind == "numerical":
                self.assertNotIn(action, {"retry", "portable"})
            COUNTS["recovery_cases"] += 1

    def test_performance_median_does_not_use_single_outlier(self):
        self.assertLess(statistics.median([10., .9, .9]), 1.15)
        self.assertGreater(statistics.median([1.5, 1.4, 1.3]), 1.15)
        self.assertEqual(statistics.median([.8, 1., 1.2, 1.4]), 1.1)

    def test_original_public_struct_fields_are_unchanged(self):
        snapshot = json.loads((ROOT / "ruLLM/tests/fixtures/public_api_v1.json").read_text())
        for record in snapshot:
            source = (ROOT / record["file"]).read_text()
            self.assertEqual(public_struct(source, record["name"]), record["body"], record["name"])

    def test_features_lockfile_and_examples_are_connected(self):
        manifest = tomllib.loads((ROOT / "ruLLM/Cargo.toml").read_text())
        self.assertEqual(manifest["features"]["default"], [])
        self.assertIn("dep:ruda-driver-hip", manifest["features"]["amd"])
        self.assertIn("ruda-tensor/hip", manifest["features"]["amd"])
        for name, dep in manifest["dependencies"].items():
            if isinstance(dep, dict) and "path" in dep:
                self.assertTrue((ROOT / "ruLLM" / dep["path"] / "Cargo.toml").is_file(), name)
        lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
        package = next(p for p in lock["package"] if p["name"] == "ruda-llm")
        self.assertIn("ruda-driver-hip", package["dependencies"])
        for example in manifest["example"]:
            self.assertTrue((ROOT / "ruLLM/examples" / (example["name"] + ".rs")).is_file())
            for feature in example.get("required-features", []):
                self.assertIn(feature, manifest["features"])

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeReferenceTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"kind": "independent_python_reference_and_source_checks", "executes_rust": False,
                      "executes_gpu": False, "counts": COUNTS, "passed": result.wasSuccessful()}, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
