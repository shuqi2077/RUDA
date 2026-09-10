#!/usr/bin/env python3
"""Independent Python algorithm/integrity checks, NOT Rust compilation or tests.

Run from any directory: python3 ruLLM/tools/verify_reference.py
This checks reference mathematics and artifact consistency. It cannot validate
Rust typing, backend execution, generated machine code, or performance.
"""
from __future__ import annotations

import math
import random
import struct
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COUNTS = {"distribution_cases": 0, "stop_match_steps": 0, "admission_cases": 0}


def nth_boundary(logits: list[float], rank: int) -> float:
    """An independent rank selection, without full sorting (small test inputs)."""
    candidates = list(logits)
    while True:
        pivot = candidates[len(candidates) // 2]
        smaller = [x for x in candidates if x < pivot]
        equal = [x for x in candidates if x == pivot]
        if rank < len(smaller):
            candidates = smaller
        elif rank < len(smaller) + len(equal):
            return pivot
        else:
            candidates = [x for x in candidates if x > pivot]
            rank -= len(smaller) + len(equal)


def ordered_sum(values) -> float:
    # Explicit left fold; Python versions may optimize builtin sum differently.
    result = 0.0
    for value in values:
        result += value
    return result


def distribution(logits: list[float], temperature: float, k: int, p: float, *, legacy: bool):
    maximum = max(logits)
    order = list(range(len(logits)))
    if legacy:
        if k > 0 or p < 1.0:
            order.sort(key=lambda i: (logits[i], i))
        threshold = logits[order[len(logits) - min(k, len(logits))]] if k else -math.inf
    else:
        threshold = nth_boundary(logits, len(logits) - k) if 0 < k < len(logits) else -math.inf
    probabilities = [
        0.0 if value < threshold else math.exp((value - maximum) / temperature)
        for value in logits
    ]
    total = ordered_sum(probabilities)
    if p < 1.0:
        if not legacy:
            order = sorted((i for i in order if logits[i] >= threshold), key=lambda i: (logits[i], i))
        cumulative = 0.0
        for token in order[:-1]:
            cumulative += probabilities[token] / total
            if cumulative <= 1.0 - p:
                probabilities[token] = 0.0
    retained = ordered_sum(probabilities)
    return [value / retained for value in probabilities]


class Matcher:
    def __init__(self, patterns):
        self.patterns = patterns
        self.prefixes = []
        self.matched = [0] * len(patterns)
        for pattern in patterns:
            prefix = [0] * len(pattern)
            for index in range(1, len(pattern)):
                length = prefix[index - 1]
                while length and pattern[index] != pattern[length]:
                    length = prefix[length - 1]
                if pattern[index] == pattern[length]:
                    length += 1
                prefix[index] = length
            self.prefixes.append(prefix)

    def push(self, token):
        first = None
        for index, pattern in enumerate(self.patterns):
            prefix = self.prefixes[index]
            length = self.matched[index]
            while length and pattern[length] != token:
                length = prefix[length - 1]
            if pattern[length] == token:
                length += 1
            if length == len(pattern):
                if first is None:
                    first = index
                length = prefix[length - 1]
            self.matched[index] = length
        return first


class ReferenceChecks(unittest.TestCase):
    def test_sampling_equivalence(self):
        rng = random.Random(2026)
        for size in (1, 2, 3, 4, 17, 65, 257):
            for _ in range(24):
                logits = [rng.choice((-math.inf, -10.0, -1.0, -0.0, 0.0, 1.0, 3.0, 10.0)) for _ in range(size)]
                logits[rng.randrange(size)] = 0.0
                for temperature in (1e-300, 0.7, 1.0, 2.0, 1e300):
                    for k in (0, 1, 2, size, 2**64 - 1):
                        for p in (0.01, 0.5, 0.9, 1.0):
                            old = distribution(logits, temperature, k, p, legacy=True)
                            new = distribution(logits, temperature, k, p, legacy=False)
                            self.assertEqual([struct.pack('d', x) for x in old], [struct.pack('d', x) for x in new])
                            self.assertAlmostEqual(ordered_sum(new), 1.0, places=12)
                            self.assertTrue(all(math.isfinite(x) and x >= 0 for x in new))
                            COUNTS['distribution_cases'] += 1

    def test_incremental_stop_matching(self):
        rng = random.Random(42)
        for _ in range(128):
            patterns = [[rng.randrange(4) for _ in range(rng.randrange(1, 9))] for _ in range(6)]
            matcher = Matcher(patterns)
            history = []
            for _ in range(256):
                token = rng.randrange(4)
                history.append(token)
                expected = next((i for i, p in enumerate(patterns) if history[-len(p):] == p), None)
                self.assertEqual(matcher.push(token), expected)
                COUNTS['stop_match_steps'] += 1

    def test_conservative_admission_bound(self):
        # An admitted request consumes prompt tokens once and all but the final
        # selected token as cached input. Every intermediate occupancy fits.
        for block in range(1, 17):
            for prompt in range(1, 33):
                for generated in range(1, 33):
                    budget = (prompt + generated - 1 + block - 1) // block
                    for completed in range(generated):
                        used = (prompt + completed + block - 1) // block
                        self.assertLessEqual(used, budget)
                    COUNTS['admission_cases'] += 1

    def test_manifest_and_lock_consistency(self):
        manifest = tomllib.loads((ROOT / 'ruLLM/Cargo.toml').read_text())
        lock = tomllib.loads((ROOT / 'Cargo.lock').read_text())
        packages = lock['package']
        llm = next(p for p in packages if p['name'] == 'ruda-llm')
        self.assertIn('ruda-tensor-host', llm['dependencies'])
        self.assertIn('ruda-tensor-host', manifest['dev-dependencies'])
        self.assertTrue(any(p['name'] == 'ruda-tensor-host' for p in packages))
        for example in manifest.get('example', []):
            self.assertTrue((ROOT / 'ruLLM/examples' / (example['name'] + '.rs')).is_file())
        for dependency in (*manifest['dependencies'].values(), *manifest['dev-dependencies'].values()):
            if isinstance(dependency, dict) and 'path' in dependency:
                self.assertTrue((ROOT / 'ruLLM' / dependency['path'] / 'Cargo.toml').is_file())

    def test_new_modules_are_wired(self):
        expected = {
            'src/generation.rs': ['mod control;', 'generate_causal_greedy_stream', 'generate_causal_sampled_stream'],
            'src/generation/sampling.rs': ['mod distribution;', 'clear_workspace'],
            'src/continuous_batch.rs': ['mod lifecycle;', 'commit_appends', 'cancel_appends'],
            'src/paged_kv.rs': ['mod tests;', 'pub fn free_page_count', 'pub fn commit_appends'],
            'src/lib.rs': ['ContinuousBatchOptions', 'KvAdmissionPolicy', 'CancelledGeneration'],
        }
        for name, entries in expected.items():
            source = (ROOT / 'ruLLM' / name).read_text()
            for entry in entries:
                self.assertIn(entry, source, f'{name}: missing {entry}')
        scheduler = (ROOT / 'ruLLM/src/continuous_batch.rs').read_text()
        snapshot = scheduler.split('pub fn snapshot(&self)', 1)[1].split('pub fn kv_cache', 1)[0]
        self.assertNotIn('kv_cache.snapshot()', snapshot)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print('\nIndependent Python reference cases:', COUNTS)
    print('Rust compilation/tests and GPU benchmarks were NOT performed by this script.')
    raise SystemExit(not result.wasSuccessful())
