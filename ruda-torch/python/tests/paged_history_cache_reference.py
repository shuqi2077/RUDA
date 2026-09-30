"""TEST ONLY: mathematical cached-history oracle and logical load accounting.

Uses CPU double arithmetic. It neither interprets DSL/PTX nor predicts hardware
transactions; the counters exclude the query/statistics kernel.
"""
from collections import Counter
import torch
from paged_pruning_reference import encode, visible_rows


class HistoryReads:
    def __init__(self, key, value, position, *, mla, cached, score_gradient):
        self.key, self.value, self.position = key, value, position
        self.mla, self.cached, self.score_gradient = mla, cached, score_gradient
        self.rows = {}
        self.counts = Counter()

    def _read(self, name, tensor, location):
        value = tensor[location]
        self.counts[name + "_elements"] += value.numel()
        self.counts[name + "_rows"] += 1
        return value

    def get(self, page, offset, head):
        location = (page, offset, head)
        if self.cached and location in self.rows:
            return self.rows[location]
        key = self._read("key", self.key, location)
        position = self._read("position", self.position, (page, offset, 0)) if self.mla else None
        value = None
        if self.score_gradient:
            value = key if self.mla and self.cached else self._read("value", self.value, location)
        result = key, value, position
        if self.cached:
            self.rows[location] = result
        return result


def cached_vjp(tensors, grad, needs, *, mla, spec, causal=True, scale=.37, cached=True):
    """Mathematical reference; returns (gradients, logical history-load counts)."""
    vals = [x.detach().double() for x in tensors]
    g = grad.double()
    if mla:
        q, qp, k, kp = vals
        v = k
        history = needs[2] or needs[3]
        score_gradient = history
    else:
        q, k, v = vals
        qp = kp = None
        history = needs[1] or needs[2]
        score_gradient = needs[1]
    outputs = [torch.zeros_like(x) if need else None for x, need in zip(vals, needs)]
    index = encode(spec)
    ebase, sbase, qbase = index[:3]
    p = spec['page_size']
    stats = q.new_zeros(q.shape[0], q.shape[1], 3)
    for row, seq in enumerate(spec['sequence_ids']):
        end = spec['kv_lengths'][seq]
        if causal:
            end = min(end, spec['positions'][row] + 1)
        if not end:
            continue
        for h in range(q.shape[1]):
            kh = h // (q.shape[1] // k.shape[2])
            slots = [(spec['block_tables'][seq][i // p], i % p) for i in range(end)]
            kk = torch.stack([k[a, b, kh] for a, b in slots])
            vv = torch.stack([v[a, b, kh] for a, b in slots])
            scores = kk @ q[row, h]
            if mla:
                pp = torch.stack([kp[a, b, 0] for a, b in slots])
                scores = scores + pp @ qp[row, h]
            scores = scores * scale
            maximum = scores.max()
            ex = (scores - maximum).exp()
            den = ex.sum()
            prob = ex / den
            dp = vv @ g[row, h]
            expected = (prob * dp).sum()
            stats[row, h] = torch.stack([maximum, den, expected])
            ds = prob * (dp - expected) * scale
            if needs[0]:
                outputs[0][row, h] = ds @ kk
            if mla and needs[1]:
                outputs[1][row, h] = ds @ pp
    reads = HistoryReads(k, v, kp, mla=mla, cached=cached, score_gradient=score_gradient)
    if history:
        for page in range(spec['num_pages']):
            for off in range(p):
                for kh in range(k.shape[2]):
                    for link in range(index[3 + page], index[4 + page]):
                        seq, logical = index[ebase + 2 * link:ebase + 2 * link + 2]
                        token = logical * p + off
                        if token >= spec['kv_lengths'][seq]:
                            continue
                        for row in visible_rows(spec, index, seq, token, causal):
                            group = q.shape[1] // k.shape[2]
                            for h in range(kh * group, (kh + 1) * group):
                                maximum, den, expected = stats[row, h]
                                if not den:
                                    continue
                                kk, vv, pp = reads.get(page, off, kh)
                                score = q[row, h] @ kk
                                if mla:
                                    score = score + qp[row, h] @ pp
                                prob = (score * scale - maximum).exp() / den
                                ds = 0.
                                if score_gradient:
                                    dp = vv @ g[row, h]
                                    ds = prob * (dp - expected) * scale
                                if mla:
                                    if needs[2]:
                                        outputs[2][page, off, 0] += ds * q[row, h] + prob * g[row, h]
                                    if needs[3]:
                                        outputs[3][page, off, 0] += ds * qp[row, h]
                                else:
                                    if needs[1]:
                                        outputs[1][page, off, kh] += ds * q[row, h]
                                    if needs[2]:
                                        outputs[2][page, off, kh] += prob * g[row, h]
    return tuple(None if o is None else o.to(t.dtype) for o, t in zip(outputs, tensors)), reads.counts
