# RUDA SOL-ExecBench shared helpers (appended after sol_launcher.py).
# Kernel names are passed as literals by the solution sections so the packager
# bundles exactly the PTX each problem needs.
import math

_KIND_BYTES = {'h': 2, 'b': 2, 't': 4, 'e': 1, 'g': 1, 'f': 4}


_BIG = 1 << 30


class _Op:
    """One GEMM operand as a 3D TMA view (inner, Y, Z) of `tensor` plus the affine
    batch -> (y, z) coordinate map:
      z = (batch // z_div) * z_mul + position   (position = MN row for K-major, K row for MN-major)
      y = ((batch % y_mod) // y_div) * y_mul + y_add + k_group * y_kstep"""
    __slots__ = ('tensor', 'dims', 'strides', 'coords')

    def __init__(self, tensor, dims, strides, z_div=_BIG, z_mul=0, y_mod=_BIG, y_div=1, y_mul=0, y_add=0, y_kstep=0):
        self.tensor, self.dims, self.strides = tensor, dims, strides
        self.coords = [z_div, z_mul, y_mod, y_div, y_mul, y_add, y_kstep]


def _op(t):
    """2D (rows, inner) operand shared by every batch."""
    if t.stride(-1) != 1:
        t = t.contiguous()
    es = t.element_size()
    return _Op(t, [t.shape[1], 1, t.shape[0]], [t.stride(0) * es, t.stride(0) * es])


def _opb(t, div=1, mul=1, add=0, kstep=0):
    """3D (Bt, rows, inner) operand; batch -> slice (batch // div) * mul + add."""
    if t.stride(-1) != 1:
        t = t.contiguous()
    es = t.element_size()
    return _Op(t, [t.shape[2], t.shape[0], t.shape[1]], [t.stride(0) * es, t.stride(1) * es],
               y_div=div, y_mul=mul, y_add=add, y_kstep=kstep)


def _opl(t, heads, div=1, mul=1, add=0, kstep=0):
    """4D (B, L, H, D) operand with batch = b * heads + h: y = ((batch % heads) // div) * mul + add."""
    if t.stride(-1) != 1 or t.stride(0) != t.shape[1] * t.stride(1):
        t = t.contiguous()
    es = t.element_size()
    return _Op(t, [t.shape[3], t.shape[2], t.shape[0] * t.shape[1]], [t.stride(2) * es, t.stride(1) * es],
               z_div=heads, z_mul=t.shape[1], y_mod=heads, y_div=div, y_mul=mul, y_add=add, y_kstep=kstep)


def _gemm(name, a, b, out, m, n, k, batches=1, k_groups=1, ld=None, o_div=1, o_s1=0, o_s2=0,
          col_split=None, col_stride=0, bias=None, residual=None, scale_a=None, scale_b=None, alpha=1.0, beta=1.0,
          groups=None, max_tiles=None, bias_stride=0, res_s1=None):
    """out = epilogue(op(A) @ op(B)^T), kernel `name` =
    gemm_<in><out>_<bn>_<a_mn b_mn>_<bias act combine scale exact>. a / b are _Op;
    out is the output tensor; element offset of (batch t, row, col) =
    (t // o_div) * o_s1 + (t % o_div) * o_s2 + row * ld + (col // col_split) * col_stride + col % col_split
    (all in elements, multiples of 8)."""
    parts = name.split('_')
    kind, bn, layout = parts[1][0], int(parts[2]), parts[3]
    a_mn, b_mn = layout[0] == '1', layout[1] == '1'
    width = 128 // _KIND_BYTES[kind]
    ld = n if ld is None else ld
    if m == 0 or n == 0 or batches == 0:
        return out
    col_split = _BIG * 8 if col_split is None else col_split
    if n % 8 or ld % 8 or o_s1 % 8 or o_s2 % 8 or col_split % 8 or col_stride % 8 or out.data_ptr() % 16:
        raise ValueError('GEMM output needs N, ld and batch strides % 8 == 0')
    maps = [_tmap(a.tensor, a.dims, a.strides, [width, 1, width if a_mn else 128], atomic32=kind == 't' and a_mn),
            _tmap(b.tensor, b.dims, b.strides, [width, 1, width if b_mn else bn], atomic32=kind == 't' and b_mn)]
    tiles = ((m + 127) // 128) * ((n + bn - 1) // bn) * batches if max_tiles is None else max_tiles * ((n + bn - 1) // bn)
    workers = max(1, min(tiles, _sm_count(out.device)))
    dummy = out
    _launch(name, [a.tensor, b.tensor, out, bias if bias is not None else dummy,
                   residual if residual is not None else dummy,
                   scale_a if scale_a is not None else dummy, scale_b if scale_b is not None else dummy,
                   groups if groups is not None else dummy],
            (workers, 1, 1), {'I': [m, n, k, batches, k_groups, ld // 8, o_div, o_s1 // 8, o_s2 // 8,
                                    col_split // 8, col_stride // 8, bias_stride,
                                    (o_s1 if res_s1 is None else res_s1) // 8] + a.coords + b.coords,
                              'f': [float(alpha), float(beta)]}, maps)
    return out


def _linear(name, x, w, bias=None, residual=None, out=None, out_dtype=None, alpha=1.0, beta=1.0):
    """y = x @ w^T (+ fused epilogue) for x (..., K), w (N, K) (name layout '00')."""
    x2 = x.reshape(-1, x.shape[-1])
    n = w.shape[0]
    if out is None:
        out = torch.empty((x2.shape[0], n), dtype=out_dtype or x.dtype, device=x.device)
    r = residual.reshape(-1, n) if residual is not None else None
    _gemm(name, _op(x2), _op(w), out, x2.shape[0], n, x2.shape[1], bias=bias, residual=r, alpha=alpha, beta=beta)
    return out.view(*x.shape[:-1], n)


def _mm(name, a, b, out=None, out_dtype=None, bias=None, residual=None, alpha=1.0, beta=1.0):
    """2D out = a @ b with a (M, K) and b (K, N) row-major (name layout '01')."""
    m, k = a.shape
    n = b.shape[1]
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype or a.dtype, device=a.device)
    _gemm(name, _op(a), _op(b), out, m, n, k, bias=bias, residual=residual, alpha=alpha, beta=beta)
    return out


def _mmt(name, a, b, out=None, out_dtype=None, alpha=1.0):
    """2D out = a^T @ b with a (K, M) and b (K, N) row-major (name layout '11')."""
    k, m = a.shape
    n = b.shape[1]
    if out is None:
        out = torch.empty((m, n), dtype=out_dtype or a.dtype, device=a.device)
    _gemm(name, _op(a), _op(b), out, m, n, k, alpha=alpha)
    return out


def _split3(x, chunk, out_chunk, a_side, rows=None, stride=None):
    """Raw 3xTF32 split: every `chunk` elements of x (rows `stride` apart) ->
    [p0, p1, p2] chunks of `out_chunk` (zero padded). Returns a flat f32 tensor."""
    if stride is None:
        x = _aligned(x)
        rows, stride = x.numel() // chunk, chunk
    count = rows * 3 * out_chunk
    out = torch.empty(count, dtype=torch.float32, device=x.device)
    _launch('tf32_split_a' if a_side else 'tf32_split_b', [x, out], _blocks(count, 256), {'I': [count, chunk, out_chunk, stride]})
    return out


def _tf32x3(x, a_side):
    """3xTF32 operand planes (3, rows, inner): [hi, lo, hi] (A side) or [hi, hi, lo]
    (B side), consumed as K groups (k_groups=3, y = group). 2D x may have a row stride."""
    if x.dim() == 2 and x.stride(1) == 1 and x.stride(0) != x.shape[1]:
        # row-strided view: per-row [p0 | p1 | p2] layout used directly (y = group)
        rows, inner = x.shape
        flat = _split3(x, inner, inner, a_side, rows=rows, stride=x.stride(0))
        return _Op(flat, [inner, 3, rows], [inner * 4, 3 * inner * 4], y_kstep=1)
    x = _aligned(x).reshape(-1, x.shape[-1])
    out = _split3(x, x.numel(), x.numel(), a_side).view((3,) + tuple(x.shape))
    return _opb(out, div=1, mul=0, kstep=1)


def _gemm3(name, a, b, out, m, n, k, **kw):
    """fp32-accurate GEMM through 3xTF32: a / b are 2D f32 tensors stored as the
    name's layout expects ((M, K) / (K, M) for A, (N, K) / (K, N) for B)."""
    return _gemm(name, _tf32x3(a, True), _tf32x3(b, False), out, m, n, k, k_groups=3, **kw)


def _hnr(name, x, out, rows, outer, inner, eps=1e-6, weight=None, cos=None, sin=None, positions=None, inv_freq=None,
         bias=None, stride=None):
    """head_norm_rope (hnr_<F><W><C>_<dim>_<norm rope computed exact [half_table]>): rows of head_dim
    `stride` elements apart; cos/sin row = (row // outer) * inner + row % inner."""
    dummy = x
    dim = int(name.split('_')[2])
    _launch(name, [x, weight if weight is not None else dummy, bias if bias is not None else dummy,
                   cos if cos is not None else dummy,
                   sin if sin is not None else dummy, positions if positions is not None else dummy,
                   inv_freq if inv_freq is not None else dummy, out],
            ((rows + 7) // 8, 1, 1), {'I': [rows, outer, inner, stride or dim], 'f': [float(eps)]})
    return out


def _norm(name, x, width, eps, weight=None, bias=None, residual=None, out=None, out_dtype=None, residual_out=None):
    """norm_rows over rows of `width` (last dim)."""
    rows = x.numel() // width
    if out is None:
        out = torch.empty(x.shape, dtype=out_dtype or x.dtype, device=x.device)
    dummy = x
    _launch(name, [x, residual if residual is not None else dummy, weight if weight is not None else dummy,
                   bias if bias is not None else dummy, out, residual_out if residual_out is not None else out],
            (rows, 1, 1), {'I': [width], 'f': [float(eps)]})
    return out


def _colsum(name_part, name_fin, x, rows, cols, out):
    """Column sums of a (rows, cols) matrix into out (cols)."""
    chunk = 256
    chunks = (rows + chunk - 1) // chunk
    partial = torch.empty((chunks, cols), dtype=torch.float32, device=x.device)
    _launch(name_part, [x, partial], ((cols + 255) // 256, chunks, 1), {'I': [rows, cols, chunk]})
    _launch(name_fin, [partial, out], _blocks(cols, 256), {'I': [cols, chunks]})
    return out


def _flash(name, q, k, v, out, lse=None, mask=None, flags=None, scale=1.0, cap=1.0, window=0, mask_strides=(0, 0, 0)):
    """Fused attention, kernel name flash_<in><out>_<dim>_<bn>_<stages>_<bhld causal window softcap mask lse>.
    bhld=1: q (B, H, Lq, D), k/v (B, Hkv, Lk, D); bhld=0: q (B, Lq, H, D), k/v (B, Lk, Hkv, D)."""
    parts = name.split('_')
    kind, dim, bn, flags_s = parts[1][0], int(parts[2]), int(parts[3]), parts[5]
    bhld = flags_s[0] == '1'
    width = 128 // _KIND_BYTES[kind]
    q, k, v = _aligned(q), _aligned(k), _aligned(v)
    es = q.element_size()
    if bhld:
        batch, heads, lq, _ = q.shape
        kv_heads, lk = k.shape[1], k.shape[2]
        def tm(t, rows, atomic32=False):
            return _tmap(t, [dim, t.shape[2], t.shape[0] * t.shape[1]], [dim * es, t.shape[2] * dim * es], [width, rows, 1], atomic32=atomic32)
    else:
        batch, lq, heads, _ = q.shape
        lk, kv_heads = k.shape[1], k.shape[2]
        def tm(t, rows, atomic32=False):
            return _tmap(t, [dim, t.shape[2], t.shape[0] * t.shape[1]], [dim * es, t.shape[2] * dim * es], [width, 1, rows], atomic32=atomic32)
    maps = [tm(q, 128), tm(k, bn), tm(v, bn, atomic32=kind == 't')]
    dummy_f = lse if lse is not None else out
    _launch(name, [q, k, v, out, lse if lse is not None else torch.empty(1, dtype=torch.float32, device=q.device),
                   mask if mask is not None else dummy_f, flags if flags is not None else dummy_f],
            ((lq + 127) // 128, heads, batch),
            {'I': [lq, lk, heads, kv_heads, window, *mask_strides], 'f': [float(scale), float(cap)]}, maps)
    return out


def _moe_route(ids, experts):
    """Counting sort of (token, k) pairs by expert. ids (T, K) int64.
    Returns groups (row + M-tile prefixes), sorted token per row, inverse row per pair."""
    ids = _aligned(ids)
    pairs = ids.numel()
    dev = ids.device
    counts = torch.zeros(experts, dtype=torch.int32, device=dev)
    _launch('moe_count', [ids, counts], _blocks(pairs, 256), {'I': [pairs]})
    groups = torch.empty(2 * experts + 2, dtype=torch.int32, device=dev)
    cursor = torch.empty(experts, dtype=torch.int32, device=dev)
    _launch('moe_offsets', [counts, groups, cursor], (1, 1, 1), {'I': [experts]})
    sorted_tokens = torch.empty(pairs, dtype=torch.int32, device=dev)
    inverse = torch.empty(pairs, dtype=torch.int32, device=dev)
    _launch('moe_place', [ids, groups, cursor, sorted_tokens, inverse], _blocks(pairs, 256),
            {'I': [pairs, ids.shape[-1] if ids.dim() > 1 else 1]})
    return groups, sorted_tokens, inverse


def _moe_gather(kernel, x, sorted_tokens):
    rows, width = sorted_tokens.numel(), x.shape[-1]
    out = torch.empty((rows, width), dtype=x.dtype, device=x.device)
    _launch(kernel, [_aligned(x), sorted_tokens, out], _blocks(rows * width // 8, 256), {'I': [rows * width // 8, width // 8]})
    return out


def _moe_gemm(name, xs, weights, groups, n, k, residual=None, out=None, out_dtype=None, scale_a=None, scale_b=None, alpha=1.0):
    """Grouped GEMM over experts: rows of xs sorted by expert, weights (E, N, K)."""
    rows = xs.shape[0]
    experts = weights.shape[0]
    if out is None:
        out = torch.empty((rows, n), dtype=out_dtype or xs.dtype, device=xs.device)
    _gemm(name, _op(xs), _opb(_aligned(weights)), out, rows, n, k, batches=experts, groups=groups,
          max_tiles=(rows + 127) // 128 + experts, residual=residual, scale_a=scale_a, scale_b=scale_b, alpha=alpha)
    return out


def _moe_combine(kernel, y, inverse, weights, tokens, top_k, residual=None):
    width = y.shape[-1]
    out = torch.empty((tokens, width), dtype=y.dtype, device=y.device)
    count = tokens * width // 8
    _launch(kernel, [y, inverse, _aligned(weights), residual if residual is not None else y, out], _blocks(count, 256),
            {'I': [count, width // 8, top_k]})
    return out


def _attn_f32(q, k, v, b, sq, sk, heads, kv_heads, d, scale=1.0, causal=False, probs=None, mask=None,
              mask_outer=_BIG, mask_inner=1, out=None):
    """fp32-accurate attention through 3xTF32 GEMMs + f32 softmax.
    q (B*Sq, H*D), k / v (B*Sk, Hkv*D) row-major (BLHD); returns (B, Sq, H, D) f32.
    probs: optional (B, H, Sq, Sk) f32 output of the probabilities. mask: additive
    f32 rows of Sk, mask row = (row // mask_outer) * mask_inner + row % mask_inner."""
    dev = q.device
    group = heads // kv_heads
    ldp = (sk + 7) // 8 * 8
    per_batch = heads * sq * ldp * 4
    chunk = max(1, (1 << 31) // max(1, per_batch))
    if b > chunk:
        # bound the score buffer: process batches in chunks
        out = out if out is not None else torch.empty((b, sq, heads, d), dtype=torch.float32, device=dev)
        for start in range(0, b, chunk):
            end = min(b, start + chunk)
            chunk_mask = mask
            if mask is not None and mask_outer <= heads * sq:
                row_offset = (start * heads * sq // mask_outer) * mask_inner
                chunk_mask = mask.reshape(-1, sk)[row_offset:]
            _attn_f32(q[start * sq:end * sq], k[start * sk:end * sk], v[start * sk:end * sk], end - start, sq, sk,
                      heads, kv_heads, d, scale, causal, None if probs is None else probs[start:end], chunk_mask, mask_outer,
                      mask_inner, out=out[start:end])
        return out
    qs = _split3(q, heads * d, heads * d, True)
    ks = _split3(k, kv_heads * d, kv_heads * d, False)
    scores = torch.empty((b * heads, sq, ldp), dtype=torch.float32, device=dev)
    _gemm('gemm_tf_256_00_00000',
          _Op(qs, [d, 3 * heads, b * sq], [d * 4, 3 * heads * d * 4], z_div=heads, z_mul=sq, y_mod=heads, y_div=1, y_mul=1, y_kstep=heads),
          _Op(ks, [d, 3 * kv_heads, b * sk], [d * 4, 3 * kv_heads * d * 4], z_div=heads, z_mul=sk, y_mod=heads, y_div=group, y_mul=1,
              y_kstep=kv_heads),
          scores, sq, ldp, d, batches=b * heads, k_groups=3, ld=ldp, o_s1=sq * ldp, alpha=scale)
    if probs is None:
        probs = torch.empty((b, heads, sq, sk), dtype=torch.float32, device=dev)
    kernel = {(False, False): 'softmax_ff_00', (False, True): 'softmax_ff_01',
              (True, False): 'softmax_ff_10', (True, True): 'softmax_ff_11'}[(mask is not None, bool(causal))]
    _launch(kernel, [scores, mask if mask is not None else scores, scores, probs], (b * heads * sq, 1, 1),
            {'I': [sk, ldp, sk, mask_outer, mask_inner, sq, sk - sq], 'f': [1.0]})
    ld4 = (sk + 3) // 4 * 4
    ps = _split3(probs, sk, ld4, True)
    vs = _split3(v, kv_heads * d, kv_heads * d, False)
    attn = out if out is not None else torch.empty((b, sq, heads, d), dtype=torch.float32, device=dev)
    _gemm('gemm_tf_64_01_00000' if d <= 64 else 'gemm_tf_128_01_00000',
          _Op(ps, [ld4, 3, b * heads * sq], [ld4 * 4, 3 * ld4 * 4], z_div=1, z_mul=sq, y_kstep=1),
          _Op(vs, [d, 3 * kv_heads, b * sk], [d * 4, 3 * kv_heads * d * 4], z_div=heads, z_mul=sk, y_mod=heads, y_div=group, y_mul=1,
              y_kstep=kv_heads),
          attn, sq, d, sk, batches=b * heads, k_groups=3, ld=heads * d, o_div=heads, o_s1=sq * heads * d, o_s2=d)
    return attn


def _im2col(kernel, x, kh, kw, stride=1, pad=0, dil=1, kp=None):
    """NCHW x -> cols (N, OH*OW, kp) (k = (c*KH + kh)*KW + kw, zero padded to kp)."""
    n, c, h, w = x.shape
    oh = (h + 2 * pad - dil * (kh - 1) - 1) // stride + 1
    ow = (w + 2 * pad - dil * (kw - 1) - 1) // stride + 1
    k = c * kh * kw
    kp = kp or (k + 7) // 8 * 8
    cols = torch.empty((n, oh * ow, kp), dtype=x.dtype, device=x.device)
    count = cols.numel()
    _launch(kernel, [_aligned(x), cols], _blocks(count, 256),
            {'I': [count, c, h, w, kh, kw, oh, ow, stride, stride, pad, pad, dil, dil, kp]})
    return cols, oh, ow


def _fq_act(x, scales=None):
    """FP8 1x128 fake-quant of a 2D activation (bf16 or f32) -> bf16 (scales computed unless given)."""
    x = _aligned(x)
    rows, cols = x.shape
    given = scales is not None
    if scales is None:
        scales = torch.empty((rows, cols // 128), dtype=torch.float32, device=x.device)
    out = torch.empty((rows, cols), dtype=torch.bfloat16, device=x.device)
    items = rows * (cols // 128)
    kernel = {(torch.bfloat16, False): 'fq_act_bb_0', (torch.bfloat16, True): 'fq_act_bb_1',
              (torch.float32, False): 'fq_act_fb_0', (torch.float32, True): 'fq_act_fb_1'}[(x.dtype, given)]
    _launch(kernel, [x, _aligned(scales), out], ((items + 7) // 8, 1, 1), {'I': [rows, cols]})
    return out


def _fq_weight(w, scales=None, s_n=None, s_k=None):
    """FP8 128x128 fake-quant of a bf16 weight (N, K) -> bf16; scale of tile (nb, kb) at nb * s_n + kb * s_k."""
    w = _aligned(w)
    n, k = w.shape
    if s_n is None:
        s_n, s_k = k // 128, 1
    given = scales is not None
    if scales is None:
        scales = torch.empty(((n + 127) // 128) * (k // 128), dtype=torch.float32, device=w.device)
    out = torch.empty((n, k), dtype=torch.bfloat16, device=w.device)
    _launch('fq_weight_bb_1' if given else 'fq_weight_bb_0', [w, _aligned(scales), out], (k // 128, (n + 127) // 128, 1),
            {'I': [k, s_n, s_k]})
    return out


def _dq_fp8(x, scales, block_rows, s_r, s_c):
    """E4M3 tensor (2D) with block scales -> bf16."""
    x = _aligned(x)
    rows, cols = x.shape
    out = torch.empty((rows, cols), dtype=torch.bfloat16, device=x.device)
    _launch('dq_fp8_b', [x.view(torch.uint8), _aligned(scales), out], _blocks(x.numel(), 256),
            {'I': [x.numel(), cols, block_rows, s_r, s_c]})
    return out


def _decoder_rest(x, attn, o_proj_weight, post_weight, gate_w, up_w, down_w, eps, norm_name):
    """o_proj + residual, post RMSNorm, SwiGLU MLP + residual (bf16 reference rounding).
    norm_name: 'norm_bbb_001001' (HF weight * x.to(dtype)) or 'norm_bbb_001000' ((weight * x).to(dtype))."""
    b, s, hid = x.shape
    h3 = _linear('gemm_bb_256_00_00101', attn, o_proj_weight, residual=x.reshape(b * s, hid))
    names = {'norm_bbb_001001': 'norm_bbb_001001', 'norm_bbb_001000': 'norm_bbb_001000'}
    h4 = _norm(names[norm_name], h3, hid, eps, weight=_aligned(post_weight))
    gate = _linear('gemm_bb_256_00_00001', h4, gate_w)
    act = _linear('gemm_bb_256_00_00201', h4, up_w, residual=gate)
    return _linear('gemm_bb_256_00_00101', act, down_w, residual=h3).view(b, s, hid)


def _binary(name, a, b, alpha=1.0, out_dtype=None):
    shape = tuple(torch.broadcast_shapes(a.shape, b.shape))
    if len(shape) > 4:
        return _binary(name, a.expand(shape).contiguous().view(-1), b.expand(shape).contiguous().view(-1), alpha, out_dtype).view(shape)
    dims = (1,) * (4 - len(shape)) + shape
    def strides(t):
        sizes = (1,) * (4 - t.dim()) + tuple(t.shape)
        raw = (0,) * (4 - t.dim()) + t.stride()
        return [s if n != 1 else 0 for n, s in zip(sizes, raw)]
    out = torch.empty(shape, dtype=out_dtype or a.dtype, device=a.device)
    _launch(name, [a, b, out], _blocks(out.numel(), 256),
            {'I': list(dims[1:]) + [out.numel()] + strides(a) + strides(b), 'f': [float(alpha)]})
    return out


def _cast(name, x, dtype):
    x = _aligned(x)
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    _launch(name, [x, out], _blocks(x.numel() // 8, 256), {'I': [x.numel() // 8]})
    return out


def _nv4_quant(x, round_block=False):
    x = _aligned(x)
    workers = max(1, min((x.numel() + 2047) // 2048, _sm_count(x.device) * 4))
    partial = torch.empty(workers, dtype=torch.float32, device=x.device)
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    global_scale = torch.empty((), dtype=torch.float32, device=x.device)
    maximum = {torch.bfloat16: 'nv4_amax_b', torch.float32: 'nv4_amax_f'}[x.dtype]
    encode = {(torch.bfloat16, False): 'nv4_encode_b0', (torch.bfloat16, True): 'nv4_encode_b1',
              (torch.float32, False): 'nv4_encode_f0', (torch.float32, True): 'nv4_encode_f0'}[(x.dtype, round_block)]
    _launch(maximum, [x, partial], (workers, 1, 1))
    _launch(encode, [x, partial, out, global_scale], _blocks(x.numel() // 16, 128))
    return out, global_scale


def _nv4_decode(x, scales):
    x, scales = _aligned(x), _aligned(scales)
    out = torch.empty((*x.shape[:-1], x.shape[-1] * 2), dtype=torch.bfloat16, device=x.device)
    _launch('nv4_decode', [x.view(torch.uint8), scales.view(torch.uint8), out], _blocks(out.numel() // 8, 256))
    return out


def _nv4_scale(value, like):
    if isinstance(value, torch.Tensor):
        return _aligned(value)
    return torch.tensor(float(value), dtype=torch.float32, device=like.device)


def _nv4_project(activation, weight, scales, global_scale, dtype=torch.bfloat16):
    x, gx = activation
    k = x.shape[-1]
    w = _nv4_decode(weight, scales).reshape(-1, k)
    m, n = x.numel() // k, w.shape[0]
    out = torch.empty((m, n), dtype=dtype, device=x.device)
    name = {torch.bfloat16: 'gemm_bb_256_00_00040', torch.float32: 'gemm_bf_256_00_00040'}[dtype]
    _gemm(name, _op(x.reshape(m, k)), _op(w), out, m, n, k, scale_a=gx, scale_b=_nv4_scale(global_scale, x))
    return out.view(*x.shape[:-1], n)


def _nv4_linear(x, weight, scales, global_scale, dtype=torch.bfloat16, round_block=False):
    return _nv4_project(_nv4_quant(x, round_block), weight, scales, global_scale, dtype)


def _nv4_bmm(a, b, dtype=torch.bfloat16, group=1):
    x, gx = a
    w, gw = b
    batches, m, k = x.shape
    n = w.shape[1]
    out = torch.empty((batches, m, n), dtype=dtype, device=x.device)
    name = {torch.bfloat16: 'gemm_bb_128_00_00040', torch.float32: 'gemm_bf_128_00_00040'}[dtype]
    _gemm(name, _opb(x), _opb(w, div=group), out, m, n, k, batches=batches, ld=n, o_s1=m * n, scale_a=gx, scale_b=gw)
    return out


def _score_softmax_bf16(scores, batch, heads, queries, columns, factor, mask=None, causal=False):
    ld = scores.shape[-1]
    name = {(False, False): 'bf16_scores_00', (False, True): 'bf16_scores_01',
            (True, False): 'bf16_scores_10', (True, True): 'bf16_scores_11'}[(mask is not None, causal)]
    adjusted = torch.empty_like(scores)
    _launch(name, [scores, _aligned(mask) if mask is not None else scores, adjusted],
            _blocks(batch * heads * queries * columns, 256),
            {'I': [batch * heads * queries * columns, columns, ld, queries, heads], 'f': [float(factor)]})
    probs = torch.empty_like(scores)
    _launch('softmax_bb_00', [adjusted, adjusted, adjusted, probs], (batch * heads * queries, 1, 1),
            {'I': [columns, ld, ld, 1, 1, queries, columns - queries], 'f': [1.0]})
    return probs[..., :columns]


def _attn_bf16(q, k, v, mask=None, causal=False, scale=None):
    b, sq, heads, d = q.shape
    sk, kv = k.shape[1:3]
    ld = (sk + 7) // 8 * 8
    scores = torch.empty((b * heads, sq, ld), dtype=torch.bfloat16, device=q.device)
    _gemm('gemm_bb_128_00_00000', _opl(q, heads), _opl(k, heads, div=heads // kv), scores,
          sq, ld, d, batches=b * heads, ld=ld, o_s1=sq * ld)
    p = _score_softmax_bf16(scores, b, heads, sq, sk, d ** -0.5 if scale is None else scale, mask, causal)
    out = torch.empty((b, sq, heads, d), dtype=torch.bfloat16, device=q.device)
    _gemm('gemm_bb_128_01_00000', _opb(p), _opl(v, heads, div=heads // kv), out, sq, d, sk,
          batches=b * heads, ld=heads * d, o_div=heads, o_s1=sq * heads * d, o_s2=d)
    return out, p.view(b, heads, sq, sk)


def _nv4_qkv(x, projections, cos, sin, q_norm=None, k_norm=None, eps=1e-6, hf_norm=False):
    b, s, _ = x.shape
    activation = _nv4_quant(x)
    q, k, v = [_nv4_project(activation, w, scales, global_scale) for w, scales, global_scale in projections]
    q, k, v = [t.view(b, s, -1, 128) for t in (q, k, v)]
    if q_norm is not None:
        name = 'norm_bbb_001001' if hf_norm else 'norm_bbb_001000'
        q = _norm(name, q, 128, eps, weight=_aligned(q_norm))
        k = _norm(name, k, 128, eps, weight=_aligned(k_norm))
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    _hnr('hnr_bbb_128_0101', q, qr, b * s * q.shape[2], q.shape[2], 1, cos=_aligned(cos), sin=_aligned(sin))
    _hnr('hnr_bbb_128_0101', k, kr, b * s * k.shape[2], k.shape[2], 1, cos=_aligned(cos), sin=_aligned(sin))
    return qr, kr, v


def _nv4_mlp(x, gate, up, down, fp32=False):
    dtype = torch.float32 if fp32 else torch.bfloat16
    activation = _nv4_quant(x)
    g = _nv4_project(activation, *gate, dtype=dtype)
    u = _nv4_project(activation, *up, dtype=dtype)
    silu = torch.empty_like(g)
    _launch('un_ff_0' if fp32 else 'un_bb_0', [g, silu], _blocks(g.numel(), 256), {'I': [g.numel()]})
    h = _binary('bin_fff_2' if fp32 else 'bin_bbb_2', silu, u)
    out = _nv4_linear(h, *down, dtype=dtype)
    return _cast('cast_fb', out, torch.bfloat16) if fp32 else out


def _linear_f32(x, weight, bias=None):
    m, k, n = x.numel() // x.shape[-1], x.shape[-1], weight.shape[0]
    np, kp = (n + 7) // 8 * 8, (k + 7) // 8 * 8
    a, w = x.reshape(m, k), weight
    if kp != k:
        a = _matrix_pad(a, m, kp)
    if kp != k or np != n:
        w = _matrix_pad(w, np, kp)
    if bias is not None and np != n:
        bias = _matrix_pad(bias.view(1, n), 1, np).view(np)
    out = torch.empty((*x.shape[:-1], np), dtype=torch.float32, device=x.device)
    _gemm3('gemm_tf_256_00_10000' if bias is not None else 'gemm_tf_256_00_00000',
           a, w, out, m, np, kp, bias=None if bias is None else _aligned(bias))
    return out[..., :n]


def _conv2d_f32(x, weight, bias=None, stride=1, pad=0):
    b, channels, h, w = x.shape
    oc, _, kh, kw = weight.shape
    oh, ow = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    p, k = oh * ow, channels * kh * kw
    kp, pp = (k + 7) // 8 * 8, (p + 7) // 8 * 8
    weights = _aligned(weight).view(oc, k)
    if k != kp:
        padded = torch.empty((oc, kp), dtype=torch.float32, device=x.device)
        _launch('pad_rows_f32', [weights, padded], _blocks(padded.numel(), 256), {'I': [padded.numel(), k, kp]})
        weights = padded
    a = _tf32x3(weights, True)
    out = torch.empty((b, oc, oh, ow), dtype=torch.float32, device=x.device)
    for batch in range(b):
        cols, _, _ = _im2col('im2col_f', x[batch:batch + 1], kh, kw, stride, pad, kp=kp)
        result = torch.empty((oc, pp), dtype=torch.float32, device=x.device)
        _gemm('gemm_tf_256_00_20000' if bias is not None else 'gemm_tf_256_00_00000', a,
              _tf32x3(cols.view(p, kp), False), result, oc, pp, kp, k_groups=3, bias=None if bias is None else _aligned(bias))
        _launch('permute_f', [result, out[batch]], _blocks(oc * p, 256), {'I': [1, oc, p, oc * p, 0, 0, pp, 1]})
    return out


def _group_norm_f32(x, weight, bias, groups, eps):
    b, c = x.shape[:2]
    spatial = x.numel() // (b * c)
    chunk = 16384
    chunks = (spatial + chunk - 1) // chunk
    partial = torch.empty(b * c * chunks * 2, dtype=torch.float64, device=x.device)
    out = torch.empty_like(x)
    xc = _aligned(x)
    _launch('l1_078_gn_partial', [xc, partial], (chunks, c, b), {'I': [spatial, chunk]})
    _launch('l1_078_gn_apply', [xc, partial, _aligned(weight), _aligned(bias), out], (chunks, c, b),
            {'I': [spatial, chunk, groups], 'f': [float(eps)]})
    return out


def _unary(name, x):
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _launch(name, [_aligned(x), out], _blocks(x.numel(), 256), {'I': [x.numel()]})
    return out


def _varlen_f32(x, cu, cos, sin, qkv_weight, qkv_bias, proj_weight, proj_bias, heads, dim):
    total = x.shape[0]
    qkv = _linear_f32(x, qkv_weight, qkv_bias).view(total, 3, heads, dim)
    q, k, v = [_aligned(qkv[:, i]) for i in range(3)]
    qr, kr = torch.empty_like(q), torch.empty_like(k)
    name = {64: 'hnr_fff_64_0100', 80: 'hnr_fff_80_0100', 128: 'hnr_fff_128_0100'}[dim]
    _hnr(name, q, qr, total * heads, heads, 1, cos=_aligned(cos), sin=_aligned(sin))
    _hnr(name, k, kr, total * heads, heads, 1, cos=_aligned(cos), sin=_aligned(sin))
    out = torch.empty_like(q)
    name = {64: 'varlen_f32_64', 80: 'varlen_f32_80', 128: 'varlen_f32_128'}[dim]
    _launch(name, [qr, kr, v, _aligned(cu), out], (total, heads, 1), {'I': [heads], 'f': [dim ** -0.5]})
    return _linear_f32(out.view(total, heads * dim), proj_weight, proj_bias)


def _fft_power2(x, inverse=False):
    n = x.shape[-2]
    log = n.bit_length() - 1
    n1, n2 = 1 << (log // 2), 1 << (log - log // 2)
    first = {1: 'fft_first_1', 2: 'fft_first_2', 4: 'fft_first_4', 8: 'fft_first_8', 16: 'fft_first_16',
             32: 'fft_first_32', 64: 'fft_first_64', 128: 'fft_first_128', 256: 'fft_first_256',
             512: 'fft_first_512', 1024: 'fft_first_1024'}
    last = {1: 'fft_last_1', 2: 'fft_last_2', 4: 'fft_last_4', 8: 'fft_last_8', 16: 'fft_last_16',
            32: 'fft_last_32', 64: 'fft_last_64', 128: 'fft_last_128', 256: 'fft_last_256',
            512: 'fft_last_512', 1024: 'fft_last_1024'}
    rows = x.numel() // (n * 2)
    scratch, out = torch.empty_like(x), torch.empty_like(x)
    scalars = {'I': [n1, n2], 'f': [1.0 if inverse else -1.0, 1.0 / n if inverse else 1.0]}
    _launch(first[n1], [x, scratch], (n2, rows, 1), scalars)
    _launch(last[n2], [scratch, out], (n1, rows, 1), scalars)
    return out


def _fft_complex(x, inverse=False):
    n = x.shape[-2]
    if n & (n - 1) == 0:
        return _fft_power2(x, inverse)
    padded = 1 << (2 * n - 2).bit_length()
    a = torch.empty((*x.shape[:-2], padded, 2), dtype=torch.float32, device=x.device)
    b = torch.empty((1, padded, 2), dtype=torch.float32, device=x.device)
    sign = 1.0 if inverse else -1.0
    _launch('fft_chirp_input', [x, a, b], _blocks(a.numel() // 2, 256), {'I': [n, padded], 'f': [sign]})
    af, bf = _fft_power2(a), _fft_power2(b)
    _launch('fft_multiply', [af, bf, af], _blocks(af.numel() // 2, 256), {'I': [padded]})
    conv = _fft_power2(af, True)
    out = torch.empty_like(x)
    _launch('fft_chirp_finish', [conv, out], _blocks(out.numel() // 2, 256),
            {'I': [n, padded], 'f': [sign, 1.0 / n if inverse else 1.0]})
    return out


def _fft_real(x, length):
    x = _aligned(x)
    padded = torch.empty((*x.shape[:-1], length, 2), dtype=torch.float32, device=x.device)
    _launch('fft_real_pad', [x, padded], _blocks(padded.numel() // 2, 256), {'I': [x.shape[-1], length]})
    return _fft_complex(padded)


def _matmul_f32(a, b, transpose_a=False, transpose_b=False, combine=None):
    m = a.shape[1] if transpose_a else a.shape[0]
    k = a.shape[0] if transpose_a else a.shape[1]
    n = b.shape[0] if transpose_b else b.shape[1]
    names = {(False, False): 'gemm_tf_256_01_00000', (False, True): 'gemm_tf_256_00_00000',
             (True, False): 'gemm_tf_256_11_00000', (True, True): 'gemm_tf_256_10_00000'}
    name = names[(transpose_a, transpose_b)]
    if combine is not None:
        assert not transpose_a and not transpose_b
        name = 'gemm_tf_256_01_00600'
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    _gemm3(name, a, b, out, m, n, k, residual=None if combine is None else _aligned(combine))
    return out


def _sum_columns_f32(x):
    x = _aligned(x).reshape(-1, x.shape[-1])
    out = torch.empty(x.shape[1], dtype=torch.float32, device=x.device)
    return _colsum('colpart_f', 'colfin_f', x, x.shape[0], x.shape[1], out)


def _attention_f32(q, k, v, heads, kv_heads=None, mask=None, causal=False, scale=None):
    b, sq, _ = q.shape
    sk = k.shape[1]
    d = q.shape[-1] // heads
    return _attn_f32(q.reshape(b * sq, -1), k.reshape(b * sk, -1), v.reshape(b * sk, -1),
                     b, sq, sk, heads, kv_heads or heads, d, scale=d ** -0.5 if scale is None else scale,
                     causal=causal, mask=None if mask is None else _aligned(mask), mask_outer=heads * sq,
                     mask_inner=sq).reshape(b, sq, heads * d)


def _rope_pair_bf16(q, k, cos, sin, dim=128):
    b, sq, qh, _ = q.shape
    kh = k.shape[2]
    qo, ko = torch.empty_like(q), torch.empty_like(k)
    name = {72: 'hnr_bbb_72_0101', 128: 'hnr_bbb_128_0101'}[dim]
    _hnr(name, _aligned(q), qo, b * sq * qh, qh, 1, cos=_aligned(cos), sin=_aligned(sin))
    _hnr(name, _aligned(k), ko, b * sq * kh, kh, 1, cos=_aligned(cos), sin=_aligned(sin))
    return qo, ko


def _matrix_pad(x, rows, columns):
    out = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
    _launch('matrix_pad_f' if x.dtype == torch.float32 else 'matrix_pad_b', [_aligned(x), out],
            _blocks(out.numel(), 256), {'I': [x.shape[0], x.shape[1], columns]})
    return out


def _conv_nd(x, weight, bias=None, stride=1, pad=0, transpose=False):
    rank = x.ndim - 2
    strides = (stride,) * rank if isinstance(stride, int) else tuple(stride)
    pads = (pad,) * rank if isinstance(pad, int) else tuple(pad)
    shape, kernel = tuple(x.shape[2:]), tuple(weight.shape[2:])
    output_shape = tuple((a - 1) * s - 2 * p + k if transpose else (a + 2 * p - k) // s + 1
                         for a, s, p, k in zip(shape, strides, pads, kernel))
    b, ic = x.shape[:2]
    if transpose:
        weight = weight.transpose(0, 1).contiguous()
    oc = weight.shape[0]
    inner = ic * math.prod(kernel)
    kp, pos = (inner + 7) // 8 * 8, math.prod(output_shape)
    pp = (pos + 7) // 8 * 8
    w = _aligned(weight).view(oc, inner)
    if inner != kp:
        w = _matrix_pad(w, oc, kp)
    fp32 = x.dtype == torch.float32
    wa = _tf32x3(w, True) if fp32 else _op(w)
    out = torch.empty((b, oc, *output_shape), dtype=x.dtype, device=x.device)
    sd, kd, od = (1,) * (3 - rank) + shape, (1,) * (3 - rank) + kernel, (1,) * (3 - rank) + output_shape
    st, pd = (1,) * (3 - rank) + strides, (0,) * (3 - rank) + pads
    for bi in range(b):
        cols = torch.empty((pos, kp), dtype=x.dtype, device=x.device)
        name = ('spatial_columns_f_1' if transpose else 'spatial_columns_f_0') if fp32 else ('spatial_columns_b_1' if transpose else 'spatial_columns_b_0')
        _launch(name, [_aligned(x[bi]), cols], _blocks(cols.numel(), 256), {'I': [ic, *sd, *od, *kd, *st, *pd, kp]})
        result = torch.empty((oc, pp), dtype=x.dtype, device=x.device)
        name = ('gemm_tf_256_00_20000' if bias is not None else 'gemm_tf_256_00_00000') if fp32 else ('gemm_bb_256_00_20000' if bias is not None else 'gemm_bb_256_00_00000')
        _gemm(name, wa, _tf32x3(cols, False) if fp32 else _op(cols), result, oc, pp, kp,
              k_groups=3 if fp32 else 1, bias=None if bias is None else _aligned(bias))
        _launch('permute_f' if fp32 else 'permute_b', [result, out[bi]], _blocks(oc * pos, 256),
                {'I': [1, oc, pos, oc * pos, 0, 0, pp, 1]})
    return out


def _layernorm_f32(x, weight, bias, eps=1e-5):
    return _norm('norm_fff_101100', _aligned(x), x.shape[-1], eps, weight=_aligned(weight), bias=_aligned(bias))


def _resnet2d(x, temb, n1w, n1b, c1w, c1b, tw, tb, n2w, n2b, c2w, c2b, eps):
    h = _unary('un_ff_0', _group_norm_f32(x, n1w, n1b, 32, eps))
    h = _conv2d_f32(h, c1w, c1b, pad=1)
    t = _linear_f32(_unary('un_ff_0', temb), tw, tb)
    h = _binary('bin_fff_0', h, t[:, :, None, None])
    h = _unary('un_ff_0', _group_norm_f32(h, n2w, n2b, 32, eps))
    return _binary('bin_fff_0', x, _conv2d_f32(h, c2w, c2b, pad=1))


def _affine(x, scale=1.0, bias=0.0, dtype=None):
    dtype = dtype or x.dtype
    name = { (torch.float32, torch.float32): 'affine_ff', (torch.bfloat16, torch.float32): 'affine_bf',
             (torch.float32, torch.bfloat16): 'affine_fb', (torch.bfloat16, torch.bfloat16): 'affine_bb'}[(x.dtype, dtype)]
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    _launch(name, [_aligned(x), out], _blocks(out.numel(), 256), {'f': [float(scale), float(bias)]})
    return out


def _f32(x):
    return x if x.dtype == torch.float32 else _affine(x, dtype=torch.float32)


def _sum(x, dims=None, keepdim=False, mean=False, dtype=None):
    dims = tuple(range(x.ndim)) if dims is None else ((dims,) if isinstance(dims, int) else tuple(dims))
    dims = tuple(d % x.ndim for d in dims)
    kept = tuple(i for i in range(x.ndim) if i not in dims)
    shape = tuple(x.shape[i] for i in kept)
    width = math.prod(x.shape[i] for i in dims)
    flat = x.permute(kept + dims).contiguous().view(-1, width)
    dtype = dtype or x.dtype
    out = torch.empty(shape, dtype=dtype, device=x.device)
    name = {(torch.float32, torch.float32): 'reduce_last_ff', (torch.bfloat16, torch.float32): 'reduce_last_bf',
            (torch.float32, torch.bfloat16): 'reduce_last_fb', (torch.bfloat16, torch.bfloat16): 'reduce_last_bb'}[(x.dtype, dtype)]
    _launch(name, [flat, out], (out.numel(), 1, 1), {'I': [width], 'f': [1.0 / width if mean else 1.0]})
    return out.view(tuple(1 if i in dims else x.shape[i] for i in range(x.ndim))) if keepdim else out


def _add(a, b):
    return _binary('bin_fff_0' if a.dtype == torch.float32 else 'bin_bbb_0', a, b)


def _sub(a, b):
    return _binary('bin_fff_1' if a.dtype == torch.float32 else 'bin_bbb_1', a, b)


def _mul(a, b):
    return _binary('bin_fff_2' if a.dtype == torch.float32 else 'bin_bbb_2', a, b)


def _div(a, b):
    return _binary('bin_fff_3' if a.dtype == torch.float32 else 'bin_bbb_3', a, b)


def _matmul(a, b, transpose_a=False, transpose_b=False):
    if transpose_a:
        a = a.transpose(-2, -1)
    if not transpose_b:
        b = b.transpose(-2, -1)
    batch = tuple(torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-2]
    batches, kp, np = math.prod(batch), (k + 7) // 8 * 8, (n + 7) // 8 * 8
    a = a.expand((*batch, m, k)).contiguous().view(batches * m, k)
    b = b.expand((*batch, n, k)).contiguous().view(batches, n, k)
    if kp != k:
        a = _matrix_pad(a, batches * m, kp)
    if kp != k or np != n:
        padded = torch.empty((batches, np, kp), dtype=b.dtype, device=b.device)
        for i in range(batches):
            padded[i].copy_(_matrix_pad(b[i], np, kp))
        b = padded
    a, b = a.view(batches, m, kp), b.view(batches, np, kp)
    out = torch.empty((batches, m, np), dtype=a.dtype, device=a.device)
    fp32 = a.dtype == torch.float32
    if fp32:
        aa = _split3(a, a.numel(), a.numel(), True).view(3 * batches, m, kp)
        bb = _split3(b, b.numel(), b.numel(), False).view(3 * batches, np, kp)
        ao, bo = _opb(aa, kstep=batches), _opb(bb, kstep=batches)
    else:
        ao, bo = _opb(a), _opb(b)
    _gemm('gemm_tf_256_00_00000' if fp32 else 'gemm_bb_256_00_00000', ao, bo, out, m, np, kp,
          batches=batches, k_groups=3 if fp32 else 1, o_s1=m * np)
    return out[..., :n].view(*batch, m, n)


def _linear_backward(g, x, weight):
    gf, xf = g.reshape(-1, g.shape[-1]), x.reshape(-1, x.shape[-1])
    return _matmul(g, weight), _matmul(gf, xf, transpose_a=True), _sum(gf, 0)


def _rope_backward(g, cos, sin, interleaved=False, wide=False):
    dtype = g.dtype
    if wide:
        g = _f32(g)
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    gs = _mul(g, s)
    if interleaved:
        rot = torch.stack((gs[..., 1::2], _affine(gs[..., 0::2], -1.0)), dim=-1).flatten(-2)
    else:
        lo, hi = gs.chunk(2, dim=-1)
        rot = torch.cat((hi, _affine(lo, -1.0)), dim=-1)
    out = _add(_mul(g, c), rot)
    return _affine(out, dtype=dtype) if out.dtype != dtype else out


def _softmax_backward(g, p, dtype=None, scale=1.0, cast_before_scale=False):
    pp, gg = _f32(p), _f32(g)
    d = _mul(pp, _sub(gg, _sum(_mul(gg, pp), -1, True)))
    dtype = dtype or g.dtype
    if cast_before_scale:
        return _affine(_affine(d, dtype=dtype), scale)
    return _affine(d, scale, dtype=dtype)


def _attention_backward(g, q, k, v, p, scale, kv_heads=None, cast_before_scale=False):
    gp, gv = _matmul(g, v, transpose_b=True), _matmul(p, g, transpose_a=True)
    ds = _softmax_backward(gp, p, q.dtype, scale, cast_before_scale)
    gq, gk = _matmul(ds, k), _matmul(ds, q, transpose_a=True)
    if kv_heads is not None and kv_heads != q.shape[1]:
        b, heads, sk, d = gk.shape
        gk = _sum(gk.reshape(b, kv_heads, heads // kv_heads, sk, d), 2)
        gv = _sum(gv.reshape(b, kv_heads, heads // kv_heads, sk, d), 2)
    return gq, gk, gv


def _rms_backward(g, x, variance, normalized, weight, eps):
    gf, xf = _f32(g), _f32(x)
    gw = _sum(_mul(gf, normalized), tuple(range(g.ndim - 1)))
    gn = _mul(gf, _f32(weight))
    r = _unary('un_ff_11', _affine(variance, bias=eps))
    dv = _mul(_affine(_sum(_mul(gn, xf), -1, True), -0.5), _mul(_mul(r, r), r))
    dx = _add(_mul(gn, r), _mul(_affine(xf, 2.0 / x.shape[-1]), dv))
    return _affine(dx, dtype=x.dtype), gw


def _softmax_raw(x, causal=False, dtype=None):
    source = _aligned(_f32(x))
    cols, queries = source.shape[-1], source.shape[-2]
    out = torch.empty_like(source)
    _launch('softmax_ff_01' if causal else 'softmax_ff_00', [source, source, source, out],
            (source.numel() // cols, 1, 1), {'I': [cols, cols, cols, _BIG, 1, queries, cols - queries], 'f': [1.0]})
    return _affine(out, dtype=dtype) if dtype is not None and dtype != torch.float32 else out


def _merge_heads(x):
    return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)


def _repeat_heads(x, heads):
    b, kv, seq, dim = x.shape
    return x[:, :, None].expand(b, kv, heads // kv, seq, dim).reshape(b, heads, seq, dim)


def _qkv_backward(gq, gk, gv, x, qw, kw, vw):
    grads = [_linear_backward(_merge_heads(g), x, w) for g, w in ((gq, qw), (gk, kw), (gv, vw))]
    return (_add(_add(grads[0][0], grads[1][0]), grads[2][0]), grads[0][1], grads[1][1], grads[2][1])


def _rope_forward(x, cos, sin, interleaved=False, wide=False):
    dtype = x.dtype
    if wide:
        x = _f32(x)
    if interleaved:
        rotated = torch.stack((_affine(x[..., 1::2], -1.0), x[..., 0::2]), dim=-1).flatten(-2)
    else:
        a, b = x.chunk(2, dim=-1)
        rotated = torch.cat((_affine(b, -1.0), a), dim=-1)
    y = _add(_mul(x, cos.unsqueeze(1)), _mul(rotated, sin.unsqueeze(1)))
    return _affine(y, dtype=dtype) if y.dtype != dtype else y


def _hyena_convolve(v, kernel, bias):
    seq, channels = v.shape[-1], v.shape[-2]
    vf, kf = _fft_real(v, 2 * seq), _fft_real(kernel, 2 * seq)
    kf = _affine(kf, 1.0 / (2 * seq))
    product = torch.empty_like(vf)
    _launch('fft_convolution_product', [vf, kf, product], _blocks(product.numel() // 2, 256), {'I': [channels, 2 * seq]})
    transformed = _fft_complex(product, True)
    y = _affine(transformed[..., :seq, 0], 2 * seq)
    return _add(y, _mul(v, bias.view(1, channels, 1)))


def _segment_exp(x):
    x = _aligned(x)
    prefix = torch.empty_like(x)
    out = torch.empty((*x.shape, x.shape[-1]), dtype=torch.float32, device=x.device)
    _launch('segment_exp_f32', [x, prefix, out], _blocks(x.numel(), 256), {'I': [x.shape[-1]]})
    return prefix, out


def _pad_axis(x, axis, size):
    if x.shape[axis] == size:
        return x
    shape = list(x.shape)
    shape[axis] = size - shape[axis]
    return torch.cat((x, torch.zeros(shape, dtype=x.dtype, device=x.device)), dim=axis)


def _ssm_chunks(x, a, b, c, initial, chunk):
    batch, length, heads, dim = x.shape
    padded = (length + chunk - 1) // chunk * chunk
    chunks, state_dim = padded // chunk, b.shape[-1]
    xc = _pad_axis(x, 1, padded).view(batch, chunks, chunk, heads, dim).permute(0, 1, 3, 2, 4)
    bc = _pad_axis(b, 1, padded).view(batch, chunks, chunk, heads, state_dim).permute(0, 1, 3, 2, 4)
    cc = _pad_axis(c, 1, padded).view(batch, chunks, chunk, heads, state_dim).permute(0, 1, 3, 2, 4)
    ac = _pad_axis(a, 1, padded).view(batch, chunks, chunk, heads).permute(0, 3, 1, 2)
    prefix, lower = _segment_exp(ac)
    gram = _matmul(cc, bc, transpose_b=True)
    m = _mul(gram, lower.permute(0, 2, 1, 3, 4))
    diag = _matmul(m, xc)
    decay = _unary('un_ff_6', _sub(prefix[..., -1:], prefix)).permute(0, 2, 1, 3)
    states = _matmul(xc, _mul(bc, decay.unsqueeze(-1)), transpose_a=True)
    prev = torch.cat((initial[:, None], states), dim=1).permute(0, 2, 1, 3, 4)
    ends = torch.cat((torch.zeros((batch, heads, 1), dtype=a.dtype, device=a.device), prefix[..., -1]), dim=-1)
    _, decay_chunk = _segment_exp(ends)
    combined = _matmul(decay_chunk, prev.reshape(batch, heads, chunks + 1, dim * state_dim)).view(batch, heads, chunks + 1, dim, state_dim)
    state = combined[:, :, :-1].permute(0, 2, 1, 3, 4)
    off = _mul(_matmul(cc, state, transpose_b=True), _unary('un_ff_6', prefix).permute(0, 2, 1, 3).unsqueeze(-1))
    y = _add(diag, off).permute(0, 1, 3, 2, 4).reshape(batch, padded, heads, dim)[:, :length]
    return y, combined[:, :, -1]


def _relative_table(table, size):
    out = torch.empty((size, size, table.shape[1]), dtype=torch.float32, device=table.device)
    _launch('relative_table', [_aligned(table), out], _blocks(out.numel(), 256), {'I': [size, table.shape[1]]})
    return out


def _relative_table_backward(grad, table):
    out = torch.empty_like(table)
    _launch('relative_table_grad', [_aligned(grad), out], _blocks(out.numel(), 256), {'I': [grad.shape[0], table.shape[1]]})
    return out


def _relative2d_bias(q, rh, rw, height, width):
    q2 = q.reshape(-1, height, width, q.shape[-1])
    h = _matmul(q2, rh.unsqueeze(0), transpose_b=True)
    w = _matmul(q2.transpose(1, 2), rw.unsqueeze(0), transpose_b=True).transpose(1, 2)
    return _add(h.unsqueeze(-1), w.unsqueeze(-2)).reshape(*q.shape[:-1], height * width)


def _three_layer_relu_backward(g, x, w1, w2, w3, h1, a1, h2, a2):
    g2, gw3, gb3 = _linear_backward(g, a2, w3)
    g2 = _mul(g2, _unary('un_ff_15', h2))
    g1, gw2, gb2 = _linear_backward(g2, a1, w2)
    g1 = _mul(g1, _unary('un_ff_15', h1))
    gx, gw1, gb1 = _linear_backward(g1, x, w1)
    return gx, gw1, gb1, gw2, gb2, gw3, gb3


def _scores_topk(scores, k=8, renorm=True, epsilon=0.0, factor=1.0):
    rows, experts = scores.shape
    ids = torch.empty((rows, k), dtype=torch.int64, device=scores.device)
    weights = torch.empty((rows, k), dtype=scores.dtype, device=scores.device)
    _launch('scores_topk_f' if scores.dtype == torch.float32 else 'scores_topk_b', [_aligned(scores), ids, weights],
            ((rows + 7) // 8, 1, 1), {'I': [experts, k, int(renorm)], 'f': [float(epsilon), float(factor)]})
    return ids, weights


def _ordered_combine(y, inverse, ids, weights):
    out = torch.empty((ids.shape[0], y.shape[-1]), dtype=y.dtype, device=y.device)
    name = 'ordered_combine_ff' if y.dtype == torch.float32 else ('ordered_combine_bb' if weights.dtype == torch.bfloat16 else 'ordered_combine_bf')
    _launch(name, [_aligned(y), inverse, _aligned(ids), _aligned(weights), out], _blocks(out.numel(), 256),
            {'I': [out.shape[-1], ids.shape[-1]]})
    return out


def _moe_gemm_f32(x, weight, groups):
    experts, n, k = weight.shape
    out = torch.empty((x.shape[0], n), dtype=torch.float32, device=x.device)
    bs = _split3(weight, weight.numel(), weight.numel(), False).view(3 * experts, n, k)
    _gemm('gemm_tf_256_00g_00000', _tf32x3(x, True), _opb(bs, kstep=experts), out, x.shape[0], n, k,
          batches=experts, k_groups=3, groups=groups, max_tiles=(x.shape[0] + 127) // 128 + experts)
    return out


def _moe_forward(x, ids, weights, gate_w, up_w, down_w, explicit_sigmoid=False):
    groups, tokens, inverse = _moe_route(ids, gate_w.shape[0])
    fp32 = x.dtype == torch.float32
    xs = _moe_gather('moe_gather_f' if fp32 else 'moe_gather_b', x, tokens)
    def project(a, w):
        return _moe_gemm_f32(a, w, groups) if fp32 else _moe_gemm('gemm_bb_256_00g_00000', a, w, groups, w.shape[1], w.shape[2])
    gate, up = project(xs, gate_w), project(xs, up_w)
    if explicit_sigmoid:
        activated = _mul(gate, _unary('un_ff_3' if fp32 else 'un_bb_3', gate))
    else:
        activated = _unary('un_ff_0' if fp32 else 'un_bb_0', gate)
    y = project(_mul(activated, up), down_w)
    return _ordered_combine(y, inverse, ids, weights)


def _clamp(x, lower=None, upper=None):
    if lower is not None:
        bound = torch.tensor(float(lower), dtype=x.dtype, device=x.device)
        x = _binary('bin_fff_4' if x.dtype == torch.float32 else 'bin_bbb_4', x, bound)
    if upper is not None:
        bound = torch.tensor(float(upper), dtype=x.dtype, device=x.device)
        x = _binary('bin_fff_5' if x.dtype == torch.float32 else 'bin_bbb_5', x, bound)
    return x


def _group_norm_bf16(x, weight, bias, groups, eps):
    return _affine(_group_norm_f32(_f32(x), _f32(weight), _f32(bias), groups, eps), dtype=torch.bfloat16)


def _varlen_backward(ga, q, k, v, cu, scale, post_scale=False):
    offsets = cu.cpu().tolist()
    outputs, gqs, gks, gvs = [], [], [], []
    for start, end in zip(offsets, offsets[1:]):
        qq, kk, vv, gg = q[:, :, start:end], k[:, :, start:end], v[:, :, start:end], ga[:, :, start:end]
        if end == start:
            continue
        p = _softmax_raw(_affine(_matmul(qq, kk, transpose_b=True), scale))
        outputs.append(_matmul(p, vv))
        gs = _softmax_backward(_matmul(gg, vv, transpose_b=True), p)
        if post_scale:
            gqs.append(_affine(_matmul(gs, kk), scale))
            gks.append(_affine(_matmul(gs, qq, transpose_a=True), scale))
        else:
            gs = _affine(gs, scale)
            gqs.append(_matmul(gs, kk)); gks.append(_matmul(gs, qq, transpose_a=True))
        gvs.append(_matmul(p, gg, transpose_a=True))
    return tuple(torch.cat(parts, dim=2) for parts in (outputs, gqs, gks, gvs))


def _maxpool2(x):
    out = torch.empty((*x.shape[:2], x.shape[2] // 2, x.shape[3] // 2), dtype=x.dtype, device=x.device)
    _launch('maxpool2_f32', [_aligned(x), out], _blocks(out.numel(), 256), {'I': [x.shape[2], x.shape[3]]})
    return out


def _bilinear2(x, h, w):
    out = torch.empty((*x.shape[:2], h, w), dtype=x.dtype, device=x.device)
    _launch('bilinear2_f32', [_aligned(x), out], _blocks(out.numel(), 256), {'I': [x.shape[2], x.shape[3], h, w]})
    return out
