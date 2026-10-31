//! B200 (sm_100a) warp-specialized persistent tcgen05 GEMM:
//! `D[b] = epilogue(op(A[b]) @ op(B[b])^T)` with M tile 128 and N tile `bn`.
//!
//! Roles (192 threads): warp 0 lane 0 issues TMA loads, warp 1 lane 0 issues
//! tcgen05.mma, warps 2..5 (128 threads) drain TMEM and run the fused epilogue.
//! The accumulator is double-buffered in TMEM (2 x bn columns), so the epilogue of
//! tile t overlaps the main loop of tile t + 1; the shared ring is `stages` deep.
//!
//! Operand kinds: 0 f16, 1 bf16, 2 tf32 (f32 storage), 3 e4m3, 4 e5m2 (u8 storage).
//! Every stage covers 128 bytes of K. K-major operands (`A` (M, K), `B` (N, K)) are
//! loaded as one 128B-swizzled box per tile; MN-major operands (`A` stored (K, M),
//! `B` stored (K, N)) as 128B-wide panels. K / MN tails are zero-filled by TMA.
//! The epilogue writes 8-element vectors straight from registers (ld % 8 == 0).
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

pub const ACT_NONE: u32 = 0;
pub const ACT_GELU_TANH: u32 = 1;
pub const ACT_GELU_ERF: u32 = 2;
pub const ACT_SILU: u32 = 3;
pub const ACT_RELU: u32 = 4;
pub const ACT_SIGMOID: u32 = 5;
pub const ACT_TANH: u32 = 6;
/// x * sigmoid(1.702 x)
pub const ACT_QUICK_GELU: u32 = 7;

/// Residual combination after the activation.
pub const COMBINE_NONE: u32 = 0;
/// out = v + beta * r
pub const COMBINE_ADD: u32 = 1;
/// out = silu(r) * v (r = gate projection)
pub const COMBINE_SILU_GATE: u32 = 2;
/// out = gelu_tanh(r) * v
pub const COMBINE_GELU_GATE: u32 = 3;
/// out = r * v
pub const COMBINE_MUL: u32 = 4;
/// out = gelu_erf(r) * v
pub const COMBINE_GELU_ERF_GATE: u32 = 5;
/// out = v * gelu_tanh'(r) (r = pre-activation)
pub const COMBINE_GELU_TANH_BWD: u32 = 6;
/// out = v * gelu_erf'(r)
pub const COMBINE_GELU_ERF_BWD: u32 = 7;
/// out = v * silu'(r)
pub const COMBINE_SILU_BWD: u32 = 8;

/// Scale handling: 0 none, 1 per-tensor scale_a[0] * scale_b[0], 2 scale_a[row] * scale_b[col].
pub const SCALE_NONE: u32 = 0;
pub const SCALE_TENSOR: u32 = 1;
pub const SCALE_VECTOR: u32 = 2;
/// 3: scale_a[row] only.
pub const SCALE_ROW: u32 = 3;
pub const SCALE_TENSOR_POST: u32 = 4;

#[ruda]
pub fn activate(value: f32, #[comptime] activation: u32) -> f32 {
    let mut result = value;
    if comptime!(activation == ACT_GELU_TANH) {
        let inner = 0.7978845608028654f32 * (value + 0.044715f32 * (value * value * value));
        result = 0.5f32 * value * (1.0f32 + inner.tanh());
    } else if comptime!(activation == ACT_GELU_ERF) {
        result = value * 0.5f32 * (1.0f32 + (value * 0.7071067811865476f32).erf());
    } else if comptime!(activation == ACT_SILU) {
        result = value / (1.0f32 + (-value).exp());
    } else if comptime!(activation == ACT_RELU) {
        if value < 0.0f32 { result = 0.0f32; }
    } else if comptime!(activation == ACT_SIGMOID) {
        result = 1.0f32 / (1.0f32 + (-value).exp());
    } else if comptime!(activation == ACT_TANH) {
        result = value.tanh();
    } else if comptime!(activation == ACT_QUICK_GELU) {
        result = value * (1.0f32 / (1.0f32 + (-(1.702f32 * value)).exp()));
    }
    result
}

#[ruda]
pub fn round_through<O: Float>(value: f32) -> f32 {
    f32::cast_from(O::cast_from(value))
}

/// Affine batch -> 3D TMA coordinates of one operand. Every operand map is 3D
/// (inner, Y, Z): inner = K (K-major) or MN (MN-major); the outer position
/// (MN for K-major, K for MN-major) runs along Z.
///   z = (batch / z_div) * z_mul + position
///   y = ((batch % y_mod) / y_div) * y_mul + y_add + group * y_kstep
/// `group` indexes K groups (K accumulated over several Y slices).

/// Loads one operand tile (`rows` MN entries x 128 bytes of K) for one stage.
#[ruda]
fn load_operand<I: Numeric>(barrier: &barrier::Barrier, source: &TensorMap<I, Tiled>, tile: &mut SharedMemory<I>,
    base: usize, batch: u32, group: u32, mn: u32, k: u32,
    z_div: u32, z_mul: u32, y_mod: u32, y_div: u32, y_mul: u32, y_add: u32, y_kstep: u32,
    #[comptime] rows: usize, #[comptime] bytes: usize, #[comptime] mn_major: bool,
) {
    let width = comptime!(128 / bytes);
    let z_base = (batch / z_div) * z_mul;
    let y = (((batch % y_mod) / y_div) * y_mul + y_add + group * y_kstep) as i32;
    if comptime!(!mn_major) {
        let mut target = tile.slice_mut(base, base + rows * width);
        barrier.tma_load_3d(source, &mut target, (z_base + mn) as i32, y, k as i32);
    } else {
        // Panels of 128 bytes along MN, each (128 / bytes) K rows deep.
        #[unroll]
        for panel in 0usize..comptime!(rows * bytes / 128) {
            let mut target = tile.slice_mut(base + panel * width * width, base + (panel + 1) * width * width);
            barrier.tma_load_3d(source, &mut target, (z_base + k) as i32, y, (mn + (panel * width) as u32) as i32);
        }
    }
}

/// Tile index -> (batch, local row start, column start, row base). Dense mode uses
/// grouped rasterization (8 M tiles) per batch; grouped (MoE) mode reads
/// groups = [row prefix (E + 1), M-tile prefix (E + 1)] and maps M tiles to experts.
#[ruda]
fn tile_coordinates(tile: usize, count_m: usize, count_n: usize, groups: &Array<u32>, experts: u32,
    #[comptime] bn: usize, #[comptime] grouped: bool,
) -> Vector<u32, Const<4>> {
    let mut result = Vector::<u32, Const<4>>::empty();
    if comptime!(grouped) {
        let e_count = experts as usize;
        let mt = (tile / count_n) as u32;
        // largest e with tile_prefix[e] <= mt
        let mut lo = 0usize;
        let mut hi = e_count;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if groups[e_count + 1 + mid] <= mt { lo = mid; } else { hi = mid; }
        }
        result[0] = lo as u32;
        result[1] = (mt - groups[e_count + 1 + lo]) * 128u32;
        result[2] = ((tile % count_n) * bn) as u32;
        result[3] = groups[lo];
    } else {
        let per_batch = count_m * count_n;
        let local_tile = tile % per_batch;
        let first = local_tile / (count_n * 8) * 8;
        let mut height = 8usize;
        if count_m - first < height { height = count_m - first; }
        let local = local_tile % (count_n * 8);
        result[0] = (tile / per_batch) as u32;
        result[1] = ((first + local % height) * 128) as u32;
        result[2] = ((local / height) * bn) as u32;
        result[3] = 0u32;
    }
    result
}

#[ruda]
fn operand_descriptor<I: Numeric>(tile: &SharedMemory<I>, base: usize, inner: usize,
    #[comptime] bytes: usize, #[comptime] mn_major: bool,
) -> u64 {
    let width = comptime!(128 / bytes);
    let mut result = 0u64;
    if comptime!(mn_major) {
        // K step of 32 bytes worth of K rows; panels are width * 128 bytes apart.
        let start = base + inner * comptime!(32 / bytes) * width;
        if comptime!(bytes == 4) {
            result = tcgen05::descriptor_tf32_mn(&tile.slice(start, start + 8), comptime!((width * 128) as u32));
        } else {
            result = tcgen05::descriptor_strided(&tile.slice(start, start + 8), comptime!((width * 128) as u32), 1024u32);
        }
    } else {
        let start = base + inner * comptime!(32 / bytes);
        result = tcgen05::descriptor(&tile.slice(start, start + 8));
    }
    result
}

/// Persistent warp-specialized tcgen05 GEMM with fused epilogue
/// `out = combine(act(alpha * scale * acc + bias), residual)`; with `exact`
/// every step rounds through `O` (reproduces separate PyTorch kernels).
/// `ld` is the output / residual row stride in 8-element vectors; batch `t`
/// starts at vector (t / o_div) * o_s1 + (t % o_div) * o_s2; output vector column
/// v lands at (v / col_split) * col_stride + v % col_split. `k` is the K extent
/// of one K group; `k_groups` groups are accumulated (operand y += group * y_kstep).
#[ruda(launch)]
pub fn gemm_ws<I: Numeric, O: Float>(
    lhs: &TensorMap<I, Tiled>, rhs: &TensorMap<I, Tiled>,
    out: &mut Array<Vector<O, Const<8>>>, bias: &Array<O>, residual: &Array<Vector<O, Const<8>>>,
    scale_a: &Array<f32>, scale_b: &Array<f32>, groups: &Array<u32>,
    m: u32, n: u32, k: u32, batches: u32, k_groups: u32, ld: u32, o_div: u32, o_s1: u32, o_s2: u32,
    col_split: u32, col_stride: u32, bias_stride: u32, res_s1: u32,
    a_z_div: u32, a_z_mul: u32, a_y_mod: u32, a_y_div: u32, a_y_mul: u32, a_y_add: u32, a_y_kstep: u32,
    b_z_div: u32, b_z_mul: u32, b_y_mod: u32, b_y_div: u32, b_y_mul: u32, b_y_add: u32, b_y_kstep: u32,
    alpha: f32, beta: f32,
    #[define(I)] _input: StorageType, #[define(O)] _output: StorageType,
    #[comptime] bn: usize, #[comptime] stages: usize, #[comptime] kind: u32,
    #[comptime] a_mn: bool, #[comptime] b_mn: bool,
    #[comptime] use_bias: u32, #[comptime] activation: u32, #[comptime] combine: u32,
    #[comptime] scale: u32, #[comptime] exact: u32, #[comptime] grouped: bool,
) {
    let bytes = comptime!(if kind == 2 { 4usize } else if kind >= 3 { 1usize } else { 2usize });
    let width = comptime!(128 / bytes);
    let tile_a = comptime!(128 * width);
    let tile_b = comptime!(bn * width);
    let mut a = SharedMemory::<I>::new_aligned(tile_a * stages, 1024usize);
    let mut b = SharedMemory::<I>::new_aligned(tile_b * stages, 1024usize);
    let mut allocation = SharedMemory::<u32>::new(1usize);
    let mut full = Sequence::<Shared<barrier::Barrier>>::new();
    let mut empty = Sequence::<Shared<barrier::Barrier>>::new();
    #[unroll]
    for _ in 0usize..stages {
        full.push(barrier::Barrier::shared(1u32, UNIT_POS == 0));
        empty.push(barrier::Barrier::shared(1u32, UNIT_POS == 0));
    }
    let accumulator_full_0 = barrier::Barrier::shared(1u32, UNIT_POS == 0);
    let accumulator_full_1 = barrier::Barrier::shared(1u32, UNIT_POS == 0);
    let accumulator_empty_0 = barrier::Barrier::shared(128u32, UNIT_POS == 0);
    let accumulator_empty_1 = barrier::Barrier::shared(128u32, UNIT_POS == 0);
    sync_async_proxy_shared();
    if UNIT_POS < 32 {
        tcgen05::allocate(&mut allocation.slice_mut(0usize, 1usize), comptime!((2 * bn) as u32));
        tcgen05::relinquish();
    }
    sync_ruda();
    let address = allocation[0];
    let warp = UNIT_POS / 32;
    let lane = UNIT_POS % 32;
    let count_m = (m as usize + 127) / 128;
    let count_n = (n as usize + bn - 1) / bn;
    let mut tiles = count_m * count_n * batches as usize;
    if comptime!(grouped) { tiles = groups[2 * batches as usize + 1] as usize * count_n; }
    let group_tiles = (k as usize + width - 1) / width;
    let tiles_k = group_tiles * k_groups as usize;
    let cycles = (tiles_k + stages - 1) / stages;
    let workers = RUDA_COUNT_X as usize;

    if warp == 0u32 {
        if lane == 0u32 {
            // TMA producer.
            let mut tile = RUDA_POS_X as usize;
            let mut iteration = 0usize;
            while tile < tiles {
                let coords = tile_coordinates(tile, count_m, count_n, groups, batches, bn, grouped);
                let batch = coords[0] as usize;
                let row = (coords[1] + coords[3]) as usize;
                let column = coords[2] as usize;
                for cycle in 0usize..cycles {
                    #[unroll]
                    for stage in 0usize..stages {
                        let step = cycle * stages + stage;
                        if step < tiles_k {
                            let uses = iteration * ((tiles_k + stages - 1 - stage) / stages) + cycle;
                            if uses > 0 { empty[stage].wait_parity(((uses - 1) & 1) as u32); }
                            full[stage].arrive_and_expect_tx(1u32, comptime!(((128 + bn) * 128) as u32));
                            let kgroup = (step / group_tiles) as u32;
                            let kk = ((step % group_tiles) * width) as u32;
                            load_operand(&full[stage], lhs, &mut a, stage * tile_a, batch as u32, kgroup, row as u32, kk,
                                a_z_div, a_z_mul, a_y_mod, a_y_div, a_y_mul, a_y_add, a_y_kstep, 128usize, bytes, a_mn);
                            load_operand(&full[stage], rhs, &mut b, stage * tile_b, batch as u32, kgroup, column as u32, kk,
                                b_z_div, b_z_mul, b_y_mod, b_y_div, b_y_mul, b_y_add, b_y_kstep, bn, bytes, b_mn);
                        }
                    }
                }
                iteration += 1;
                tile += workers;
            }
        }
    } else if warp == 1u32 {
        if lane == 0u32 {
            // tcgen05 issuer.
            let mut tile = RUDA_POS_X as usize;
            let mut iteration = 0usize;
            while tile < tiles {
                let buffer = iteration % 2;
                let round = iteration / 2;
                if round > 0 {
                    if buffer == 0 { accumulator_empty_0.wait_parity(((round - 1) & 1) as u32); }
                    else { accumulator_empty_1.wait_parity(((round - 1) & 1) as u32); }
                }
                tcgen05::fence(false);
                let target = address + (buffer * bn) as u32;
                for cycle in 0usize..cycles {
                    #[unroll]
                    for stage in 0usize..stages {
                        let step = cycle * stages + stage;
                        if step < tiles_k {
                            let uses = iteration * ((tiles_k + stages - 1 - stage) / stages) + cycle;
                            full[stage].wait_parity((uses & 1) as u32);
                            tcgen05::fence(false);
                            #[unroll]
                            for inner in 0usize..4 {
                                let ad = operand_descriptor(&a, stage * tile_a, inner, bytes, a_mn);
                                let bd = operand_descriptor(&b, stage * tile_b, inner, bytes, b_mn);
                                tcgen05::mma_kind(target, ad, bd, step != 0 || inner != 0, bn as u32, kind, a_mn, b_mn);
                            }
                            tcgen05::commit(&empty[stage]);
                        }
                    }
                }
                if buffer == 0 { tcgen05::commit(&accumulator_full_0); } else { tcgen05::commit(&accumulator_full_1); }
                iteration += 1;
                tile += workers;
            }
        }
    } else {
        // Epilogue warpgroup: warp w reads TMEM lanes 32 * (w % 4).
        let quadrant = (warp % 4) as usize;
        let mut tile = RUDA_POS_X as usize;
        let mut iteration = 0usize;
        let mut tensor_scale = alpha;
        if comptime!(scale == SCALE_TENSOR || scale == SCALE_TENSOR_POST) { tensor_scale = alpha * scale_a[0] * scale_b[0]; }
        while tile < tiles {
            let coords = tile_coordinates(tile, count_m, count_n, groups, batches, bn, grouped);
            let mut batch = coords[0] as usize;
            let local_row = coords[1] as usize + quadrant * 32 + lane as usize;
            let column = coords[2] as usize;
            let mut limit = m as usize;
            let mut row = local_row;
            if comptime!(grouped) {
                limit = (groups[batch + 1] - groups[batch]) as usize;
                row = coords[3] as usize + local_row;
                batch = 0usize;
            }
            let buffer = iteration % 2;
            let round = iteration / 2;
            if buffer == 0 { accumulator_full_0.wait_parity((round & 1) as u32); }
            else { accumulator_full_1.wait_parity((round & 1) as u32); }
            tcgen05::fence(false);
            let mut row_scale = tensor_scale;
            if comptime!(scale == SCALE_VECTOR || scale == SCALE_ROW) {
                if local_row < limit { row_scale = alpha * scale_a[batch * m as usize + row]; }
            }
            #[unroll]
            for chunk in 0usize..(bn / 32) {
                let values = tcgen05::load(address + ((quadrant as u32 * 32u32) << 16) + (buffer * bn + chunk * 32) as u32);
                #[unroll]
                for part in 0usize..4 {
                    let col = column + chunk * 32 + part * 8;
                    if local_row < limit && col < n as usize {
                        let vector = col / 8;
                        let index = (batch / o_div as usize) * o_s1 as usize + (batch % o_div as usize) * o_s2 as usize
                            + row * ld as usize + (vector / col_split as usize) * col_stride as usize + vector % col_split as usize;
                        let mut prior = Vector::<f32, Const<8>>::new(0.0f32);
                        if comptime!(combine != COMBINE_NONE) {
                            let r_index = index - (batch / o_div as usize) * o_s1 as usize + (batch / o_div as usize) * res_s1 as usize;
                            prior = Vector::<f32, Const<8>>::cast_from(residual[r_index]);
                        }
                        let mut result = Vector::<O, Const<8>>::empty();
                        #[unroll]
                        for i in 0usize..8 {
                            let mut v = row_scale * values[part * 8 + i];
                            if comptime!(scale == SCALE_TENSOR_POST) { v = row_scale * round_through::<O>(values[part * 8 + i]); }
                            if comptime!(scale == SCALE_VECTOR) { v = v * scale_b[batch * n as usize + col + i]; }
                            if comptime!(use_bias != 0u32) {
                                if comptime!(exact >= 2u32) { v = round_through::<O>(v); }
                                if comptime!(use_bias == 2u32) {
                                    v += f32::cast_from(bias[coords[0] as usize * bias_stride as usize + local_row]);
                                } else {
                                    v += f32::cast_from(bias[coords[0] as usize * bias_stride as usize + col + i]);
                                }
                            }
                            if comptime!(exact >= 1u32) { v = round_through::<O>(v); }
                            if comptime!(activation != ACT_NONE) {
                                v = activate(v, activation);
                                if comptime!(exact >= 1u32) { v = round_through::<O>(v); }
                            }
                            if comptime!(combine == COMBINE_ADD) {
                                v += beta * prior[i];
                            } else if comptime!(combine == COMBINE_SILU_GATE) {
                                let mut g = activate(prior[i], ACT_SILU);
                                if comptime!(exact >= 1u32) { g = round_through::<O>(g); }
                                v = g * v;
                            } else if comptime!(combine == COMBINE_GELU_GATE) {
                                let mut g = activate(prior[i], ACT_GELU_TANH);
                                if comptime!(exact >= 1u32) { g = round_through::<O>(g); }
                                v = g * v;
                            } else if comptime!(combine == COMBINE_GELU_ERF_GATE) {
                                let mut g = activate(prior[i], ACT_GELU_ERF);
                                if comptime!(exact >= 1u32) { g = round_through::<O>(g); }
                                v = g * v;
                            } else if comptime!(combine == COMBINE_MUL) {
                                v = prior[i] * v;
                            } else if comptime!(combine == COMBINE_GELU_TANH_BWD) {
                                let x = prior[i];
                                let t = (0.7978845608028654f32 * (x + 0.044715f32 * (x * x * x))).tanh();
                                let d = 0.7978845608028654f32 * (1.0f32 + 3.0f32 * 0.044715f32 * x * x);
                                v = v * (0.5f32 * (1.0f32 + t) + 0.5f32 * x * (1.0f32 - t * t) * d);
                            } else if comptime!(combine == COMBINE_GELU_ERF_BWD) {
                                let x = prior[i];
                                let cdf = 0.5f32 * (1.0f32 + (x * 0.7071067811865476f32).erf());
                                let pdf = 0.3989422804014327f32 * (-0.5f32 * x * x).exp();
                                v = v * (cdf + x * pdf);
                            } else if comptime!(combine == COMBINE_SILU_BWD) {
                                let x = prior[i];
                                let s = 1.0f32 / (1.0f32 + (-x).exp());
                                v = v * (s * (1.0f32 + x * (1.0f32 - s)));
                            }
                            result[i] = O::cast_from(v);
                        }
                        out[index] = result;
                    }
                }
            }
            tcgen05::fence(true);
            if buffer == 0 { accumulator_empty_0.arrive(); } else { accumulator_empty_1.arrive(); }
            iteration += 1;
            tile += workers;
        }
    }
    sync_ruda();
    if UNIT_POS < 32 { tcgen05::deallocate(address, comptime!((2 * bn) as u32)); }
}
