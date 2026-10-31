//! B200 (sm_100a) tcgen05 flash-attention forward.
//!
//! One CTA (128 threads) owns 128 queries of one (batch, head); thread t owns query
//! row t, which is also TMEM lane t. Per KV block of `bn` keys:
//!   S_j = Q K_j^T   (tcgen05, TMEM double buffer, issued one block ahead)
//!   online softmax in registers, P_j -> 128B-swizzled shared memory
//!   PV_j = P_j V_j  (tcgen05 into a TMEM block, accumulated into registers)
//! K/V are double-buffered TMA loads. Q/K/V are 3D tensor maps: `bhld` layouts
//! (d, L, B*H) with box (w, rows, 1), otherwise (B, L, H, d) as (d, H, B*L) with
//! box (w, 1, rows), w = 128 bytes of d. `kind` is the tcgen05 kind (0 f16,
//! 1 bf16, 2 tf32 with f32 storage).
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*};

pub const MASK_NONE: u32 = 0;
/// additive f32 mask[b * mask_b + h * mask_h + q * mask_q + k]
pub const MASK_ADD: u32 = 1;
/// u8 mask (nonzero = keep), same indexing
pub const MASK_BOOL: u32 = 2;

#[ruda(fast_math = FastMath::ReducedPrecision.into())]
pub fn exp2_fast(x: f32) -> f32 {
    f32::powf(2.0f32, x)
}

#[ruda]
fn load_rows<I: Numeric>(barrier: &barrier::Barrier, source: &TensorMap<I, Tiled>, tile: &mut SharedMemory<I>,
    base: usize, outer: usize, position: usize, head: usize,
    #[comptime] rows: usize, #[comptime] blocks: usize, #[comptime] width: usize, #[comptime] bhld: bool,
) {
    // `outer` = b * H + h for bhld (position = l0), b * L + l0 for blhd (head = h).
    #[unroll]
    for block in 0usize..blocks {
        let mut target = tile.slice_mut(base + block * rows * width, base + (block + 1) * rows * width);
        if comptime!(bhld) {
            barrier.tma_load_3d(source, &mut target, outer as i32, position as i32, (block * width) as i32);
        } else {
            barrier.tma_load_3d(source, &mut target, outer as i32, head as i32, (block * width) as i32);
        }
    }
}

/// Scaled log2-domain score with masking; -inf when masked.
#[ruda]
fn score(raw: f32, query: usize, key: usize, keys: usize, offset: usize, window: u32, scale: f32, cap: f32,
    mask: &Array<f32>, flags: &Array<u8>, mask_base: usize, mask_q: u32,
    #[comptime] causal: bool, #[comptime] windowed: bool, #[comptime] softcap: bool, #[comptime] mask_mode: u32,
) -> f32 {
    let mut s = raw * scale;
    if comptime!(softcap) { s = cap * (s / cap).tanh(); }
    if comptime!(mask_mode == MASK_ADD) { s += mask[mask_base + query * mask_q as usize + key]; }
    s = s * 1.4426950408889634f32;
    let mut valid = key < keys;
    if comptime!(causal) { valid = valid && key <= query + offset; }
    if comptime!(windowed) { valid = valid && key + window as usize > query + offset; }
    if comptime!(mask_mode == MASK_BOOL) {
        if valid { valid = flags[mask_base + query * mask_q as usize + key] != 0u8; }
    }
    if !valid { s = f32::new(f32::NEG_INFINITY); }
    s
}

#[ruda(launch)]
pub fn flash_fwd<I: Numeric, O: Float>(
    q_map: &TensorMap<I, Tiled>, k_map: &TensorMap<I, Tiled>, v_map: &TensorMap<I, Tiled>,
    out: &mut Array<Vector<O, Const<8>>>, lse: &mut Array<f32>, mask: &Array<f32>, flags: &Array<u8>,
    queries: u32, keys: u32, heads: u32, kv_heads: u32, window: u32, mask_b: u32, mask_h: u32, mask_q: u32,
    scale: f32, cap: f32,
    #[define(I)] _input: StorageType, #[define(O)] _output: StorageType,
    #[comptime] dim: usize, #[comptime] bn: usize, #[comptime] stages: usize, #[comptime] kind: u32,
    #[comptime] bhld: bool, #[comptime] causal: bool, #[comptime] windowed: bool, #[comptime] softcap: bool,
    #[comptime] mask_mode: u32, #[comptime] write_lse: bool,
) {
    let bytes = comptime!(if kind == 2 { 4usize } else { 2usize });
    let width = comptime!(128 / bytes);
    let dblocks = comptime!(dim / width);
    let kblocks = comptime!(bn / width);
    let q_tile = comptime!(128 * dim);
    let kv_tile = comptime!(bn * dim);
    let mut q = SharedMemory::<I>::new_aligned(q_tile, 1024usize);
    let mut kt = SharedMemory::<I>::new_aligned(kv_tile * stages, 1024usize);
    let mut vt = SharedMemory::<I>::new_aligned(kv_tile * stages, 1024usize);
    let mut p = SharedMemory::<I>::new_aligned(comptime!(128 * bn), 1024usize);
    let mut allocation = SharedMemory::<u32>::new(1usize);
    let q_full = barrier::Barrier::shared(1u32, UNIT_POS == 0);
    let mut k_full = Sequence::<Shared<barrier::Barrier>>::new();
    let mut v_full = Sequence::<Shared<barrier::Barrier>>::new();
    #[unroll]
    for _ in 0usize..stages {
        k_full.push(barrier::Barrier::shared(1u32, UNIT_POS == 0));
        v_full.push(barrier::Barrier::shared(1u32, UNIT_POS == 0));
    }
    let s_done_0 = barrier::Barrier::shared(1u32, UNIT_POS == 0);
    let s_done_1 = barrier::Barrier::shared(1u32, UNIT_POS == 0);
    let pv_done = barrier::Barrier::shared(1u32, UNIT_POS == 0);
    let columns = comptime!((2 * bn + dim).next_power_of_two() as u32);
    sync_async_proxy_shared();
    if UNIT_POS < 32 {
        tcgen05::allocate(&mut allocation.slice_mut(0usize, 1usize), columns);
        tcgen05::relinquish();
    }
    sync_ruda();
    let address = allocation[0];
    let lane_address = address + ((UNIT_POS / 32 * 32) << 16);
    let thread = UNIT_POS as usize;
    let tile = RUDA_POS_X as usize;
    let head = RUDA_POS_Y as usize;
    let batch = RUDA_POS_Z as usize;
    let kv_head = head / (heads as usize / kv_heads as usize);
    let q0 = tile * 128;
    let query = q0 + thread;
    let offset = keys as usize - queries as usize;
    let mut blocks = (keys as usize + bn - 1) / bn;
    if comptime!(causal) {
        let reach = (q0 + 128 + offset + bn - 1) / bn;
        if reach < blocks { blocks = reach; }
    }
    let mut q_outer = batch * queries as usize + q0;
    let mut kv_outer_base = batch * keys as usize;
    let mut kv_step = bn;
    if comptime!(bhld) {
        q_outer = batch * heads as usize + head;
        kv_outer_base = batch * kv_heads as usize + kv_head;
        kv_step = 0usize;
    }
    let mask_base = batch * mask_b as usize + head * mask_h as usize;
    let tx_q = comptime!((128 * dim * bytes) as u32);
    let tx_kv = comptime!((bn * dim * bytes) as u32);

    if thread == 0 {
        q_full.arrive_and_expect_tx(1u32, tx_q);
        load_rows(&q_full, q_map, &mut q, 0usize, q_outer, q0, head, 128usize, dblocks, width, bhld);
        #[unroll]
        for stage in 0usize..stages {
            if stage < blocks {
                let k_outer = kv_outer_base + stage * kv_step;
                k_full[stage].arrive_and_expect_tx(1u32, tx_kv);
                load_rows(&k_full[stage], k_map, &mut kt, stage * kv_tile, k_outer, stage * bn, kv_head, bn, dblocks, width, bhld);
                v_full[stage].arrive_and_expect_tx(1u32, tx_kv);
                load_rows(&v_full[stage], v_map, &mut vt, stage * kv_tile, k_outer, stage * bn, kv_head, bn, dblocks, width, bhld);
            }
        }
        if blocks > 0 {
            q_full.wait_parity(0u32);
            k_full[0].wait_parity(0u32);
            tcgen05::fence(false);
            #[unroll]
            for step in 0usize..comptime!(dim * bytes / 32) {
                let base = (step / 4) * 128 * width + (step % 4) * comptime!(32 / bytes);
                let kbase = (step / 4) * bn * width + (step % 4) * comptime!(32 / bytes);
                let ad = tcgen05::descriptor(&q.slice(base, base + 8));
                let bd = tcgen05::descriptor(&kt.slice(kbase, kbase + 8));
                tcgen05::mma_kind(address, ad, bd, step != 0, bn as u32, kind, false, false);
            }
            tcgen05::commit(&s_done_0);
        }
    }

    let mut o = Array::<f32>::new(dim);
    #[unroll]
    for i in 0usize..dim { o[i] = 0.0f32; }
    let mut maximum = f32::new(f32::NEG_INFINITY);
    let mut total = 0.0f32;
    for j in 0usize..blocks {
        let buffer = j % 2;
        if buffer == 0 { s_done_0.wait_parity(((j / 2) & 1) as u32); } else { s_done_1.wait_parity(((j / 2) & 1) as u32); }
        tcgen05::fence(false);
        let stage = j % stages;
        if thread == 0 {
            // K stage of block j is free: prefetch K_{j + stages}; issue S_{j + 1}.
            let next = j + stages;
            if next < blocks {
                let k_outer = kv_outer_base + next * kv_step;
                if stage == 0 {
                    k_full[0].arrive_and_expect_tx(1u32, tx_kv);
                    load_rows(&k_full[0], k_map, &mut kt, 0usize, k_outer, next * bn, kv_head, bn, dblocks, width, bhld);
                } else {
                    if comptime!(stages > 1) {
                        k_full[1].arrive_and_expect_tx(1u32, tx_kv);
                        load_rows(&k_full[1], k_map, &mut kt, kv_tile, k_outer, next * bn, kv_head, bn, dblocks, width, bhld);
                    }
                }
            }
            if j + 1 < blocks {
                let ns = (j + 1) % stages;
                let k_phase = (((j + 1) / stages) & 1) as u32;
                if ns == 0 { k_full[0].wait_parity(k_phase); } else {
                    if comptime!(stages > 1) { k_full[1].wait_parity(k_phase); }
                }
                tcgen05::fence(false);
                let target = address + (((j + 1) % 2) * bn) as u32;
                #[unroll]
                for step in 0usize..comptime!(dim * bytes / 32) {
                    let base = (step / 4) * 128 * width + (step % 4) * comptime!(32 / bytes);
                    let kbase = ns * kv_tile + (step / 4) * bn * width + (step % 4) * comptime!(32 / bytes);
                    let ad = tcgen05::descriptor(&q.slice(base, base + 8));
                    let bd = tcgen05::descriptor(&kt.slice(kbase, kbase + 8));
                    tcgen05::mma_kind(target, ad, bd, step != 0, bn as u32, kind, false, false);
                }
                if (j + 1) % 2 == 0 { tcgen05::commit(&s_done_0); } else { tcgen05::commit(&s_done_1); }
            }
        }
        // Pass 1: block maximum.
        let s_address = lane_address + (buffer * bn) as u32;
        let mut block_max = f32::new(f32::NEG_INFINITY);
        #[unroll]
        for chunk in 0usize..(bn / 32) {
            let values = tcgen05::load(s_address + (chunk * 32) as u32);
            #[unroll]
            for i in 0usize..32 {
                let s = score(values[i], query, j * bn + chunk * 32 + i, keys as usize, offset, window, scale, cap,
                    mask, flags, mask_base, mask_q, causal, windowed, softcap, mask_mode);
                block_max = f32::max(block_max, s);
            }
        }
        let next_max = f32::max(maximum, block_max);
        let mut used = next_max;
        if next_max == f32::new(f32::NEG_INFINITY) { used = 0.0f32; }
        let correction = exp2_fast(maximum - used);
        // Fold the pending PV of block j - 1 (P_{j-1} is relative to `maximum`).
        if j > 0 {
            pv_done.wait_parity(((j - 1) & 1) as u32);
            tcgen05::fence(false);
            if thread == 0 {
                let previous = (j - 1) % stages;
                let next = j - 1 + stages;
                if next < blocks {
                    let v_outer = kv_outer_base + next * kv_step;
                    if previous == 0 {
                        v_full[0].arrive_and_expect_tx(1u32, tx_kv);
                        load_rows(&v_full[0], v_map, &mut vt, 0usize, v_outer, next * bn, kv_head, bn, dblocks, width, bhld);
                    } else {
                        if comptime!(stages > 1) {
                            v_full[1].arrive_and_expect_tx(1u32, tx_kv);
                            load_rows(&v_full[1], v_map, &mut vt, kv_tile, v_outer, next * bn, kv_head, bn, dblocks, width, bhld);
                        }
                    }
                }
            }
            #[unroll]
            for chunk in 0usize..(dim / 32) {
                let values = tcgen05::load(lane_address + (2 * bn + chunk * 32) as u32);
                #[unroll]
                for i in 0usize..32 { o[chunk * 32 + i] += values[i]; }
            }
        }
        #[unroll]
        for i in 0usize..dim { o[i] = o[i] * correction; }
        total = total * correction;
        maximum = next_max;
        // Pass 2: probabilities into 128B-swizzled K-major P (row = thread).
        #[unroll]
        for chunk in 0usize..(bn / 32) {
            let values = tcgen05::load(s_address + (chunk * 32) as u32);
            #[unroll]
            for i in 0usize..32 {
                let key = chunk * 32 + i;
                let s = score(values[i], query, j * bn + key, keys as usize, offset, window, scale, cap,
                    mask, flags, mask_base, mask_q, causal, windowed, softcap, mask_mode);
                let probability = I::cast_from(exp2_fast(s - used));
                total += f32::cast_from(probability);
                let block = key / width;
                let within = key % width;
                let chunk16 = (within * bytes / 16) ^ (thread % 8);
                let element = block * 128 * width + thread * width + chunk16 * comptime!(16 / bytes) + within % comptime!(16 / bytes);
                p[element] = probability;
            }
        }
        tcgen05::fence(true);
        sync_async_proxy_shared();
        sync_ruda();
        if thread == 0 {
            tcgen05::fence(false);
            let v_phase = ((j / stages) & 1) as u32;
            if stage == 0 { v_full[0].wait_parity(v_phase); } else {
                if comptime!(stages > 1) { v_full[1].wait_parity(v_phase); }
            }
            #[unroll]
            for step in 0usize..comptime!(bn * bytes / 32) {
                let abase = (step / 4) * 128 * width + (step % 4) * comptime!(32 / bytes);
                let vbase = stage * kv_tile + step * comptime!(32 / bytes) * width;
                let ad = tcgen05::descriptor(&p.slice(abase, abase + 8));
                let mut bd = 0u64;
                if comptime!(kind == 2) {
                    bd = tcgen05::descriptor_tf32_mn(&vt.slice(vbase, vbase + 8), comptime!((bn * 128) as u32));
                } else {
                    bd = tcgen05::descriptor_strided(&vt.slice(vbase, vbase + 8), comptime!((bn * 128) as u32), 1024u32);
                }
                tcgen05::mma_kind(address + (2 * bn) as u32, ad, bd, step != 0, dim as u32, kind, false, true);
            }
            tcgen05::commit(&pv_done);
        }
    }
    if blocks > 0 {
        pv_done.wait_parity(((blocks - 1) & 1) as u32);
        tcgen05::fence(false);
        #[unroll]
        for chunk in 0usize..(dim / 32) {
            let values = tcgen05::load(lane_address + (2 * bn + chunk * 32) as u32);
            #[unroll]
            for i in 0usize..32 { o[chunk * 32 + i] += values[i]; }
        }
    }
    if query < queries as usize {
        let mut inverse = 0.0f32;
        if total > 0.0f32 { inverse = 1.0f32 / total; }
        let mut row = (batch * queries as usize + query) * heads as usize + head;
        if comptime!(bhld) { row = (batch * heads as usize + head) * queries as usize + query; }
        #[unroll]
        for v in 0usize..(dim / 8) {
            let mut result = Vector::<O, Const<8>>::empty();
            #[unroll]
            for i in 0usize..8 { result[i] = O::cast_from(o[v * 8 + i] * inverse); }
            out[row * (dim / 8) + v] = result;
        }
        if comptime!(write_lse) {
            let mut value = f32::new(f32::NEG_INFINITY);
            if total > 0.0f32 { value = (maximum * 0.6931471805599453f32) + total.ln(); }
            lse[(batch * heads as usize + head) * queries as usize + query] = value;
        }
    }
    tcgen05::fence(true);
    sync_ruda();
    if UNIT_POS < 32 { tcgen05::deallocate(address, columns); }
}
