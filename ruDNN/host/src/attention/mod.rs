//! Attention (scaled dot-product) for CPU.
//!
//! Computes: softmax(Q @ K^T * scale + bias) @ V
//!
//! Two strategies, auto-selected by `attention()` based on sequence length:
//!
//! - **Naive** (seq_kv <= 8*TILE_KV): materializes full score matrix, two
//!   large gemm calls per (batch, head). Faster for short sequences.
//! - **Flash** (seq_kv > 8*TILE_KV): tiles over KV with online softmax,
//!   O(TILE_KV) scratch per row. Better cache behavior for long sequences.

use alloc::vec;
use alloc::vec::Vec;
use ruda_core::tensor::DType;
use ruda_core::tensor::spatial::AttentionModuleOptions;
use ruda_core::bytes::Bytes;
use bytemuck::Pod;
use num_traits::Float;

use ruda_core::tensor::host::{HostTensor, Layout};

/// KV tile size for flash attention.
///
/// Chosen so the score row (TILE_KV * 4 bytes for f32) and the V tile
/// [TILE_KV, val_dim] fit comfortably in L1. With val_dim=128 the V tile
/// is 32KB; this fits well on Apple Silicon (64-128KB L1) and modern x86
/// (48KB+ L1d since Golden Cove / Zen 4). On older x86 with 32KB L1d the
/// tile saturates L1 but still benefits from L2 residency.
/// WASM targets use a smaller tile to stay within tighter cache budgets.
#[cfg(target_family = "wasm")]
const TILE_KV: usize = 32;
#[cfg(not(target_family = "wasm"))]
const TILE_KV: usize = 64;

/// Max score matrix size (in elements) for the naive path.
///
/// Naive attention materializes a [seq_q, seq_kv] score matrix per head.
/// When this exceeds the budget, flash attention is used instead.
/// 256K elements = 1 MB for f32, fits comfortably in L2.
const NAIVE_SCORE_BUDGET: usize = 256 * 1024;

/// Auto-selecting attention: picks the fastest strategy based on sequence length.
///
/// Uses naive attention when the score matrix (seq_q * seq_kv) fits within
/// `NAIVE_SCORE_BUDGET`. Falls back to flash attention for larger shapes.
pub fn attention(
    query: HostTensor,
    key: HostTensor,
    value: HostTensor,
    mask: Option<HostTensor>,
    attn_bias: Option<HostTensor>,
    options: AttentionModuleOptions,
) -> HostTensor {
    debug_assert!(
        query.layout().shape().num_dims() == 4,
        "attention: query must be 4D, got {}D",
        query.layout().shape().num_dims()
    );
    debug_assert!(
        key.layout().shape().num_dims() == 4,
        "attention: key must be 4D, got {}D",
        key.layout().shape().num_dims()
    );
    let seq_q = query.layout().shape()[2];
    let seq_kv = key.layout().shape()[2];
    if seq_q * seq_kv <= NAIVE_SCORE_BUDGET {
        return attention_naive(query, key, value, mask, attn_bias, options);
    }
    attention_flash(query, key, value, mask, attn_bias, options)
}

/// Dispatch attention by dtype, casting f16/bf16 to f32 for computation.
macro_rules! dispatch_attention_dtype {
    ($query:expr, $key:expr, $value:expr, $mask:expr, $attn_bias:expr, $options:expr, $impl_fn:ident) => {{
        let query = $query;
        let key = $key;
        let value = $value;
        let mask = $mask;
        let attn_bias = $attn_bias;
        let options = $options;
        let dtype = query.dtype();
        debug_assert_eq!(key.dtype(), dtype, "attention: key dtype mismatch");
        debug_assert_eq!(value.dtype(), dtype, "attention: value dtype mismatch");
        if let Some(ref b) = attn_bias {
            debug_assert_eq!(b.dtype(), dtype, "attention: attn_bias dtype mismatch");
        }
        match dtype {
            DType::F32 => $impl_fn::<f32>(query, key, value, mask, attn_bias, options),
            DType::F64 => $impl_fn::<f64>(query, key, value, mask, attn_bias, options),
            DType::F16 => {
                use half::f16;
                let r = $impl_fn::<f32>(
                    cast_to_f32(query, f16::to_f32),
                    cast_to_f32(key, f16::to_f32),
                    cast_to_f32(value, f16::to_f32),
                    mask,
                    attn_bias.map(|b| cast_to_f32(b, f16::to_f32)),
                    options,
                );
                cast_from_f32(r, f16::from_f32)
            }
            DType::BF16 => {
                use half::bf16;
                let r = $impl_fn::<f32>(
                    cast_to_f32(query, bf16::to_f32),
                    cast_to_f32(key, bf16::to_f32),
                    cast_to_f32(value, bf16::to_f32),
                    mask,
                    attn_bias.map(|b| cast_to_f32(b, bf16::to_f32)),
                    options,
                );
                cast_from_f32(r, bf16::from_f32)
            }
            dtype => panic!("attention: unsupported dtype {:?}", dtype),
        }
    }};
}

/// Contiguous mask/bias tensor plus the per-batch and per-head element offsets the
/// inner loop should use to locate the `[seq_q, seq_kv]` tile for each `(batch, head)`
/// pair. When a leading dim (batch or heads) is `1` in the source, its step is `0`, so
/// the inner loop re-reads the same tile for every pair without allocating an expanded
/// copy. The tile length itself is always `seq_q * seq_kv` and is computed at the call
/// site, so it is not stored here.
struct BroadcastMaskBias {
    tensor: HostTensor,
    batch_step: usize,
    head_step: usize,
}

/// Prepare an attention mask or bias for the inner loop, accepting ONNX Attention-23
/// broadcast shapes.
///
/// Stride-0 along the leading `[batch, heads]` dims is handled without materializing,
/// so the common ONNX patterns (`[1, 1, seq_q, seq_kv]`, `[batch, 1, seq_q, seq_kv]`,
/// `[1, heads, seq_q, seq_kv]`) stay zero-copy. That matters especially for the flash
/// path, where materializing an expanded mask/bias would allocate a full
/// `[batch, heads, seq_q, seq_kv]` buffer and negate flash attention's memory
/// efficiency. If the trailing `[seq_q, seq_kv]` dims are themselves broadcast (rare in
/// practice), we fall back to `expand` + `to_contiguous` so the tile stays contiguous
/// in memory for the inner loop's slice-based access.
fn broadcast_attn_mask_bias(
    tensor: HostTensor,
    target: [usize; 4],
    name: &'static str,
) -> BroadcastMaskBias {
    let ndim = tensor.layout().shape().num_dims();
    assert!(ndim == 4, "attention: {name} must be 4D, got {ndim}D");
    let shape = tensor.layout().shape();
    let src = [shape[0], shape[1], shape[2], shape[3]];
    for i in 0..4 {
        assert!(
            src[i] == target[i] || src[i] == 1,
            "attention: {name} dim {i} must be {} or 1, got {}",
            target[i],
            src[i]
        );
    }

    let tile_len = target[2] * target[3];

    // Broadcast on seq_q or seq_kv: the source's trailing tile has fewer elements
    // than `tile_len`, so per-pair slice access would under-read. Materialize via
    // expand + to_contiguous in that case.
    if src[2] != target[2] || src[3] != target[3] {
        let expanded = ruprim_host::expand::expand(tensor, ruda_core::tensor::Shape::new(target));
        return BroadcastMaskBias {
            tensor: expanded.to_contiguous(),
            batch_step: target[1] * tile_len,
            head_step: tile_len,
        };
    }

    // Trailing dims match the target. Keep the source at its own shape (size
    // `src[0] * src[1] * tile_len`) and zero-out the step for any leading dim of 1.
    BroadcastMaskBias {
        tensor: tensor.to_contiguous(),
        batch_step: if src[0] == 1 { 0 } else { src[1] * tile_len },
        head_step: if src[1] == 1 { 0 } else { tile_len },
    }
}

/// Flash attention: tiled computation with online softmax. Use directly to bypass auto-selection.
pub fn attention_flash(
    query: HostTensor,
    key: HostTensor,
    value: HostTensor,
    mask: Option<HostTensor>,
    attn_bias: Option<HostTensor>,
    options: AttentionModuleOptions,
) -> HostTensor {
    dispatch_attention_dtype!(query, key, value, mask, attn_bias, options, attention_impl)
}

fn cast_to_f32<E: ruda_core::tensor::element::Element + Pod + Copy>(
    tensor: HostTensor,
    to_f32: fn(E) -> f32,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let data: &[E] = tensor.storage();
    let f32_data: Vec<f32> = data.iter().map(|&v| to_f32(v)).collect();
    HostTensor::new(
        Bytes::from_elems(f32_data),
        Layout::contiguous(shape),
        DType::F32,
    )
}

fn cast_from_f32<E: ruda_core::tensor::element::Element + Pod + Copy>(
    tensor: HostTensor,
    from_f32: fn(f32) -> E,
) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let data: &[f32] = tensor.storage();
    let half_data: Vec<E> = data.iter().map(|&v| from_f32(v)).collect();
    HostTensor::new(
        Bytes::from_elems(half_data),
        Layout::contiguous(shape),
        E::dtype(),
    )
}

mod schedule;
use schedule::*;

/// Gemm dispatch for flash attention block matmuls.
///
/// Wraps `gemm::gemm` for f32 and f64 so `flash_attention_head` stays generic.
/// Convention: dst = alpha * dst + beta * (lhs @ rhs)
trait FlashGemm: Float + Pod + Copy + core::ops::AddAssign {
    /// Block matrix multiply used for score and value matmuls.
    ///
    /// # Safety
    /// All pointers must be valid for the given dimensions and strides.
    unsafe fn block_gemm(args: BlockGemmArgs<Self>);
}

/// Arguments for a block matrix multiply: dst = alpha * dst + beta * (lhs @ rhs).
struct BlockGemmArgs<T> {
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
}

macro_rules! impl_flash_gemm {
    ($ty:ty) => {
        impl FlashGemm for $ty {
            unsafe fn block_gemm(a: BlockGemmArgs<Self>) {
                unsafe {
                    gemm::gemm(
                        a.m,
                        a.n,
                        a.k,
                        a.dst,
                        a.dst_cs,
                        a.dst_rs,
                        a.read_dst,
                        a.lhs,
                        a.lhs_cs,
                        a.lhs_rs,
                        a.rhs,
                        a.rhs_cs,
                        a.rhs_rs,
                        a.alpha,
                        a.beta,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
            }
        }
    };
}

impl_flash_gemm!(f32);
impl_flash_gemm!(f64);

/// Scratch buffers reused across (batch, head) pairs to avoid per-head allocation.
struct ScratchBuffers<T> {
    row_max: Vec<T>,
    row_sum: Vec<T>,
    scores: Vec<T>,
}

/// Parameters for a single (batch, head) flash attention computation.
struct AttentionParams<T> {
    scale: T,
    softcap: Option<T>,
    causal_offset: Option<isize>,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    val_dim: usize,
}

mod flash;
use flash::*;

mod naive;
pub use naive::*;

// Tests kept here exercise flex-specific internals: direct calls into
// `attention_flash` / `attention_naive` (the public `attention()` dispatcher
// routes small shapes to naive so flash-path coverage requires a direct
// call), the `broadcast_attn_mask_bias` helper's rank/dim validation,
// flex-internal tiling and online-softmax correction paths, and
// dtype-specific kernels (f16/f64). Generic attention semantics (causal,
// custom scale, softcap, cross-attention, bool mask, additive bias,
// multi-batch/multi-head, single-element) live in
// crates/ruda-backend-tests/tests/tensor/float/module/attention.rs, which
// exercises every backend. When adding new tests, keep them here only if
// they probe flex internals; otherwise add them to that suite.
#[cfg(test)]
mod tests;
