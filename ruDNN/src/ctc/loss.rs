use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{Runtime, prelude::*};
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, contiguous::into_contiguous};
use ruda_core::tensor::{Shape, TensorMetadata};
use super::common::{SHARED_ALPHA_CAPACITY, empty_input_nll, finalize_nll, l_prime_class, recurrence_step};

/// CTC alpha-recursion kernel.
///
/// Each ruda handles one batch element. `ruda_dim.x` is fixed at launch time
/// (capped to the runtime's hardware limit); each thread strides over the `s`
/// positions of the modified label sequence `l'` (length `2 * target_len + 1`),
/// covering arbitrary target lengths up to `SHARED_ALPHA_CAPACITY`. `alpha` is
/// kept in shared memory and the time loop runs sequentially inside the kernel,
/// using two `sync_ruda()` barriers per iteration: one to fence reads of
/// `alpha[t-1]` before any thread writes `alpha[t]`, one to publish the new row
/// before the next iteration. This collapses what would otherwise be roughly
/// `40 * T` host-side dispatches into a single kernel launch.
///
/// Impossible alignments use a large finite negative sentinel (`-6.0e4`)
/// rather than true `-inf`, because WGSL rejects `f32(-inf)` as an identifier
/// and f16's range caps at ~65504. The recurrence treats values below a
/// threshold (`-1.0e4`) as unreachable. If an entire sequence has no valid
/// alignment (e.g. `target_length > input_length`), the kernel synthesizes
/// `+inf` in the output so downstream `zero_infinity` masking in `ruda-nn`
/// can detect it via `is_inf`.
#[ruda(launch)]
fn ctc_loss_kernel<F: Float, I: Numeric>(
    log_probs: &Tensor<F>,      // [T, N, C]
    targets: &Tensor<I>,        // [N, S_max]
    input_lengths: &Tensor<I>,  // [N]
    target_lengths: &Tensor<I>, // [N]
    output: &mut Tensor<F>,     // [N]
    blank: u32,
    #[comptime] alpha_capacity: u32,
    #[define(F, I)] _dtypes: [StorageType; 2],
) {
    let n = RUDA_POS_X as usize;
    let ruda_dim = RUDA_DIM_X as usize;
    let alpha_cap = alpha_capacity as usize;
    let blank_u = blank as usize;

    let target_len = u32::cast_from(target_lengths[n]) as usize;
    let input_len = u32::cast_from(input_lengths[n]) as usize;
    let l_prime_len = 2 * target_len + 1;

    // Empty-input edge case: handled identically in both kernels to keep the
    // forward loss and the backward nll agreeing for this sample.
    if input_len == 0 {
        if UNIT_POS_X == 0 {
            output[n] = empty_input_nll::<F>(target_len);
        }
        terminate!();
    }

    let lp_t = log_probs.stride(0);
    let lp_n = log_probs.stride(1);
    let lp_c = log_probs.stride(2);
    let tgt_n = targets.stride(0);
    let tgt_s = targets.stride(1);

    // Two adjacent regions: alpha[0..alpha_cap] is the active row, the second
    // half [alpha_cap..2*alpha_cap] is a write scratch buffer that we copy back
    // to the active region after a sync. This avoids RAW hazards across stride
    // batches in the t-loop (a thread writing alpha[s] races with another
    // thread still reading alpha[s-1] or alpha[s-2] for its own s).
    let mut alpha = SharedMemory::<F>::new(2 * alpha_cap);
    // Sentinel for unreachable states. f16 caps at ~65504 magnitude, so we
    // can't go lower than `-6e4` without blowing past that range; WGSL also
    // rejects `f32(-inf)` as an identifier, so a real -inf literal isn't an
    // option anyway. On f32 the sentinel drifts slightly each recursion step
    // (log(2) per step when both log_sum_exp inputs sit at the sentinel),
    // which is why the recurrence compares against a threshold instead of
    // checking `== neg_inf`. See `log_sum_exp2` for the long-sequence caveat.
    let neg_inf = F::new(-6.0e4_f32);
    let unreachable_threshold = F::new(-1.0e4_f32);
    let one = F::new(1.0);

    // Initialize alpha at t = 0 for s < l_prime_len; positions beyond that
    // are never read by the recurrence (s < l_prime_len in every read) so
    // they don't need to be touched.
    let mut s = UNIT_POS_X as usize;
    while s < l_prime_len {
        let mut init = neg_inf;
        if s == 0 {
            init = log_probs[n * lp_n + blank_u * lp_c];
        } else if s == 1 {
            let l1 = u32::cast_from(targets[n * tgt_n]) as usize;
            init = log_probs[n * lp_n + l1 * lp_c];
        }
        alpha[s] = init;
        s += ruda_dim;
    }
    sync_ruda();

    // Sequential time loop. Each iteration re-strides over s positions to
    // compute alpha[t, s] from alpha[t-1, *] and writes back to the same
    // shared memory after a full read fence.
    for t in 1..input_len {
        let mut s = UNIT_POS_X as usize;
        while s < l_prime_len {
            let l_class = l_prime_class::<I>(s, targets, n, tgt_n, tgt_s, blank_u);
            let log_p = log_probs[t * lp_t + n * lp_n + l_class * lp_c];

            let l_class_m2 = if s >= 2 {
                l_prime_class::<I>(s - 2, targets, n, tgt_n, tgt_s, blank_u)
            } else {
                blank_u
            };
            let skip_allowed = s >= 2 && l_class != blank_u && l_class != l_class_m2;

            let a_s = alpha[s];
            let mut a_s_m1 = neg_inf;
            if s >= 1 {
                a_s_m1 = alpha[s - 1];
            }
            let mut a_s_m2 = neg_inf;
            if s >= 2 {
                a_s_m2 = alpha[s - 2];
            }

            alpha[alpha_cap + s] = recurrence_step::<F>(
                a_s,
                a_s_m1,
                a_s_m2,
                log_p,
                skip_allowed,
                unreachable_threshold,
                one,
            );
            s += ruda_dim;
        }
        sync_ruda();

        // Second pass: copy scratch back into the active alpha slots.
        let mut s = UNIT_POS_X as usize;
        while s < l_prime_len {
            alpha[s] = alpha[alpha_cap + s];
            s += ruda_dim;
        }
        sync_ruda();
    }

    // Reduce: only thread 0 writes the output for this batch element.
    if UNIT_POS_X == 0 {
        let last_blank = alpha[2 * target_len];
        // Guard target_len = 0: index 2*0 - 1 underflows. Use -inf so
        // log_sum_exp(last_blank, -inf) = last_blank (log_sum_exp(x, x) = x+ln2
        // would be wrong here).
        let mut last_label = neg_inf;
        if target_len > 0 {
            last_label = alpha[2 * target_len - 1];
        }
        output[n] = finalize_nll::<F>(
            last_blank,
            last_label,
            target_len,
            unreachable_threshold,
            one,
        );
    }
}

/// Fused CTC loss for ruda-tensor-device. Single kernel launch covers the entire
/// alpha recursion across all timesteps.
///
/// Panics if `2 * max_target_len + 1` exceeds `SHARED_ALPHA_CAPACITY` (8192).
pub fn ctc_loss<R: Runtime>(
    log_probs: RudaTensor<R>,
    targets: RudaTensor<R>,
    input_lengths: RudaTensor<R>,
    target_lengths: RudaTensor<R>,
    blank: usize,
) -> RudaTensor<R> {
    // Manual stride indexing below requires a contiguous physical layout;
    // fusion-produced tensors may arrive with layouts that break that
    // assumption. No-op when already contiguous.
    let log_probs = into_contiguous(log_probs);
    let targets = into_contiguous(targets);
    let input_lengths = into_contiguous(input_lengths);
    let target_lengths = into_contiguous(target_lengths);

    let log_probs_shape = log_probs.shape();
    let [_t, batch_size, _c] = log_probs_shape.dims::<3>();
    let target_shape = targets.shape();
    let max_target_len = target_shape.dims::<2>()[1];
    let max_l_prime = 2 * max_target_len + 1;

    assert!(
        max_l_prime as u32 <= SHARED_ALPHA_CAPACITY,
        "ctc_loss: 2 * max_target_len + 1 = {} exceeds the kernel's shared-memory \
         alpha capacity ({}). Reduce target length or raise SHARED_ALPHA_CAPACITY.",
        max_l_prime,
        SHARED_ALPHA_CAPACITY,
    );

    // Pick a thread count that fits the runtime's per-ruda limit. We don't
    // need one thread per s position - threads stride over s.
    let hw_max = log_probs.client.properties().hardware.max_ruda_dim.0;
    let ruda_dim_x = (max_l_prime as u32).min(hw_max).min(256);

    let client = log_probs.client.clone();
    let device = log_probs.device.clone();
    let f_dtype = log_probs.dtype;
    let i_dtype = targets.dtype;
    let output = empty_device_dtype::<R>(client.clone(), device, Shape::new([batch_size]), f_dtype);

    let ruda_count = RudaCount::Static(batch_size as u32, 1, 1);
    let ruda_dim = RudaDim::new_1d(ruda_dim_x);

    // Pass the actual max_l_prime (not the static capacity) so shared memory
    // is sized to what we need. Metal limits threadgroup memory to 32 KB;
    // allocating 2 * 8192 * sizeof(f32) = 64 KB would silently corrupt on
    // Apple GPUs. Different max_l_prime values trigger separate kernel
    // compilations (it's a comptime param), but that's fine: target lengths
    // are stable within a dataset.
    ctc_loss_kernel::launch::<R>(
        &client,
        ruda_count,
        ruda_dim,
        log_probs.into_tensor_arg(),
        targets.into_tensor_arg(),
        input_lengths.into_tensor_arg(),
        target_lengths.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        blank as u32,
        max_l_prime as u32,
        [f_dtype.into(), i_dtype.into()],
    );

    output
}
