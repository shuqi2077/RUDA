use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{Runtime, prelude::*};
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, contiguous::into_contiguous};
use ruda_core::tensor::{Shape, TensorMetadata};
use super::common::{SHARED_ALPHA_CAPACITY, empty_input_nll, finalize_nll, l_prime_class, recurrence_step};

/// Fused CTC alpha + beta recursion kernel.
///
/// Runs the full forward alpha recursion and reverse beta recursion for one
/// batch element per ruda, reusing the same shared-memory layout twice.
/// Writes `alpha_out[T, N, 2S+1]`, `beta_out[T, N, 2S+1]` and the per-sample
/// negative log-likelihood `nll_out[N]`. The three outputs are everything the
/// default CTC gradient-composition helper needs, so the caller can finish the
/// backward pass with a handful of element-wise tensor ops.
///
/// The alpha phase is identical to `ctc_loss_kernel` except it additionally
/// publishes each row to global memory. The beta phase mirrors it in reverse:
/// initialize at `t = input_len - 1` from `log_probs[t, l'[s]]` at the two
/// boundary `s` positions, then step backward reading `beta[t+1, s]`,
/// `beta[t+1, s+1]`, and (when the skip transition is allowed) `beta[t+1, s+2]`.
#[ruda(launch)]
fn ctc_alpha_beta_kernel<F: Float, I: Numeric>(
    log_probs: &Tensor<F>,      // [T, N, C]
    targets: &Tensor<I>,        // [N, S_max]
    input_lengths: &Tensor<I>,  // [N]
    target_lengths: &Tensor<I>, // [N]
    alpha_out: &mut Tensor<F>,  // [T, N, 2S+1]
    beta_out: &mut Tensor<F>,   // [T, N, 2S+1]
    nll_out: &mut Tensor<F>,    // [N]
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

    // Empty input: alpha_out and beta_out stay at the host-side -inf pre-fill.
    // Emit the semantically correct nll (0 for target_len=0, +inf otherwise).
    if input_len == 0 {
        if UNIT_POS_X == 0 {
            nll_out[n] = empty_input_nll::<F>(target_len);
        }
        terminate!();
    }

    let lp_t = log_probs.stride(0);
    let lp_n = log_probs.stride(1);
    let lp_c = log_probs.stride(2);
    let tgt_n = targets.stride(0);
    let tgt_s = targets.stride(1);
    let ao_t = alpha_out.stride(0);
    let ao_n = alpha_out.stride(1);
    let ao_s = alpha_out.stride(2);
    let bo_t = beta_out.stride(0);
    let bo_n = beta_out.stride(1);
    let bo_s = beta_out.stride(2);

    // Shared memory layout: [0..alpha_cap] is the active row; [alpha_cap..2*alpha_cap]
    // is scratch for the next row. Same layout is reused for alpha and beta. Beta
    // reads are guarded by `s + 1 < l_prime_len` / `s + 2 < l_prime_len`, so the
    // residual alpha values sitting in the active row between phases are never
    // observed by beta (its boundary init overwrites every slot it reads).
    let mut state = SharedMemory::<F>::new(2 * alpha_cap);
    // Sentinel for unreachable states. See ctc_loss_kernel for the full
    // rationale: f16's 65504 magnitude cap forces the -6e4 floor, WGSL
    // rejects f32(-inf) literals, and the threshold catches sentinel drift.
    let neg_inf = F::new(-6.0e4_f32);
    let unreachable_threshold = F::new(-1.0e4_f32);
    let one = F::new(1.0);

    // Alpha phase (forward).
    //
    // Initialize alpha at t = 0 for s < l_prime_len. Positions beyond
    // l_prime_len are never read by the recurrence, so they don't need
    // to be touched in shared memory; and they stay at the host-side -inf
    // pre-fill in alpha_out.
    let mut s = UNIT_POS_X as usize;
    while s < l_prime_len {
        let mut init = neg_inf;
        if s == 0 {
            init = log_probs[n * lp_n + blank_u * lp_c];
        } else if s == 1 {
            let l1 = u32::cast_from(targets[n * tgt_n]) as usize;
            init = log_probs[n * lp_n + l1 * lp_c];
        }
        state[s] = init;
        alpha_out[n * ao_n + s * ao_s] = init;
        s += ruda_dim;
    }
    sync_ruda();

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

            let a_s = state[s];
            let mut a_s_m1 = neg_inf;
            if s >= 1 {
                a_s_m1 = state[s - 1];
            }
            let mut a_s_m2 = neg_inf;
            if s >= 2 {
                a_s_m2 = state[s - 2];
            }

            state[alpha_cap + s] = recurrence_step::<F>(
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

        let mut s = UNIT_POS_X as usize;
        while s < l_prime_len {
            state[s] = state[alpha_cap + s];
            alpha_out[t * ao_t + n * ao_n + s * ao_s] = state[s];
            s += ruda_dim;
        }
        sync_ruda();
    }

    if UNIT_POS_X == 0 {
        let last_blank = state[2 * target_len];
        // See ctc_loss_kernel: -inf sentinel keeps log_sum_exp correct for target_len = 0.
        let mut last_label = neg_inf;
        if target_len > 0 {
            last_label = state[2 * target_len - 1];
        }
        nll_out[n] = finalize_nll::<F>(
            last_blank,
            last_label,
            target_len,
            unreachable_threshold,
            one,
        );
    }

    // Fence thread 0's read of state[2*target_len] / state[2*target_len - 1]
    // against the beta boundary init, which writes those same positions.
    sync_ruda();

    // Beta phase (reverse).
    //
    // Boundary initialization at t = input_len - 1: set beta[s] = log_probs[t, l'[s]]
    // at s = 2*target_len, and when target_len > 0 also at s = 2*target_len - 1.
    // All other s positions in range get -inf.
    let t_last = input_len - 1;
    let mut s = UNIT_POS_X as usize;
    while s < l_prime_len {
        let is_last_blank = s == 2 * target_len;
        let is_last_label = target_len > 0 && s == 2 * target_len - 1;
        let mut init = neg_inf;
        if is_last_blank || is_last_label {
            let l_class = l_prime_class::<I>(s, targets, n, tgt_n, tgt_s, blank_u);
            init = log_probs[t_last * lp_t + n * lp_n + l_class * lp_c];
        }
        state[s] = init;
        beta_out[t_last * bo_t + n * bo_n + s * bo_s] = init;
        s += ruda_dim;
    }
    sync_ruda();

    // Step back from t = input_len - 2 down to t = 0.
    for t_rev in 1..input_len {
        let t = input_len - 1 - t_rev;

        let mut s = UNIT_POS_X as usize;
        while s < l_prime_len {
            let l_class = l_prime_class::<I>(s, targets, n, tgt_n, tgt_s, blank_u);
            let log_p = log_probs[t * lp_t + n * lp_n + l_class * lp_c];

            let l_class_p2 = if s + 2 < l_prime_len {
                l_prime_class::<I>(s + 2, targets, n, tgt_n, tgt_s, blank_u)
            } else {
                blank_u
            };
            let skip_allowed = s + 2 < l_prime_len && l_class != blank_u && l_class != l_class_p2;

            let b_s = state[s];
            let mut b_s_p1 = neg_inf;
            if s + 1 < l_prime_len {
                b_s_p1 = state[s + 1];
            }
            let mut b_s_p2 = neg_inf;
            if s + 2 < l_prime_len {
                b_s_p2 = state[s + 2];
            }

            state[alpha_cap + s] = recurrence_step::<F>(
                b_s,
                b_s_p1,
                b_s_p2,
                log_p,
                skip_allowed,
                unreachable_threshold,
                one,
            );
            s += ruda_dim;
        }
        sync_ruda();

        let mut s = UNIT_POS_X as usize;
        while s < l_prime_len {
            state[s] = state[alpha_cap + s];
            beta_out[t * bo_t + n * bo_n + s * bo_s] = state[s];
            s += ruda_dim;
        }
        sync_ruda();
    }
}

/// Host entry point for the fused alpha + beta + nll kernel.
///
/// Returns `(log_alpha_full, log_beta_full, nll)` with shapes
/// `([T, N, 2S+1], [T, N, 2S+1], [N])`. Positions outside the valid
/// `(t < input_length, s < 2*target_length+1)` rectangle hold the
/// pre-fill value `-inf`, matching the default backend's convention.
///
/// Panics if `2 * max_target_len + 1` exceeds `SHARED_ALPHA_CAPACITY`.
pub fn ctc_alpha_beta<R: Runtime>(
    log_probs: RudaTensor<R>,
    targets: RudaTensor<R>,
    input_lengths: RudaTensor<R>,
    target_lengths: RudaTensor<R>,
    blank: usize,
) -> (RudaTensor<R>, RudaTensor<R>, RudaTensor<R>) {
    // Manual stride indexing below requires a contiguous physical layout;
    // fusion-produced tensors may arrive with layouts that break that
    // assumption. No-op when already contiguous.
    let log_probs = into_contiguous(log_probs);
    let targets = into_contiguous(targets);
    let input_lengths = into_contiguous(input_lengths);
    let target_lengths = into_contiguous(target_lengths);

    let log_probs_shape = log_probs.shape();
    let [max_input_length, batch_size, _c] = log_probs_shape.dims::<3>();
    let target_shape = targets.shape();
    let max_target_len = target_shape.dims::<2>()[1];
    let max_l_prime = 2 * max_target_len + 1;

    assert!(
        max_l_prime as u32 <= SHARED_ALPHA_CAPACITY,
        "ctc_loss_backward: 2 * max_target_len + 1 = {} exceeds the kernel's shared-memory \
         alpha capacity ({}). Reduce target length or raise SHARED_ALPHA_CAPACITY.",
        max_l_prime,
        SHARED_ALPHA_CAPACITY,
    );

    let hw_max = log_probs.client.properties().hardware.max_ruda_dim.0;
    let ruda_dim_x = (max_l_prime as u32).min(hw_max).min(256);

    let client = log_probs.client.clone();
    let device = log_probs.device.clone();
    let f_dtype = log_probs.dtype;
    let i_dtype = targets.dtype;

    // Pre-fill alpha/beta with -inf so positions the kernel doesn't touch
    // (s >= 2U+1, or t outside the valid range for an individual batch
    // element) are not read as stale zeros by the gradient composition.
    let shape_abt = Shape::new([max_input_length, batch_size, max_l_prime]);
    let neg_inf = InputScalar::new(f32::NEG_INFINITY, f_dtype);
    let alpha_out = ruda_kernel::tensor::initialization::full_device_dtype::<R>(
        client.clone(),
        shape_abt.clone(),
        device.clone(),
        neg_inf,
        f_dtype,
    );
    let beta_out = ruda_kernel::tensor::initialization::full_device_dtype::<R>(
        client.clone(),
        shape_abt,
        device.clone(),
        neg_inf,
        f_dtype,
    );
    let nll_out =
        empty_device_dtype::<R>(client.clone(), device, Shape::new([batch_size]), f_dtype);

    let ruda_count = RudaCount::Static(batch_size as u32, 1, 1);
    let ruda_dim = RudaDim::new_1d(ruda_dim_x);

    ctc_alpha_beta_kernel::launch::<R>(
        &client,
        ruda_count,
        ruda_dim,
        log_probs.into_tensor_arg(),
        targets.into_tensor_arg(),
        input_lengths.into_tensor_arg(),
        target_lengths.into_tensor_arg(),
        alpha_out.clone().into_tensor_arg(),
        beta_out.clone().into_tensor_arg(),
        nll_out.clone().into_tensor_arg(),
        blank as u32,
        max_l_prime as u32,
        [f_dtype.into(), i_dtype.into()],
    );

    (alpha_out, beta_out, nll_out)
}
