use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

/// Maximum `2 * max_target_len + 1` the kernel supports. The alpha/beta state is
/// held in shared memory as two f32 buffers of this size (active row + scratch),
/// so peak shared use at full capacity is `2 * 8192 * 4 = 64 KB`. Apple Metal
/// caps shared memory at 32 KB per block, so the launch site sizes the buffer to
/// the actual per-batch `max_l_prime`; this constant is only the kernel-side
/// upper bound. Inputs exceeding it panic rather than silently degrade.
pub(super) const SHARED_ALPHA_CAPACITY: u32 = 8192;

/// Class label at position `s` of the blank-inserted label sequence `l'`.
/// Odd `s` reads the underlying target at index `(s-1)/2`; even `s` is a blank.
#[ruda]
pub(super) fn l_prime_class<I: Numeric>(
    s: usize,
    targets: &Tensor<I>,
    n: usize,
    tgt_n: usize,
    tgt_s: usize,
    blank: usize,
) -> usize {
    if s % 2 == 1 {
        u32::cast_from(targets[n * tgt_n + ((s - 1) / 2) * tgt_s]) as usize
    } else {
        blank
    }
}

/// Numerically stable `log(exp(a) + exp(b))` with a sentinel short-circuit.
/// When `max(a, b) < unreachable_threshold`, returns `max(a, b)` directly so
/// the sentinel value doesn't drift upward each recursion step when both
/// inputs sit at the `-6e4` floor.
///
/// The threshold's magnitude is forced by f16: the sentinel can't go below
/// `-65504` (f16 max magnitude), so it's `-6e4`, and the threshold has to sit
/// above the sentinel but below any plausible legit alpha value, leaving a
/// narrow band around `-1e4`. On sufficiently long sequences where legit
/// alpha values naturally drop below `-1e4` (roughly `T * log(1/C) < -1e4`),
/// reachable states get misclassified as unreachable. Mitigation is a
/// WGSL-only path with a smaller sentinel; WGSL spec 8.7 lets implementations
/// replace runtime `1/0` with zero, so `-inf` can't be synthesized reliably.
#[ruda]
pub(super) fn log_sum_exp2<F: Float>(a: F, b: F, unreachable_threshold: F, one: F) -> F {
    let mut mx = a;
    let mut mn = b;
    if b > a {
        mx = b;
        mn = a;
    }
    if mx < unreachable_threshold {
        mx
    } else {
        mx + (one + (mn - mx).exp()).ln()
    }
}

/// Single alpha (or beta) recurrence step. `near`, `near_m1`, `near_m2` are
/// the three values from the previous time row (alpha: `t-1`; beta: `t+1`).
/// `log_p` is the emission log-prob at the current `(t, l'[s])` and
/// `skip_allowed` toggles the 2-position skip transition.
#[ruda]
pub(super) fn recurrence_step<F: Float>(
    near: F,
    near_m1: F,
    near_m2: F,
    log_p: F,
    skip_allowed: bool,
    unreachable_threshold: F,
    one: F,
) -> F {
    let lse_01 = log_sum_exp2::<F>(near, near_m1, unreachable_threshold, one);
    let combined = if skip_allowed {
        log_sum_exp2::<F>(lse_01, near_m2, unreachable_threshold, one)
    } else {
        lse_01
    };
    log_p + combined
}

/// Final `-log(alpha_last_blank + alpha_last_label)` reduction. Synthesizes a
/// true `+inf` via `exp()` overflow when both final alphas are at the sentinel
/// (the target is unreachable), so downstream `zero_infinity` logic can detect
/// it via `is_inf`. Builds the overflow arithmetically from a runtime-dependent
/// value (`target_len`, guaranteed >= 1 here) to keep WGSL's comptime-overflow
/// validator quiet.
#[ruda]
pub(super) fn finalize_nll<F: Float>(
    last_blank: F,
    last_label: F,
    target_len: usize,
    unreachable_threshold: F,
    one: F,
) -> F {
    let mut mx = last_blank;
    let mut mn = last_label;
    if last_label > last_blank {
        mx = last_label;
        mn = last_blank;
    }
    if mx < unreachable_threshold {
        (F::new(1000.0_f32) * F::cast_from(target_len as u32)).exp()
    } else {
        F::new(0.0) - (mx + (one + (mn - mx).exp()).ln())
    }
}

/// Value to emit when `input_len == 0`. `target_len == 0` is the only case
/// with a valid alignment (P(empty | empty) = 1, nll = 0); otherwise the
/// target is unreachable and the output is `+inf` synthesized via overflow.
#[ruda]
pub(super) fn empty_input_nll<F: Float>(target_len: usize) -> F {
    if target_len == 0 {
        F::new(0.0)
    } else {
        (F::new(1000.0_f32) * F::cast_from(target_len as u32)).exp()
    }
}

