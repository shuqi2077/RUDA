//! Complex FFT used internally by the large-`n_fft` real-FFT paths.
//!
//! Two flavours live here:
//!
//! * A shared-memory kernel (`cfft_kernel`) for any size up to
//!   [`MAX_SHARED_N_FFT`]. Structurally identical to `rfft_kernel` /
//!   `irfft_kernel` except the imaginary input is read rather than zeroed
//!   and all `N` bins are written out (no Hermitian truncation).
//! * A four-step Cooley-Tukey orchestrator for `N > MAX_SHARED_N_FFT` that
//!   factors `N = N1 * N2` with both factors `<= MAX_SHARED_N_FFT`. Each
//!   sub-FFT reuses the shared-memory butterfly via a dedicated "strided /
//!   twiddled" radix kernel (`cfft_four_step_radix_kernel`). A single
//!   transpose kernel converts the (N1, N2) internal layout to natural
//!   linear bin order on the way out.
//!
//! The public API of this module is the single
//! [`cfft_launch_any_size`] function which picks the right path.
//!
//! The caller is responsible for allocating any scratch buffers; see
//! `rfft_large` for the allocator shape contract.

use ruda_kernel::dsl as kernel_dsl;
use core::f32::consts::PI;

use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::AsView as _;
use ruda_kernel::library::tensor::AsViewExpand;
use ruda_kernel::library::tensor::AsViewMut as _;
use ruda_kernel::library::tensor::AsViewMutExpand;
use ruda_kernel::library::tensor::TensorHandle;

use crate::{
    fft::{
        FftMode,
        fft_parallel::{bit_reverse, fft_butterfly_parallel},
    },
    layout::BatchSignalLayout,
};

/// Portable size limit for the single-pass shared-memory path. The kernel
/// allocates two `f32` shared buffers of length `n_fft`, so `n_fft = 4096`
/// uses 2 * 4096 * 4 bytes = 32 KiB. Larger sizes use the packed-real /
/// four-step path instead of relying on backend-specific larger workgroup
/// memory limits.
pub(crate) const MAX_SHARED_N_FFT: usize = 4096;

/// Portable cap on the number of units in one ruda. Larger FFTs still cover
/// all bins by having each unit process multiple indices.
const MAX_UNITS_PER_RUDA: usize = 256;

pub(crate) struct CfftBindings<R: Runtime> {
    pub(crate) input_re: TensorBinding<R>,
    pub(crate) input_im: TensorBinding<R>,
    pub(crate) output_re: TensorBinding<R>,
    pub(crate) output_im: TensorBinding<R>,
}

#[derive(Clone, Copy)]
struct CfftPlan {
    dim: usize,
    count: usize,
    n_fft: usize,
    fft_mode: FftMode,
}

/// Factor `n_fft = N1 * N2` for the four-step FFT. Both factors are powers
/// of two, both `<= MAX_SHARED_N_FFT`, and the split is as balanced as
/// possible.
pub(crate) fn factor_four_step(n_fft: usize) -> (usize, usize) {
    assert!(
        n_fft.is_power_of_two(),
        "four-step needs power-of-two n_fft"
    );
    let log2_n = n_fft.trailing_zeros() as usize;
    let max_log2 = MAX_SHARED_N_FFT.trailing_zeros() as usize;
    // Balanced split, then push each factor up to the shared-mem cap if the
    // other factor would otherwise exceed it.
    let log2_n1 = log2_n / 2;
    let log2_n2 = log2_n - log2_n1;
    let (log2_n1, log2_n2) = if log2_n2 > max_log2 {
        (log2_n - max_log2, max_log2)
    } else {
        (log2_n1, log2_n2)
    };
    assert!(
        log2_n1 <= max_log2 && log2_n2 <= max_log2,
        "four-step cannot handle n_fft = {n_fft} with MAX_SHARED_N_FFT = {MAX_SHARED_N_FFT}",
    );
    (1 << log2_n1, 1 << log2_n2)
}

/// Entry point: complex FFT of `input` into `output`, along `dim`.
/// `input` and `output` must be contiguous and have identical shape. The
/// caller may safely pass the same buffer for input and output (aliasing is
/// allowed for the small path; the large path does its own scratch
/// management).
pub(crate) fn cfft_launch_any_size<R: Runtime>(
    client: &ComputeClient<R>,
    bindings: CfftBindings<R>,
    dim: usize,
    dtype: StorageType,
    fft_mode: FftMode,
) -> Result<(), LaunchError> {
    let n_fft = bindings.input_re.shape[dim];
    assert!(n_fft.is_power_of_two(), "cfft needs power-of-two n_fft");
    assert!(n_fft >= 2);
    let count: usize = bindings
        .input_re
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, e)| *e)
        .product();
    if count == 0 {
        return Ok(());
    }
    let plan = CfftPlan {
        dim,
        count,
        n_fft,
        fft_mode,
    };

    if n_fft <= MAX_SHARED_N_FFT {
        cfft_shared_launch::<R>(client, bindings, plan)
    } else {
        cfft_four_step_launch::<R>(client, bindings, dtype, plan)
    }
}

fn cfft_shared_launch<R: Runtime>(
    client: &ComputeClient<R>,
    bindings: CfftBindings<R>,
    plan: CfftPlan,
) -> Result<(), LaunchError> {
    let log2_n = plan.n_fft.trailing_zeros() as usize;
    let threads_per_ruda = (plan.n_fft / 2).clamp(1, MAX_UNITS_PER_RUDA);
    let ruda_dim = RudaDim::new_1d(threads_per_ruda as u32);
    let ruda_count =
        ruda_kernel::dsl::calculate_ruda_count_elemwise(client, plan.count, RudaDim::new_single());

    cfft_shared_kernel::launch::<f32, R>(
        client,
        ruda_count,
        ruda_dim,
        bindings.input_re.into_tensor_arg(),
        bindings.input_im.into_tensor_arg(),
        bindings.output_re.into_tensor_arg(),
        bindings.output_im.into_tensor_arg(),
        plan.count as u32,
        plan.n_fft,
        log2_n,
        threads_per_ruda,
        plan.dim,
        plan.fft_mode,
    );
    Ok(())
}

// --- Four-step path ----------------------------------------------------

/// Four-step complex FFT for `n_fft > MAX_SHARED_N_FFT`.
///
/// Layout convention: each window's `n_fft` axis is viewed as
/// `(N1, N2)` row-major with the flat index `n = n1 * N2 + n2`. After the
/// four-step pipeline the output has `X[k1 + k2 * N1]` at flat index
/// `k2 * N1 + k1` (natural linear order over `k`). Caller's `output_re` /
/// `output_im` tensors receive this natural order.
fn cfft_four_step_launch<R: Runtime>(
    client: &ComputeClient<R>,
    bindings: CfftBindings<R>,
    dtype: StorageType,
    plan: CfftPlan,
) -> Result<(), LaunchError> {
    let (n1, n2) = factor_four_step(plan.n_fft);

    // Scratch buffer, same shape as input. Two passes ping-pong through
    // scratch and output; the transpose at the end lands in `output`.
    let scratch_shape: Vec<usize> = bindings.input_re.shape.to_vec();
    let elems: usize = scratch_shape.iter().product();
    let scratch_re = TensorHandle::<R>::new_contiguous(
        scratch_shape.clone(),
        client.empty(elems * dtype.size()),
        dtype,
    );
    let scratch_im = TensorHandle::<R>::new_contiguous(
        scratch_shape.clone(),
        client.empty(elems * dtype.size()),
        dtype,
    );

    // Step 1: strided FFT_{N1} along the n1 axis of (N1, N2). One ruda per
    // (window, n2). Reads from `input_*`, writes to `scratch_*` with fused
    // twiddle multiplication by W_N^{k1 * n2} for the inter-stage factor.
    {
        let threads_per_ruda = (n1 / 2).clamp(1, MAX_UNITS_PER_RUDA);
        let log2_n1 = n1.trailing_zeros() as usize;
        let ruda_dim = RudaDim::new_1d(threads_per_ruda as u32);
        let ruda_count =
            ruda_kernel::dsl::calculate_ruda_count_elemwise(client, plan.count * n2, RudaDim::new_single());

        cfft_four_step_radix1_kernel::launch::<f32, R>(
            client,
            ruda_count,
            ruda_dim,
            bindings.input_re.into_tensor_arg(),
            bindings.input_im.into_tensor_arg(),
            scratch_re.clone().binding().into_tensor_arg(),
            scratch_im.clone().binding().into_tensor_arg(),
            (plan.count * n2) as u32,
            n1,
            n2,
            log2_n1,
            threads_per_ruda,
            plan.dim,
            plan.fft_mode,
        );
    }

    // Step 2: contiguous FFT_{N2} along the n2 axis of (N1, N2). One ruda
    // per (window, k1). Reads/writes scratch in place.
    {
        let threads_per_ruda = (n2 / 2).clamp(1, MAX_UNITS_PER_RUDA);
        let log2_n2 = n2.trailing_zeros() as usize;
        let ruda_dim = RudaDim::new_1d(threads_per_ruda as u32);
        let ruda_count =
            ruda_kernel::dsl::calculate_ruda_count_elemwise(client, plan.count * n1, RudaDim::new_single());

        cfft_four_step_radix2_kernel::launch::<f32, R>(
            client,
            ruda_count,
            ruda_dim,
            scratch_re.clone().binding().into_tensor_arg(),
            scratch_im.clone().binding().into_tensor_arg(),
            (plan.count * n1) as u32,
            n1,
            n2,
            log2_n2,
            threads_per_ruda,
            plan.dim,
            plan.fft_mode,
        );
    }

    // Step 3: transpose (N1, N2) -> (N2, N1). Writes natural-order output.
    {
        let total = plan.count * plan.n_fft;
        let ruda_dim = RudaDim::new_1d(256);
        let ruda_count = ruda_kernel::dsl::calculate_ruda_count_elemwise(client, total, ruda_dim);

        cfft_four_step_transpose_kernel::launch::<f32, R>(
            client,
            ruda_count,
            ruda_dim,
            scratch_re.binding().into_tensor_arg(),
            scratch_im.binding().into_tensor_arg(),
            bindings.output_re.into_tensor_arg(),
            bindings.output_im.into_tensor_arg(),
            total as u32,
            n1,
            n2,
            plan.dim,
        );
    }

    Ok(())
}

/// First four-step pass: FFT_{N1} along n1 (strided by N2 in the packed
/// `n_fft` axis), with a fused W_N^{k1 * n2} twiddle on the way out.
///
/// Grid: `count * N2` rudas. `RUDA_POS = window * N2 + n2`.

/// Second four-step pass: FFT_{N2} along n2 (contiguous), in place.
///
/// Grid: `count * N1` rudas. `RUDA_POS = window * N1 + k1`.

/// Transpose (N1, N2) -> (N2, N1) in each selected-axis window.
/// Converts four-step output `X'[k1, k2]` at flat `k1*N2 + k2` into natural
/// linear order `X[k]` at flat `k = k1 + k2*N1` (= `k2*N1 + k1`).
///
/// One thread per output element.
mod kernels;
use kernels::*;
