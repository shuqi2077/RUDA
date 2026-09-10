//! Large-`n_fft` RFFT / IRFFT via the packed-real trick.
//!
//! Packed-real FFT turns an N-point real FFT into an M=N/2 point complex
//! FFT plus two elementwise passes. With the current shared-memory cutoff,
//! the large paths are:
//!
//! * N = 8192 → M = 4096 → fused shared-memory packed FFT (fast path).
//! * N = 16384 → M = 8192 → four-step cfft (still fast, no global
//!   ping-pong per stage).
//!
//! Compared to running a complex FFT with a zeroed imaginary input, the
//! packed form halves both FLOPs and memory traffic in the inner FFT. That
//! lets N = 8192 use a single shared-memory CFFT of size 4096 instead of the
//! four-step path.
//!
//! Forward `rfft` pipeline above the fused shared-memory limit:
//!   1. `rfft_pack_kernel` — pack real `x[0..N]` into complex
//!      `y[k] = x[2k] + i*x[2k+1]`, length M.
//!   2. `cfft_launch_any_size(FORWARD)` — complex FFT of `y`.
//!   3. `rfft_post_kernel` — recover the half-spectrum
//!      `X[0..N/2+1]` from `Y` using the Z_even / Z_odd split.
//!
//! Inverse `irfft` pipeline above the fused shared-memory limit:
//!   1. `irfft_pre_kernel` — rebuild the packed `Y` of length M from the
//!      half-spectrum `X[0..N/2+1]` (inverse of step 3 above).
//!   2. `cfft_launch_any_size(INVERSE)` — complex IFFT of `Y` into `y`.
//!      Note: butterfly output is unnormalised (sum, not mean); we fold
//!      the `1/M` factor into the unpack step.
//!   3. `irfft_unpack_kernel` — write `x[2k] = Re(y[k])`,
//!      `x[2k+1] = Im(y[k])`, applying the `1/M` normalisation.
//!
//! Invariants:
//! * `n_fft` power of two, `n_fft >= 4` (M >= 2 for the packed FFT).

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
        cfft::{CfftBindings, MAX_SHARED_N_FFT, cfft_launch_any_size},
        fft_parallel::{bit_reverse, fft_butterfly_parallel},
    },
    layout::BatchSignalLayout,
};

/// Forward large-`n_fft` RFFT. Shapes:
/// * `signal`: (..., <= n_fft) real.
/// * `spectrum_re`, `spectrum_im`: (..., n_fft/2 + 1) complex.
pub(crate) fn rfft_large_launch<R: Runtime>(
    client: &ComputeClient<R>,
    signal: TensorBinding<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    dim: usize,
    signal_len: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let n_fft = (spectrum_re.shape[dim] - 1) * 2;
    let m = n_fft / 2;
    let count: usize = signal
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, e)| *e)
        .product();

    if m <= MAX_SHARED_N_FFT {
        let threads = (m / 2).clamp(1, 256);
        let grid =
            ruda_kernel::dsl::calculate_ruda_count_elemwise(client, count, RudaDim::new_single());
        rfft_fused_kernel::launch::<f32, R>(
            client,
            grid,
            RudaDim::new_1d(threads as u32),
            signal.into_tensor_arg(),
            spectrum_re.into_tensor_arg(),
            spectrum_im.into_tensor_arg(),
            count as u32,
            signal_len as u32,
            n_fft,
            m,
            m.trailing_zeros() as usize,
            threads,
            dim,
        );
        return Ok(());
    }

    // Packed buffers of length M.
    let packed_shape: Vec<usize> = signal
        .shape
        .iter()
        .enumerate()
        .map(|(i, &s)| if i == dim { m } else { s })
        .collect();
    let packed_elems: usize = packed_shape.iter().product();
    let packed_re = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );
    let packed_im = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );

    // Step 1: pack x → y.
    {
        let ruda_dim = RudaDim::new_1d(256);
        let ruda_count = ruda_kernel::dsl::calculate_ruda_count_elemwise(client, count * m, ruda_dim);

        rfft_pack_kernel::launch::<f32, R>(
            client,
            ruda_count,
            ruda_dim,
            signal.into_tensor_arg(),
            packed_re.clone().binding().into_tensor_arg(),
            packed_im.clone().binding().into_tensor_arg(),
            (count * m) as u32,
            signal_len as u32,
            m,
            dim,
        );
    }

    // Step 2: Y = FFT_M(y), in-place on the packed buffers.
    cfft_launch_any_size::<R>(
        client,
        CfftBindings {
            input_re: packed_re.clone().binding(),
            input_im: packed_im.clone().binding(),
            output_re: packed_re.clone().binding(),
            output_im: packed_im.clone().binding(),
        },
        dim,
        dtype,
        FftMode::Forward,
    )?;

    // Step 3: recover half-spectrum X from Y.
    {
        let n_freq = m + 1;
        let ruda_dim = RudaDim::new_1d(256);
        let ruda_count = ruda_kernel::dsl::calculate_ruda_count_elemwise(client, count * n_freq, ruda_dim);

        rfft_post_kernel::launch::<f32, R>(
            client,
            ruda_count,
            ruda_dim,
            packed_re.binding().into_tensor_arg(),
            packed_im.binding().into_tensor_arg(),
            spectrum_re.into_tensor_arg(),
            spectrum_im.into_tensor_arg(),
            (count * n_freq) as u32,
            n_fft,
            m,
            dim,
        );
    }

    Ok(())
}

/// Inverse large-`n_fft` IRFFT. Shapes:
/// * `spectrum_re`, `spectrum_im`: (..., n_fft/2 + 1) complex.
/// * `signal`: (..., n_fft) real.
pub(crate) fn irfft_large_launch<R: Runtime>(
    client: &ComputeClient<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    signal: TensorBinding<R>,
    dim: usize,
    spec_bins: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let n_fft = signal.shape[dim];
    let m = n_fft / 2;
    let count: usize = signal
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, e)| *e)
        .product();

    if m <= MAX_SHARED_N_FFT {
        let threads = (m / 2).clamp(1, 256);
        let grid =
            ruda_kernel::dsl::calculate_ruda_count_elemwise(client, count, RudaDim::new_single());
        irfft_fused_kernel::launch::<f32, R>(
            client,
            grid,
            RudaDim::new_1d(threads as u32),
            spectrum_re.into_tensor_arg(),
            spectrum_im.into_tensor_arg(),
            signal.into_tensor_arg(),
            count as u32,
            spec_bins as u32,
            n_fft,
            m,
            m.trailing_zeros() as usize,
            threads,
            dim,
        );
        return Ok(());
    }

    let packed_shape: Vec<usize> = signal
        .shape
        .iter()
        .enumerate()
        .map(|(i, &s)| if i == dim { m } else { s })
        .collect();
    let packed_elems: usize = packed_shape.iter().product();
    let packed_in_re = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );
    let packed_in_im = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );
    let packed_out_re = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );
    let packed_out_im = TensorHandle::<R>::new_contiguous(
        packed_shape.clone(),
        client.empty(packed_elems * dtype.size()),
        dtype,
    );

    // Step 1: build packed Y from half-spectrum X.
    {
        let ruda_dim = RudaDim::new_1d(256);
        let ruda_count = ruda_kernel::dsl::calculate_ruda_count_elemwise(client, count * m, ruda_dim);

        irfft_pre_kernel::launch::<f32, R>(
            client,
            ruda_count,
            ruda_dim,
            spectrum_re.into_tensor_arg(),
            spectrum_im.into_tensor_arg(),
            packed_in_re.clone().binding().into_tensor_arg(),
            packed_in_im.clone().binding().into_tensor_arg(),
            (count * m) as u32,
            spec_bins as u32,
            n_fft,
            m,
            dim,
        );
    }

    // Step 2: y = IFFT_M(Y). Need a separate destination because
    // cfft_four_step_launch ping-pongs internally and may not support
    // aliasing. (Small-path cfft aliases fine but we keep one code path.)
    cfft_launch_any_size::<R>(
        client,
        CfftBindings {
            input_re: packed_in_re.binding(),
            input_im: packed_in_im.binding(),
            output_re: packed_out_re.clone().binding(),
            output_im: packed_out_im.clone().binding(),
        },
        dim,
        dtype,
        FftMode::Inverse,
    )?;

    // Step 3: unpack y into real x with the 1/M normalisation.
    {
        let ruda_dim = RudaDim::new_1d(256);
        let ruda_count = ruda_kernel::dsl::calculate_ruda_count_elemwise(client, count * m, ruda_dim);

        irfft_unpack_kernel::launch::<f32, R>(
            client,
            ruda_count,
            ruda_dim,
            packed_out_re.binding().into_tensor_arg(),
            packed_out_im.binding().into_tensor_arg(),
            signal.into_tensor_arg(),
            (count * m) as u32,
            m,
            dim,
        );
    }

    Ok(())
}

// --- pack / post / pre / unpack kernels --------------------------------

/// `y[k] = x[2k] + i * x[2k+1]`, one thread per `k`.

/// Recover `X[0..N/2+1]` from `Y[0..M]` for the packed-real forward path.
///
/// Let `A = Y[k]`, `B = conj(Y[M-k])` for 0 < k < M.
///   `Z_e[k] = (A + B) / 2`
///   `Z_o[k] = -i * (A - B) / 2`
///   `X[k] = Z_e[k] + W_N^k * Z_o[k]`
/// Edge cases:
///   `X[0] = Re(Y[0]) + Im(Y[0])`, `X[M] = Re(Y[0]) - Im(Y[0])` (both real).
///
/// One thread per output bin `k in [0, M+1)`.

/// Build packed `Y[0..M]` from half-spectrum `X[0..N/2+1]` for the
/// packed-real inverse path. Inverse of `rfft_post_kernel`.
///
/// For 0 < k < M:
///   `Z_e[k] = (X[k] + conj(X[M-k])) / 2`
///   `Z_o[k] = W_N^{-k} * (X[k] - conj(X[M-k])) / 2`
///   `Y[k] = Z_e[k] + i * Z_o[k]`
/// Edge case `k = 0`:
///   `Y[0] = (X[0] + X[M]) / 2  +  i * (X[0] - X[M]) / 2`.
///
/// One thread per packed bin `k in [0, M)`.

/// Unpack `y[k]` into real `x` with the 1/M normalisation folded in.
/// `x[2k] = Re(y[k]) / M`, `x[2k+1] = Im(y[k]) / M`. One thread per `k`.
mod kernels;
use kernels::*;
