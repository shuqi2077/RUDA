//! Real-valued FFT with an intra-ruda-parallel radix-2 Cooley-Tukey kernel.

use ruda_kernel::dsl as kernel_dsl;
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
        rfft_large::rfft_large_launch,
    },
    layout::BatchSignalLayout,
};

const MAX_UNITS_PER_RUDA: usize = 256;
pub(crate) const SHARED_MEM_CAP: usize = 4096;

/// Real-valued Fast Fourier Transform.
pub fn rfft<R: Runtime>(
    signal: TensorHandle<R>,
    dim: usize,
    dtype: StorageType,
) -> (TensorHandle<R>, TensorHandle<R>) {
    assert!(
        dim < signal.shape().len(),
        "dim must be between 0 and {}",
        signal.shape().len()
    );
    assert!(
        signal.shape()[dim].is_power_of_two(),
        "RFFT requires power-of-2 length"
    );
    let client = <R as Runtime>::client(&Default::default());

    let mut spectrum_shape = signal.shape().clone();
    spectrum_shape[dim] = signal.shape()[dim] / 2 + 1;

    let spectrum_re = TensorHandle::new_contiguous(
        spectrum_shape.clone(),
        client.empty(spectrum_shape.iter().product::<usize>() * dtype.size()),
        dtype,
    );

    let spectrum_im = TensorHandle::new_contiguous(
        spectrum_shape.clone(),
        client.empty(spectrum_shape.iter().product::<usize>() * dtype.size()),
        dtype,
    );

    rfft_launch::<R>(
        &client,
        signal.binding(),
        spectrum_re.clone().binding(),
        spectrum_im.clone().binding(),
        dim,
        dtype,
    )
    .unwrap();

    (spectrum_re, spectrum_im)
}

/// Launches the RFFT kernel.
pub fn rfft_launch<R: Runtime>(
    client: &ComputeClient<R>,
    signal: TensorBinding<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    dim: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let signal_len = signal.shape[dim];
    rfft_launch_padded::<R>(
        client,
        signal,
        spectrum_re,
        spectrum_im,
        dim,
        signal_len,
        dtype,
    )
}

/// Launches the RFFT kernel while treating samples at `signal_len..n_fft` as zero.
pub fn rfft_launch_padded<R: Runtime>(
    client: &ComputeClient<R>,
    signal: TensorBinding<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    dim: usize,
    signal_len: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    assert!(
        spectrum_re.shape == spectrum_im.shape,
        "spectrum real and imaginary shapes must match"
    );
    assert!(dim < signal.shape.len(), "dim must be in bounds");

    assert!(
        spectrum_re.shape[dim] >= 2,
        "RFFT spectrum dimension must contain at least DC and Nyquist bins"
    );
    let n_fft = (spectrum_re.shape[dim] - 1) * 2;
    assert!(n_fft.is_power_of_two(), "RFFT requires power-of-2 length");
    assert!(n_fft >= 2, "RFFT requires n_fft >= 2");
    assert!(
        signal_len <= signal.shape[dim],
        "signal_len ({signal_len}) must be <= signal dimension ({})",
        signal.shape[dim]
    );
    assert!(
        signal_len <= n_fft,
        "signal_len ({signal_len}) must be <= n_fft ({n_fft})"
    );

    let count: usize = signal
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, e)| *e)
        .product();
    if count == 0 {
        return Ok(());
    }

    if n_fft > SHARED_MEM_CAP {
        return rfft_large_launch::<R>(
            client,
            signal,
            spectrum_re,
            spectrum_im,
            dim,
            signal_len,
            dtype,
        );
    }

    let log2_n = n_fft.trailing_zeros() as usize;
    let threads_per_ruda = (n_fft / 2).clamp(1, MAX_UNITS_PER_RUDA);

    let ruda_dim = RudaDim::new_1d(threads_per_ruda as u32);
    let ruda_count = ruda_kernel::dsl::calculate_ruda_count_elemwise(client, count, RudaDim::new_single());

    rfft_kernel::launch::<f32, R>(
        client,
        ruda_count,
        ruda_dim,
        signal.into_tensor_arg(),
        spectrum_re.into_tensor_arg(),
        spectrum_im.into_tensor_arg(),
        count as u32,
        signal_len as u32,
        n_fft,
        log2_n,
        threads_per_ruda,
        dim,
    );
    Ok(())
}

mod kernels;
use kernels::*;
