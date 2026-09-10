//! Inverse real-valued FFT with an intra-ruda-parallel radix-2 kernel.

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
        rfft::SHARED_MEM_CAP,
        rfft_large::irfft_large_launch,
    },
    layout::BatchSignalLayout,
};

const MAX_UNITS_PER_RUDA: usize = 256;

/// Inverse Real-valued Fast Fourier Transform.
pub fn irfft<R: Runtime>(
    spectrum_re: TensorHandle<R>,
    spectrum_im: TensorHandle<R>,
    dim: usize,
    dtype: StorageType,
) -> TensorHandle<R> {
    assert!(
        spectrum_re.shape() == spectrum_im.shape(),
        "Spectrum's real and imaginary parts should be the same shape, got {:?} and {:?}",
        spectrum_re.shape(),
        spectrum_im.shape()
    );

    let client = <R as Runtime>::client(&Default::default());

    let mut signal_shape = spectrum_re.shape().clone();
    signal_shape[dim] = (spectrum_re.shape()[dim] - 1) * 2;
    let num_elems = signal_shape.iter().product::<usize>();
    let signal = TensorHandle::new_contiguous(
        signal_shape.clone(),
        client.empty(num_elems * dtype.size()),
        dtype,
    );

    irfft_launch::<R>(
        &client,
        spectrum_re.binding(),
        spectrum_im.binding(),
        signal.clone().binding(),
        dim,
        dtype,
    )
    .unwrap();

    signal
}

/// Launches the IRFFT kernel.
pub fn irfft_launch<R: Runtime>(
    client: &ComputeClient<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    signal: TensorBinding<R>,
    dim: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let spec_bins = spectrum_re.shape[dim];
    irfft_launch_padded::<R>(
        client,
        spectrum_re,
        spectrum_im,
        signal,
        dim,
        spec_bins,
        dtype,
    )
}

/// Launches the IRFFT kernel while treating bins at `spec_bins..n_freq` as zero.
pub fn irfft_launch_padded<R: Runtime>(
    client: &ComputeClient<R>,
    spectrum_re: TensorBinding<R>,
    spectrum_im: TensorBinding<R>,
    signal: TensorBinding<R>,
    dim: usize,
    spec_bins: usize,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    assert!(
        spectrum_re.shape == spectrum_im.shape,
        "spectrum real and imaginary shapes must match"
    );
    assert!(dim < signal.shape.len(), "dim must be in bounds");

    let n_fft = signal.shape[dim];
    assert!(n_fft.is_power_of_two(), "IRFFT requires power-of-2 length");
    assert!(n_fft >= 2, "IRFFT requires n_fft >= 2");
    let n_freq = n_fft / 2 + 1;
    assert!(
        spec_bins <= spectrum_re.shape[dim],
        "spec_bins ({spec_bins}) must be <= spectrum dimension ({})",
        spectrum_re.shape[dim]
    );
    assert!(spec_bins >= 1, "spec_bins must be >= 1");
    assert!(
        spec_bins <= n_freq,
        "spec_bins ({spec_bins}) must be <= n_fft / 2 + 1 ({n_freq})"
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
        return irfft_large_launch::<R>(
            client,
            spectrum_re,
            spectrum_im,
            signal,
            dim,
            spec_bins,
            dtype,
        );
    }

    let log2_n = n_fft.trailing_zeros() as usize;
    let threads_per_ruda = (n_fft / 2).clamp(1, MAX_UNITS_PER_RUDA);

    let ruda_dim = RudaDim::new_1d(threads_per_ruda as u32);
    let ruda_count = ruda_kernel::dsl::calculate_ruda_count_elemwise(client, count, RudaDim::new_single());

    irfft_kernel::launch::<f32, R>(
        client,
        ruda_count,
        ruda_dim,
        spectrum_re.into_tensor_arg(),
        spectrum_im.into_tensor_arg(),
        signal.into_tensor_arg(),
        count as u32,
        spec_bins as u32,
        n_fft,
        log2_n,
        threads_per_ruda,
        dim,
    );
    Ok(())
}

mod kernels;
use kernels::*;
