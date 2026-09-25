use ruda_core::tensor::{DType, Shape, TensorMetadata};
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, reshape::reshape};
use crate::{FftMode, RealFftPlan};

/// Legacy compatibility API: the requested signal length is rounded up to a
/// power of two. Use `rfft_exact` for an actual N-point DFT. Virtual padding
/// avoids allocating and copying a padded signal.
pub fn rfft<R: Runtime>(signal: RudaTensor<R>, dim: usize, n: Option<usize>)
    -> (RudaTensor<R>, RudaTensor<R>)
{
    forward(signal, dim, n, false)
}

/// Exact N-point real FFT, returning floor(N/2)+1 F32 frequency bins. A supplied
/// N truncates or virtually zero-pads the input, but does not change the DFT's
/// length. Non-power-of-two N uses the device Bluestein path.
pub fn rfft_exact<R: Runtime>(signal: RudaTensor<R>, dim: usize, n: Option<usize>)
    -> (RudaTensor<R>, RudaTensor<R>)
{
    forward(signal, dim, n, true)
}

fn forward<R: Runtime>(signal: RudaTensor<R>, dim: usize, n: Option<usize>, exact: bool)
    -> (RudaTensor<R>, RudaTensor<R>)
{
    assert_eq!(signal.dtype, DType::F32, "ruFFT device kernels currently require F32 storage");
    let shape = signal.shape();
    assert!(dim < shape.len(), "rfft: dimension out of bounds");
    let requested = n.unwrap_or(shape[dim]);
    assert!(requested > 0, "rfft: transform length must be positive");
    let length = if exact { requested } else {
        requested.checked_next_power_of_two().expect("rfft length overflow")
    };
    let used = requested.min(shape[dim]);
    let rank_one = shape.len() == 1;
    let (signal, dim) = if rank_one {
        (reshape(signal, Shape::new([1, shape[0]])), 1)
    } else { (signal, dim) };
    let mut plan = RealFftPlan::new(signal.client.clone(), length, FftMode::Forward)
        .unwrap_or_else(|e| panic!("rfft plan failed (requested={requested}, actual={length}): {e}"));
    let mut out_shape = signal.shape();
    out_shape[dim] = length / 2 + 1;
    let real = empty_device_dtype(signal.client.clone(), signal.device.clone(), out_shape.clone(), DType::F32);
    let imag = empty_device_dtype(signal.client.clone(), signal.device.clone(), out_shape, DType::F32);
    plan.forward(signal.binding(), real.clone().binding(), imag.clone().binding(), dim, used)
        .unwrap_or_else(|e| panic!("rfft launch failed (requested={requested}, actual={length}): {e}"));
    if rank_one {
        (reshape(real, Shape::new([length / 2 + 1])), reshape(imag, Shape::new([length / 2 + 1])))
    } else { (real, imag) }
}

/// Legacy inverse: compute the next-power-of-two inverse, then crop to N.
/// Prefer `irfft_exact` when the spectrum describes an actual N-point DFT.
pub fn irfft<R: Runtime>(real: RudaTensor<R>, imag: RudaTensor<R>, dim: usize, n: Option<usize>)
    -> RudaTensor<R>
{
    inverse(real, imag, dim, n, false)
}

/// Exact normalized N-point inverse real FFT. For odd N, pass Some(N): the
/// half-spectrum alone cannot distinguish an odd from an even original length.
pub fn irfft_exact<R: Runtime>(real: RudaTensor<R>, imag: RudaTensor<R>, dim: usize, n: Option<usize>)
    -> RudaTensor<R>
{
    inverse(real, imag, dim, n, true)
}

fn inverse<R: Runtime>(real: RudaTensor<R>, imag: RudaTensor<R>, dim: usize, n: Option<usize>, exact: bool)
    -> RudaTensor<R>
{
    assert_eq!(real.dtype, DType::F32, "ruFFT device kernels currently require F32 storage");
    assert_eq!(imag.dtype, real.dtype, "irfft: real and imaginary dtypes differ");
    assert_eq!(real.client.device_id(), imag.client.device_id(), "irfft: input devices differ");
    let shape = real.shape();
    assert_eq!(shape, imag.shape(), "irfft: real and imaginary shapes differ");
    assert!(dim < shape.len(), "irfft: dimension out of bounds");
    assert!(shape[dim] > 0, "irfft: spectrum must contain at least one bin");
    let inferred = (shape[dim] - 1).checked_mul(2).expect("irfft inferred length overflow");
    let requested = n.unwrap_or(inferred);
    assert!(requested > 0, "irfft: positive N is required (use Some(1) for one bin)");
    let length = if exact { requested } else {
        requested.checked_next_power_of_two().expect("irfft length overflow")
    };
    let used = shape[dim].min(length / 2 + 1);
    let rank_one = shape.len() == 1;
    let (real, imag, dim) = if rank_one {
        (reshape(real, Shape::new([1, shape[0]])), reshape(imag, Shape::new([1, shape[0]])), 1)
    } else { (real, imag, dim) };
    let mut plan = RealFftPlan::new(real.client.clone(), length, FftMode::Inverse)
        .unwrap_or_else(|e| panic!("irfft plan failed (requested={requested}, actual={length}): {e}"));
    let mut out_shape = real.shape();
    out_shape[dim] = length;
    let output = empty_device_dtype(real.client.clone(), real.device.clone(), out_shape.clone(), DType::F32);
    plan.inverse(real.binding(), imag.binding(), output.clone().binding(), dim, used)
        .unwrap_or_else(|e| panic!("irfft launch failed (requested={requested}, actual={length}): {e}"));
    // Cropping is part of the old API only. Neither path pads/copies spectra.
    let output = if length != requested {
        let ranges: Vec<_> = out_shape.iter().enumerate()
            .map(|(axis, &size)| 0..if axis == dim { requested } else { size }).collect();
        ruprim::indexing::slice(output, &ranges)
    } else { output };
    if rank_one { reshape(output, Shape::new([requested])) } else { output }
}
