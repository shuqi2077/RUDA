mod kernel;
mod softmax;
mod rms;

pub use softmax::softmax_last_axis;
pub use rms::rms_norm;

use ruda_core::{device::Device, tensor::DType};
use ruda_kernel::{
    dsl::{
        Runtime,
        prelude::{RudaCount, RudaDim},
    },
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous},
};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizationError(pub &'static str);

impl fmt::Display for NormalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for NormalizationError {}

/// Last-axis LayerNorm with FP32 Welford statistics and affine arithmetic.
/// Input/output are F32, F16 or BF16; gamma and optional beta are F32 vectors.
pub fn layer_norm<R: Runtime>(
    input: RudaTensor<R>,
    gamma: RudaTensor<R>,
    beta: Option<RudaTensor<R>>,
    epsilon: f32,
) -> Result<RudaTensor<R>, NormalizationError> {
    let shape = input.meta.shape();
    let width = *shape
        .last()
        .ok_or(NormalizationError("LayerNorm input must have an axis"))?;
    let elements = shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    if width == 0
        || width > u32::MAX as usize
        || elements.is_none_or(|n| n > u32::MAX as usize)
        || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
        || input.qparams.is_some()
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err(NormalizationError(
            "invalid LayerNorm shape, dtype or epsilon",
        ));
    }
    for affine in std::iter::once(&gamma).chain(beta.iter()) {
        if affine.meta.shape()[..] != [width]
            || affine.dtype != DType::F32
            || affine.qparams.is_some()
            || affine.device.to_id() != input.device.to_id()
        {
            return Err(NormalizationError(
                "LayerNorm affine must be a same-device F32 feature vector",
            ));
        }
    }
    let plane = input.client.properties().hardware.plane_size_max;
    let maximum = input.client.properties().hardware.max_ruda_dim;
    if !plane.is_power_of_two() || plane > maximum.0 || maximum.1 < 4 {
        return Err(NormalizationError(
            "LayerNorm runtime cannot launch four planes",
        ));
    }
    let output = empty_device_contiguous_dtype(
        input.client.clone(),
        input.device.clone(),
        shape.clone().into(),
        input.dtype,
    );
    let rows = elements.unwrap() / width;
    if rows == 0 {
        return Ok(output);
    }
    let client = input.client.clone();
    let dtype = input.dtype;
    let has_beta = beta.is_some();
    let beta = beta.unwrap_or_else(|| gamma.clone());
    kernel::layer_norm::launch::<R>(
        &client,
        RudaCount::Static(rows as u32, 1, 1),
        RudaDim::new_2d(plane, 4),
        into_contiguous(input).into_array_arg(),
        into_contiguous(gamma).into_array_arg(),
        into_contiguous(beta).into_array_arg(),
        output.clone().into_array_arg(),
        width as u32,
        epsilon,
        has_beta,
        include_str!("kernel.rs").to_owned(),
        dtype.into(),
    );
    Ok(output)
}
