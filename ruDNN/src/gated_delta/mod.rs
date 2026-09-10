mod kernel;
mod chunk;
mod chunk_kernel;

pub use chunk::chunk_gated_delta_rule;

use ruda_core::{
    device::Device,
    tensor::{DType, Shape},
};
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::RudaDim},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous},
};
use std::fmt;

pub struct GatedDeltaInput<R: Runtime> {
    pub query: RudaTensor<R>,
    pub key: RudaTensor<R>,
    pub value: RudaTensor<R>,
    pub beta: RudaTensor<R>,
    pub log_decay: RudaTensor<R>,
    pub initial_state: RudaTensor<R>,
    pub query_scale: f32,
}

pub struct GatedDeltaOutput<R: Runtime> {
    pub output: RudaTensor<R>,
    pub final_state: RudaTensor<R>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatedDeltaError(pub &'static str);

impl fmt::Display for GatedDeltaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for GatedDeltaError {}

/// Gated delta recurrence with FP32 state and accumulation. Q/K are supplied
/// after any model-specific normalization, in [batch, heads, sequence, key_dim].
/// V uses value_dim instead of key_dim; beta and log_decay omit the last axis.
/// Initial state is [batch, heads, key_dim, value_dim] and is never overwritten.
pub fn gated_delta_rule<R: Runtime>(
    input: GatedDeltaInput<R>,
) -> Result<GatedDeltaOutput<R>, GatedDeltaError> {
    let [batch, heads, sequence, key_dim, value_dim] = validate(&input)?;
    recurrent_validated(input, batch, heads, sequence, key_dim, value_dim)
}

fn validate<R: Runtime>(input: &GatedDeltaInput<R>) -> Result<[usize; 5], GatedDeltaError> {
    let q = &input.query;
    if q.meta.num_dims() != 4 || !matches!(q.dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(GatedDeltaError(
            "Q must be a four-dimensional F32/F16/BF16 tensor",
        ));
    }
    let [batch, heads, sequence, key_dim]: [usize; 4] = q.meta.shape()[..].try_into().unwrap();
    if input.value.meta.num_dims() != 4 {
        return Err(GatedDeltaError("V must be four-dimensional"));
    }
    let value_dim = input.value.meta.shape()[3];
    if batch == 0 || heads == 0 || key_dim == 0 || value_dim == 0 || !input.query_scale.is_finite()
    {
        return Err(GatedDeltaError(
            "head dimensions must be positive and query_scale finite",
        ));
    }
    for (tensor, shape, dtype) in [
        (q, vec![batch, heads, sequence, key_dim], q.dtype),
        (&input.key, vec![batch, heads, sequence, key_dim], q.dtype),
        (
            &input.value,
            vec![batch, heads, sequence, value_dim],
            q.dtype,
        ),
        (&input.beta, vec![batch, heads, sequence], q.dtype),
        (&input.log_decay, vec![batch, heads, sequence], DType::F32),
        (
            &input.initial_state,
            vec![batch, heads, key_dim, value_dim],
            DType::F32,
        ),
    ] {
        if &tensor.meta.shape()[..] != shape.as_slice()
            || tensor.dtype != dtype
            || tensor.qparams.is_some()
        {
            return Err(GatedDeltaError(
                "incompatible gated delta tensor shape or dtype",
            ));
        }
        if tensor.device.to_id() != q.device.to_id() {
            return Err(GatedDeltaError("gated delta tensors must share a device"));
        }
        if shape.iter().any(|&n| n > u32::MAX as usize)
            || shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(GatedDeltaError("gated delta tensor exceeds U32 indexing"));
        }
    }
    Ok([batch, heads, sequence, key_dim, value_dim])
}

fn recurrent_validated<R: Runtime>(
    input: GatedDeltaInput<R>, batch: usize, heads: usize, sequence: usize,
    key_dim: usize, value_dim: usize,
) -> Result<GatedDeltaOutput<R>, GatedDeltaError> {
    let q = &input.query;
    let allocate = |shape: Shape, dtype| {
        empty_device_contiguous_dtype(q.client.clone(), q.device.clone(), shape, dtype)
    };
    let output = allocate([batch, heads, sequence, value_dim].into(), q.dtype);
    if sequence == 0 {
        return Ok(GatedDeltaOutput {
            output,
            final_state: input.initial_state,
        });
    }
    let final_state = allocate([batch, heads, key_dim, value_dim].into(), DType::F32);
    let columns = batch * heads * value_dim;
    let dim = RudaDim::new(q.client.properties(), columns);
    let count = calculate_ruda_count_elemwise(&q.client, columns, dim);
    let client = q.client.clone();
    let dtype = q.dtype;
    kernel::recurrent::launch::<R>(
        &client,
        count,
        dim,
        into_contiguous(input.query).into_array_arg(),
        into_contiguous(input.key).into_array_arg(),
        into_contiguous(input.value).into_array_arg(),
        into_contiguous(input.beta).into_array_arg(),
        into_contiguous(input.log_decay).into_array_arg(),
        into_contiguous(input.initial_state).into_array_arg(),
        output.clone().into_array_arg(),
        final_state.clone().into_array_arg(),
        sequence as u32,
        key_dim as u32,
        value_dim as u32,
        input.query_scale,
        dtype.into(),
    );
    Ok(GatedDeltaOutput {
        output,
        final_state,
    })
}

#[cfg(test)]
mod tests;
