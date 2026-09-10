use super::GradientsParams;
use alloc::{format, vec::Vec};
use hashbrown::HashSet;
use ruda_model::{
    module::ParamId,
    record::{PrecisionSettings, Record, RecorderError},
    tensor::{DType, TensorData, TensorMetadata, TensorPrimitive, backend::Backend, try_read_sync},
};
use serde::{Deserialize, Serialize};
use ruda_model::tensor::quantization::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue};

/// Host-owned gradient state, keyed by the original model parameter IDs.
///
/// This record can be stored alongside model, optimizer and scheduler records.
/// Recorder precision settings apply to floating-point values; the original
/// gradient dtype and shape are restored when loading. Use settings that do not
/// narrow the gradient values when exact continuation is required.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradientsParamsRecord {
    gradients: Vec<(u64, DType, TensorData)>,
}

impl<B: Backend> Record<B> for GradientsParamsRecord {
    type Item<S: PrecisionSettings> = Self;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        Self {
            gradients: self
                .gradients
                .into_iter()
                .map(|(id, dtype, data)| {
                    let data = if matches!(data.dtype, DType::QFloat(_)) {
                        data
                    } else {
                        data.convert::<S::FloatElem>()
                    };
                    (id, dtype, data)
                })
                .collect(),
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item
    }
}

impl GradientsParams {
    /// Snapshot all gradients without clearing them, including accumulated gradients.
    ///
    /// `B` must be the backend used to register the gradients (normally the
    /// autodiff backend's inner backend). Device reads complete before returning.
    pub fn try_to_record<B: Backend>(&self) -> Result<GradientsParamsRecord, RecorderError> {
        try_read_sync(self.to_record_async::<B>()).ok_or_else(|| {
            RecorderError::Unknown(
                "Synchronous gradient read is unavailable; use to_record_async".into(),
            )
        })?
    }

    /// Asynchronously snapshot all gradients without clearing the container.
    ///
    /// `B` must match the backend used to register the gradients.
    pub async fn to_record_async<B: Backend>(&self) -> Result<GradientsParamsRecord, RecorderError> {
        let mut ids = self.container.ids().into_iter().copied().collect::<Vec<_>>();
        ids.sort();
        let mut gradients = Vec::with_capacity(ids.len());
        for id in ids {
            let primitive = self.container.get::<B>(&id).ok_or_else(|| {
                RecorderError::Unknown(format!("Missing gradient for parameter {id}"))
            })?;
            let dtype = primitive.dtype();
            let data = match primitive {
                TensorPrimitive::Float(tensor) => B::float_into_data(tensor).await,
                TensorPrimitive::QFloat(tensor) => B::q_into_data(tensor).await,
            }
            .map_err(|error| {
                RecorderError::Unknown(format!("Reading gradient for parameter {id}: {error}"))
            })?;
            gradients.push((id.val(), dtype, data));
        }
        Ok(GradientsParamsRecord { gradients })
    }

    /// Restore recorded gradients to a device without summing or clearing entries.
    ///
    /// Restore the model's parameter IDs from the same checkpoint before using
    /// these gradients. `B` is the gradient backend, normally the inner backend.
    pub fn from_record<B: Backend>(
        record: GradientsParamsRecord,
        device: &B::Device,
    ) -> Result<Self, RecorderError> {
        let mut ids = HashSet::with_capacity(record.gradients.len());
        for (id, dtype, data) in &record.gradients {
            if !ids.insert(*id) {
                return Err(RecorderError::Unknown(format!(
                    "Duplicate gradient parameter ID {id}"
                )));
            }
            let compatible = match (dtype, data.dtype) {
                (DType::QFloat(expected), DType::QFloat(actual)) => *expected == actual,
                (DType::QFloat(_), _) | (_, DType::QFloat(_)) => false,
                _ => dtype.is_float() && data.dtype.is_float(),
            };
            if !compatible {
                return Err(RecorderError::Unknown(format!(
                    "Invalid gradient dtype for parameter {id}: {dtype:?}, stored {:?}",
                    data.dtype
                )));
            }
            let num_elements = data.shape.iter().try_fold(1usize, |count, dim| {
                count.checked_mul(*dim)
            }).ok_or_else(|| RecorderError::Unknown(format!(
                "Gradient shape overflows for parameter {id}: {:?}", data.shape
            )))?;
            if let DType::QFloat(scheme) = data.dtype {
                validate_quantized_gradient(*id, data, scheme, num_elements)?;
            }
            if data.dtype.is_float() {
                let stored_bytes = num_elements.checked_mul(data.dtype.size()).ok_or_else(|| {
                    RecorderError::Unknown(format!(
                        "Stored gradient byte count overflows for parameter {id}"
                    ))
                })?;
                if data.bytes.len() != stored_bytes {
                    return Err(RecorderError::Unknown(format!(
                        "Invalid gradient byte count for parameter {id}: expected {stored_bytes}, got {}",
                        data.bytes.len()
                    )));
                }
                let restored_bytes = num_elements.checked_mul(dtype.size()).ok_or_else(|| {
                    RecorderError::Unknown(format!(
                        "Restored gradient byte count overflows for parameter {id}"
                    ))
                })?;
                if restored_bytes > isize::MAX as usize {
                    return Err(RecorderError::Unknown(format!(
                        "Restored gradient allocation exceeds addressable size for parameter {id}"
                    )));
                }
            }
        }

        let mut gradients = Self::new();
        for (id, dtype, data) in record.gradients {
            let primitive = match dtype {
                DType::QFloat(_) => TensorPrimitive::QFloat(B::q_from_data(data, device)),
                _ => {
                    let tensor = B::float_from_data(data.convert_dtype(dtype), device);
                    let tensor = if tensor.dtype() == dtype {
                        tensor
                    } else {
                        B::float_cast(tensor, dtype.into())
                    };
                    TensorPrimitive::Float(tensor)
                }
            };
            gradients
                .container
                .register::<B>(ParamId::from(id), primitive);
        }
        Ok(gradients)
    }
}

fn validate_quantized_gradient(
    id: u64,
    data: &TensorData,
    scheme: QuantScheme,
    num_elements: usize,
) -> Result<(), RecorderError> {
    let invalid = |reason: &str| RecorderError::Unknown(format!(
        "Invalid quantized gradient for parameter {id}: {reason}"
    ));
    let (values_count, value_size) = match scheme.store {
        QuantStore::Native => (num_elements, 1usize),
        QuantStore::PackedU32(dim) | QuantStore::PackedNative(dim) => {
            let axis = data.shape.rank().checked_sub(dim).and_then(|rank| rank.checked_sub(1))
                .ok_or_else(|| invalid("packing dimension exceeds tensor rank"))?;
            let value_size = match scheme.store {
                QuantStore::PackedU32(_) => 4usize,
                QuantStore::PackedNative(_) if scheme.value == QuantValue::E2M1 => 1,
                _ => return Err(invalid("value type does not support native packing")),
            };
            let num_quants = scheme.num_quants();
            let count = data.shape.iter().enumerate().try_fold(1usize, |count, (dim, size)| {
                let size = if dim == axis { size.div_ceil(num_quants) } else { *size };
                count.checked_mul(size)
            }).ok_or_else(|| invalid("packed value count overflows"))?;
            (count, value_size)
        }
    };
    let values_bytes = values_count.checked_mul(value_size)
        .ok_or_else(|| invalid("packed value byte count overflows"))?;
    let params_count = match scheme.level {
        QuantLevel::Tensor => 1usize,
        QuantLevel::Block(blocks) => {
            let block_dims = blocks.to_dim_vec(data.shape.rank());
            if block_dims.contains(&0) {
                return Err(invalid("quantization block dimension is zero"));
            }
            data.shape.iter().zip(block_dims).try_fold(1usize, |count, (size, block)| {
                count.checked_mul(size.div_ceil(block as usize))
            }).ok_or_else(|| invalid("scale count overflows"))?
        }
    };
    let param_size = match scheme.param {
        QuantParam::F32 => 4usize,
        QuantParam::F16 | QuantParam::BF16 => 2,
        QuantParam::UE8M0 | QuantParam::UE4M3 => 1,
    };
    let required_bytes = params_count.checked_mul(param_size)
        .and_then(|params_bytes| values_bytes.checked_add(params_bytes))
        .ok_or_else(|| invalid("value and scale byte count overflows"))?;
    if required_bytes > isize::MAX as usize {
        return Err(invalid("value and scale allocation exceeds addressable size"));
    }
    let exact = scheme.param == QuantParam::F32;
    if data.bytes.len() < required_bytes || (exact && data.bytes.len() != required_bytes) {
        let qualifier = if exact { "" } else { "at least " };
        return Err(RecorderError::Unknown(format!(
            "Invalid quantized gradient byte count for parameter {id}: expected {qualifier}{required_bytes}, got {}",
            data.bytes.len()
        )));
    }
    Ok(())
}
