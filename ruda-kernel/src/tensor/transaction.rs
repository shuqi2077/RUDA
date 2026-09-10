use super::RudaTensor;
use crate::dsl::{Runtime, client::ComputeClient, server::{CopyDescriptor, Handle}};
use ruda_core::tensor::{DType, Shape, Strides, data::TensorData, execution::ExecutionError};
use ruda_core::tensor::transaction::TransactionData;

pub struct ReadbackBatch<R: Runtime> {
    pub read_floats: Vec<RudaTensor<R>>,
    pub read_qfloats: Vec<RudaTensor<R>>,
    pub read_ints: Vec<RudaTensor<R>>,
    pub read_bools: Vec<RudaTensor<R>>,
}

pub async fn execute<R: Runtime>(transaction: ReadbackBatch<R>) -> Result<TransactionData, ExecutionError> {
    enum Kind {
        Float,
        QFloat { shape: Shape, dtype: DType },
        QParams,
        Int,
        Bool,
    }

    #[derive(derive_new::new)]
    struct BindingData<R: Runtime> {
        index: usize,
        client: ComputeClient<R>,
        kind: Kind,
        handle: Option<Handle>,
        shape: Shape,
        strides: Strides,
        dtype: DType,
    }

    let mut num_bindings = 0;

    let mut kinds = Vec::new();

    for t in transaction.read_floats.into_iter() {
        let t = super::contiguous::into_contiguous_aligned(t);
        let binding = BindingData::new(
            num_bindings,
            t.client.clone(),
            Kind::Float,
            Some(t.handle.clone()),
            t.meta.shape.clone(),
            t.meta.strides.clone(),
            t.dtype,
        );

        kinds.push(binding);
        num_bindings += 1;
    }
    for t in transaction.read_qfloats {
        let shape = t.meta.shape.clone();
        let dtype = t.dtype;
        let (values, params) = if t.qparams.is_some() {
            let (values, params) = t.quantized_handles().ok_or_else(|| ExecutionError::WithContext {
                reason: "Missing quantized tensor handles during transaction readback".into(),
            })?;
            (values, Some(params))
        } else {
            (t, None)
        };
        let values = super::contiguous::into_contiguous_aligned(values);
        kinds.push(BindingData::new(
            num_bindings,
            values.client.clone(),
            Kind::QFloat { shape, dtype },
            Some(values.handle.clone()),
            values.meta.shape.clone(),
            values.meta.strides.clone(),
            values.dtype,
        ));
        num_bindings += 1;
        if let Some(params) = params {
            let params = super::contiguous::into_contiguous_aligned(params);
            kinds.push(BindingData::new(
                num_bindings,
                params.client.clone(),
                Kind::QParams,
                Some(params.handle.clone()),
                params.meta.shape.clone(),
                params.meta.strides.clone(),
                params.dtype,
            ));
            num_bindings += 1;
        }
    }
    for t in transaction.read_ints.into_iter() {
        let t = super::contiguous::into_contiguous_aligned(t);
        let binding = BindingData::new(
            num_bindings,
            t.client.clone(),
            Kind::Int,
            Some(t.handle.clone()),
            t.meta.shape.clone(),
            t.meta.strides.clone(),
            t.dtype,
        );

        kinds.push(binding);
        num_bindings += 1;
    }
    for t in transaction.read_bools.into_iter() {
        let t = super::contiguous::into_contiguous_aligned(t);
        let binding = BindingData::new(
            num_bindings,
            t.client.clone(),
            Kind::Bool,
            Some(t.handle.clone()),
            t.meta.shape.clone(),
            t.meta.strides.clone(),
            t.dtype,
        );

        kinds.push(binding);
        num_bindings += 1;
    }

    if kinds.is_empty() {
        return Ok(TransactionData::default());
    }

    struct ReadGroup<R: Runtime> {
        client: ComputeClient<R>,
        indices: Vec<usize>,
        bindings: Vec<CopyDescriptor>,
    }
    let mut groups = Vec::<ReadGroup<R>>::new();
    for binding in &mut kinds {
        let group = groups.iter().position(|group| group.client.same_execution_queue(&binding.client));
        let group = match group {
            Some(index) => index,
            None => {
                groups.push(ReadGroup {
                    client: binding.client.clone(),
                    indices: Vec::new(),
                    bindings: Vec::new(),
                });
                groups.len() - 1
            }
        };
        groups[group].indices.push(binding.index);
        groups[group].bindings.push(CopyDescriptor::new(
            binding.handle.take().unwrap().binding(),
            binding.shape.clone(),
            binding.strides.clone(),
            binding.dtype.size(),
        ));
    }

    let requests = groups.iter_mut().map(|group| {
        let indices = core::mem::take(&mut group.indices);
        let bindings = core::mem::take(&mut group.bindings);
        (indices, group.client.read_tensor_async(bindings))
    }).collect::<Vec<_>>();
    let mut data = (0..num_bindings).map(|_| None).collect::<Vec<Option<_>>>();
    for (indices, request) in requests {
        let buffers = request.await.map_err(|err| ExecutionError::WithContext {
            reason: format!("{err:?}"),
        })?;
        if buffers.len() != indices.len() {
            return Err(ExecutionError::WithContext {
                reason: format!("Transaction readback expected {} buffers, got {}", indices.len(), buffers.len()),
            });
        }
        for (index, bytes) in indices.into_iter().zip(buffers) {
            data[index] = Some(bytes);
        }
    }

    let mut result = TransactionData::default();

    for binding in kinds {
        let bytes = data.get_mut(binding.index).unwrap().take().unwrap();
        let t_data = TensorData::from_bytes(bytes, binding.shape, binding.dtype);

        match binding.kind {
            Kind::Float => {
                result.read_floats.push(t_data);
            }
            Kind::QFloat { shape, dtype } => {
                result.read_qfloats.push(TensorData::from_bytes(t_data.bytes, shape, dtype));
            }
            Kind::QParams => {
                result.read_qfloats.last_mut().unwrap().bytes.extend_from_byte_slice(&t_data.bytes);
            }
            Kind::Int => {
                result.read_ints.push(t_data);
            }
            Kind::Bool => {
                result.read_bools.push(t_data);
            }
        }
    }

    Ok(result)
}
