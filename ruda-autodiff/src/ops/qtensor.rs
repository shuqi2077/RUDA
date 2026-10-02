//! Quantized storage remains quantized while a surrogate retains its graph.
//! The QAT input derivative is an **unclipped identity STE**, not the derivative
//! of rounding. Tracked integer scales opt into a clipped learned-step estimator.
//! Scales use FP32 and a positive floor; no calibration gradient is invented.
use ruda_tensor::{
    Backend, ExecutionError, TensorData, TensorMetadata,
    ops::{QTensorOps, FloatTensorOps},
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor,
        quantization::QuantizationParametersPrimitive},
};
use ruda_core::tensor::{FloatDType, IntDType, QuantScheme, Shape};
use crate::{
    Autodiff, checkpoint::{strategy::CheckpointStrategy, base::Checkpointer},
    grads::Gradients, ops::{Backward, Ops, unary},
    tensor::{AutodiffTensor, AutodiffQTensor},
};

#[derive(Debug)]
struct IdentitySte;
impl<B: Backend> Backward<B, 1> for IdentitySte {
    type State = ();
    fn backward(self, ops: Ops<(), 1>, grads: &mut Gradients, _: &mut Checkpointer) {
        unary::<B, _>(ops.parents, ops.node, grads, |grad| grad);
    }
}

// A different value needs a distinct node ID, including when checkpointing.
fn ste<B: Backend, C: CheckpointStrategy>(
    input: AutodiffTensor<B>, value: B::FloatTensorPrimitive,
) -> AutodiffTensor<B> {
    IdentitySte.prepare::<C>([input.node.clone()]).compute_bound().stateless(value)
}

// Differentiate the scale broadcast using ordinary autodiff operators, including
// trimming partial blocks. The stored primitive remains genuinely quantized.
fn expand_learned_scales<B: Backend, C: CheckpointStrategy>(
    scales: AutodiffTensor<B>, shape: &Shape, level: ruda_core::tensor::QuantLevel,
) -> AutodiffTensor<B> {
    use ruda_core::tensor::QuantLevel;
    type A<B, C> = Autodiff<B, C>;
    if let QuantLevel::Block(block) = level {
        assert!(!block.to_dim_vec(shape.rank()).contains(&0), "learned scale block must be nonzero");
    }
    let params = ruda_tensor::quantization::params_shape(shape, level);
    assert_eq!(scales.shape().num_elements(), params.num_elements(), "learned scale count mismatch");
    match level {
        QuantLevel::Tensor => A::<B,C>::float_reshape(scales, Shape::from(alloc::vec![1; shape.rank()])),
        QuantLevel::Block(block) => {
            let block = block.to_dim_vec(shape.rank());
            assert!(!block.contains(&0), "learned scale block must be nonzero");
            let mut source = alloc::vec::Vec::new();
            let mut expanded = alloc::vec::Vec::new();
            let mut padded = alloc::vec::Vec::new();
            for axis in 0..shape.rank() {
                source.extend([params[axis], 1]);
                expanded.extend([params[axis], block[axis] as usize]);
                padded.push(params[axis] * block[axis] as usize);
            }
            let value = A::<B,C>::float_reshape(scales, Shape::from(source));
            let value = A::<B,C>::float_expand(value, Shape::from(expanded));
            let value = A::<B,C>::float_reshape(value, Shape::from(padded));
            let slices = shape.iter().map(|&size| ruda_tensor::Slice::from(0..size)).collect::<alloc::vec::Vec<_>>();
            A::<B,C>::float_slice(value, &slices)
        }
    }
}

fn learned_quantize<B: Backend, C: CheckpointStrategy>(
    tensor: AutodiffTensor<B>, scheme: &QuantScheme, scales: AutodiffTensor<B>,
) -> AutodiffQTensor<B> {
    use ruda_core::tensor::{QuantParam, QuantValue};
    type A<B, C> = Autodiff<B, C>;
    assert!(matches!(scheme.value, QuantValue::Q8F | QuantValue::Q8S |
        QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S)
        && scheme.param == QuantParam::F32,
        "learned scales require integer Q8/Q4/Q2 and FP32 scale parameters");
    assert_eq!(B::float_device(&tensor.primitive), B::float_device(&scales.primitive),
        "learned scales must be on the input device");
    let shape = tensor.shape();
    assert!(shape.num_elements() > 0, "learned quantization requires nonempty input");
    let dtype: FloatDType = tensor.dtype().into();
    let scales = A::<B,C>::float_clamp_min(A::<B,C>::float_cast(scales, FloatDType::F32), 1e-8f32.into());
    let primitive = B::quantize(tensor.primitive.clone(), scheme,
        QuantizationParametersPrimitive { scales: scales.primitive.clone() });
    let value = B::dequantize(primitive.clone(), FloatDType::F32);
    let scales = expand_learned_scales::<B,C>(scales, &shape, scheme.level);
    let normalized = A::<B,C>::float_div(A::<B,C>::float_cast(tensor, FloatDType::F32), scales.clone());
    let (lower, upper) = scheme.value.range();
    let clipped = A::<B,C>::float_clamp(normalized, lower.into(), upper.into());
    // Detached integer codes match the BACKEND's actual rounding, not an
    // independently reimplemented rounding rule. Zero-valued correction supplies
    // d(code)/d(input/scale)=1 inside bounds, 0 outside.
    let codes = AutodiffTensor::new(B::float_div(value.clone(), scales.primitive.clone()));
    let correction = A::<B,C>::float_sub(clipped.clone(), AutodiffTensor::new(clipped.primitive));
    let surrogate = A::<B,C>::float_mul(scales, A::<B,C>::float_add(codes, correction));
    let surrogate = A::<B,C>::float_cast(surrogate, dtype);
    let exact_value = B::dequantize(primitive.clone(), dtype);
    AutodiffQTensor { primitive, surrogate: Some(ste::<B,C>(surrogate, exact_value)) }
}

impl<B: Backend, C: CheckpointStrategy> QTensorOps<Self> for Autodiff<B, C> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        AutodiffQTensor::untracked(B::q_from_data(data, device))
    }
    fn quantize(tensor: FloatTensor<Self>, scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>) -> QuantizedTensor<Self> {
        if qparams.scales.is_tracked() {
            return learned_quantize::<B, C>(tensor, scheme, qparams.scales);
        }
        let primitive = B::quantize(tensor.primitive.clone(), scheme,
            QuantizationParametersPrimitive { scales: qparams.scales.primitive });
        let surrogate = tensor.is_tracked().then(|| {
            let value = B::dequantize(primitive.clone(), tensor.dtype().into());
            ste::<B, C>(tensor, value)
        });
        AutodiffQTensor { primitive, surrogate }
    }
    fn quantize_dynamic(tensor: FloatTensor<Self>, scheme: &QuantScheme) -> QuantizedTensor<Self> {
        // Backend calibration deliberately runs outside the autodiff graph.
        let primitive = B::quantize_dynamic(tensor.primitive.clone(), scheme);
        let surrogate = tensor.is_tracked().then(|| {
            let value = B::dequantize(primitive.clone(), tensor.dtype().into());
            ste::<B, C>(tensor, value)
        });
        AutodiffQTensor { primitive, surrogate }
    }
    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let value = B::dequantize(tensor.primitive, dtype);
        match tensor.surrogate {
            // Block layout operations may requantize. Always return the actual
            // stored forward value, not an older surrogate's float value.
            Some(parent) => ste::<B, C>(Self::float_cast(parent, dtype), value),
            None => AutodiffTensor::new(value),
        }
    }
    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        B::q_device(&tensor.primitive)
    }
    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        B::q_into_data(tensor.primitive).await
    }
    fn q_detach(tensor: QuantizedTensor<Self>) -> QuantizedTensor<Self> {
        // Match float_detach: sever history but preserve the leaf-grad setting.
        let surrogate = tensor.surrogate.map(Self::float_detach);
        AutodiffQTensor { primitive: tensor.primitive, surrogate }
    }
    fn q_set_require_grad(tensor: QuantizedTensor<Self>, require_grad: bool) -> QuantizedTensor<Self> {
        if !require_grad { return AutodiffQTensor::untracked(tensor.primitive); }
        let surrogate = match tensor.surrogate {
            Some(value) => Self::float_set_require_grad(value, true),
            None => AutodiffTensor::new(B::dequantize(tensor.primitive.clone(), FloatDType::F32))
                .require_grad(),
        };
        AutodiffQTensor { primitive: tensor.primitive, surrogate: Some(surrogate) }
    }
    fn q_is_require_grad(tensor: &QuantizedTensor<Self>) -> bool {
        tensor.surrogate.as_ref().is_some_and(Self::float_is_require_grad)
    }
    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        let surrogate = tensor.surrogate.map(|t| Self::float_to_device(t, device));
        let primitive = B::q_to_device(tensor.primitive, device);
        AutodiffQTensor { primitive, surrogate }
    }
    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        let surrogate = tensor.surrogate.map(|t| Self::float_reshape(t, shape.clone()));
        let primitive = B::q_reshape(tensor.primitive, shape);
        AutodiffQTensor { primitive, surrogate }
    }
    fn q_swap_dims(tensor: QuantizedTensor<Self>, dim1: usize, dim2: usize) -> QuantizedTensor<Self> {
        let surrogate = tensor.surrogate.map(|t| Self::float_swap_dims(t, dim1, dim2));
        let primitive = B::q_swap_dims(tensor.primitive, dim1, dim2);
        AutodiffQTensor { primitive, surrogate }
    }
    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        let surrogate = tensor.surrogate.map(|t| Self::float_permute(t, axes));
        let primitive = B::q_permute(tensor.primitive, axes);
        AutodiffQTensor { primitive, surrogate }
    }
    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        let surrogate = tensor.surrogate.map(|t| Self::float_flip(t, axes));
        let primitive = B::q_flip(tensor.primitive, axes);
        AutodiffQTensor { primitive, surrogate }
    }
    fn q_select(tensor: QuantizedTensor<Self>, dim: usize, indices: IntTensor<Self>) -> QuantizedTensor<Self> {
        let surrogate = tensor.surrogate.map(|t| Self::float_select(t, dim, indices.clone()));
        let primitive = B::q_select(tensor.primitive, dim, indices);
        AutodiffQTensor { primitive, surrogate }
    }
    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[ruda_core::tensor::Slice]) -> QuantizedTensor<Self> {
        let surrogate = tensor.surrogate.map(|t| Self::float_slice(t, slices));
        let primitive = B::q_slice(tensor.primitive, slices);
        AutodiffQTensor { primitive, surrogate }
    }
    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        let surrogate = tensor.surrogate.map(|t| Self::float_expand(t, shape.clone()));
        let primitive = B::q_expand(tensor.primitive, shape);
        AutodiffQTensor { primitive, surrogate }
    }
    fn q_gather(dim: usize, tensor: QuantizedTensor<Self>, indices: IntTensor<Self>) -> QuantizedTensor<Self> {
        let surrogate = tensor.surrogate.map(|t| Self::float_gather(dim, t, indices.clone()));
        let primitive = B::q_gather(dim, tensor.primitive, indices);
        AutodiffQTensor { primitive, surrogate }
    }
    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize, dtype: IntDType) -> IntTensor<Self> {
        B::q_argmax(tensor.primitive, dim, dtype)
    }
    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize, dtype: IntDType) -> IntTensor<Self> {
        B::q_argmin(tensor.primitive, dim, dtype)
    }
}

#[cfg(all(test, not(feature = "distributed")))]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use ruda_tensor::{backend::AutodiffBackend, ops::IntTensorOps, read_sync,
        quantization::{QuantStore, QuantValue}};
    use ruda_tensor_host::Host;
    use crate::checkpoint::strategy::{NoCheckpointing, BalancedCheckpointing};
    type A = Autodiff<Host>;

    fn input<C: CheckpointStrategy>(values: Vec<f32>) -> AutodiffTensor<Host> {
        let len = values.len();
        Autodiff::<Host,C>::float_set_require_grad(
            Autodiff::<Host,C>::float_from_data(TensorData::new(values,[len]), &Default::default()), true)
    }
    fn values(tensor: <Host as ruda_tensor::BackendTypes>::FloatTensorPrimitive) -> Vec<f32> {
        read_sync(Host::float_into_data(tensor)).unwrap().to_vec::<f32>().unwrap()
    }
    fn quantize<C: CheckpointStrategy>(x: AutodiffTensor<Host>) -> AutodiffQTensor<Host> {
        let scale = Autodiff::<Host,C>::float_from_data(TensorData::new(vec![0.5f32],[1]), &Default::default());
        Autodiff::<Host,C>::quantize(x,
            &QuantScheme::default().with_value(QuantValue::Q8S).with_store(QuantStore::Native),
            QuantizationParametersPrimitive { scales: scale })
    }
    #[test]
    fn transaction_readback_preserves_quantized_storage_and_gradient_history() {
        use ruda_tensor::ops::{TransactionOps, TransactionPrimitive};
        let x = input::<NoCheckpointing>(vec![-1.25, 0.25, 2.75]);
        let q = quantize::<NoCheckpointing>(x.clone());
        let expected = read_sync(Host::q_into_data(q.primitive.clone())).unwrap();
        let untracked = A::q_from_data(expected.clone(), &Default::default());
        let data = read_sync(A::tr_execute(TransactionPrimitive::new(
            vec![x.clone()], vec![q.clone(), untracked], vec![], vec![],
        ))).unwrap();
        assert_eq!(data.read_floats.len(), 1);
        assert_eq!(data.read_floats[0].to_vec::<f32>().unwrap(), vec![-1.25, 0.25, 2.75]);
        assert_eq!(data.read_qfloats.len(), 2);
        for actual in &data.read_qfloats {
            assert_eq!(actual.dtype, expected.dtype);
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(actual.bytes, expected.bytes);
        }
        assert!(data.read_ints.is_empty() && data.read_bools.is_empty());
        let gradients = A::backward(A::float_sum(A::dequantize(q, FloatDType::F32)));
        assert_eq!(values(A::grad(&x, &gradients).unwrap()), vec![1.0; 3]);
    }

    #[test]
    fn unclipped_identity_ste_has_real_quantized_forward_and_input_gradients() {
        let x = input::<NoCheckpointing>(vec![-100.0,-0.25,0.25,100.0]);
        let y = A::dequantize(quantize::<NoCheckpointing>(x.clone()), FloatDType::F32);
        assert_eq!(values(y.primitive.clone()), vec![-63.5,-0.5,0.5,63.5]);
        let gradients = A::backward(A::float_sum(y));
        assert_eq!(values(A::grad(&x,&gradients).unwrap()), vec![1.0;4]);
    }
    fn branch_check<C: CheckpointStrategy>() {
        let x = input::<C>(vec![0.25,0.75,-0.25]);
        let q = Autodiff::<Host,C>::dequantize(quantize::<C>(x.clone()), FloatDType::F32);
        let y = Autodiff::<Host,C>::float_sum(Autodiff::<Host,C>::float_mul(q,x.clone()));
        let gradients = Autodiff::<Host,C>::backward(y);
        assert_eq!(values(Autodiff::<Host,C>::grad(&x,&gradients).unwrap()), vec![0.75,1.75,-0.75]);
    }
    #[test] fn original_input_branch_without_checkpointing() { branch_check::<NoCheckpointing>(); }
    #[test] fn original_input_branch_with_checkpointing() { branch_check::<BalancedCheckpointing>(); }
    #[test]
    fn duplicate_indices_accumulate_and_detach_severs_original_history() {
        let x = input::<NoCheckpointing>(vec![1.0,2.0,3.0]);
        let q = quantize::<NoCheckpointing>(x.clone());
        let indices = Host::int_from_data(TensorData::new(vec![2i64,0,2],[3]), &Default::default());
        let selected = A::q_select(q.clone(),0,indices);
        let gradients = A::backward(A::float_sum(A::dequantize(selected,FloatDType::F32)));
        assert_eq!(values(A::grad(&x,&gradients).unwrap()), vec![1.0,0.0,2.0]);
        let detached = A::q_set_require_grad(A::q_detach(q),true);
        let gradients = A::backward(A::float_sum(A::dequantize(detached,FloatDType::F32)));
        assert!(A::grad(&x,&gradients).is_none());
    }
    #[test]
    fn learned_step_has_clipped_input_gradient_and_nonzero_scale_gradient() {
        let x = input::<NoCheckpointing>(vec![-10.0, -0.25, 0.25, 10.0]);
        let scales = input::<NoCheckpointing>(vec![0.5]);
        let scheme = QuantScheme::default().with_value(QuantValue::Q4S).with_store(QuantStore::Native);
        let q = A::quantize(x.clone(), &scheme, QuantizationParametersPrimitive { scales: scales.clone() });
        let y = A::dequantize(q, FloatDType::F32);
        assert_eq!(values(y.primitive.clone()), vec![-3.5, -0.5, 0.5, 3.5]);
        let weights = A::float_from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]), &Default::default());
        let gradients = A::backward(A::float_sum(A::float_mul(y, weights)));
        assert_eq!(values(A::grad(&x, &gradients).unwrap()), vec![0.0, 2.0, 3.0, 0.0]);
        assert_eq!(values(A::grad(&scales, &gradients).unwrap()), vec![21.5]);
    }

    #[test]
    fn learned_scale_only_parent_is_retained() {
        let x = A::float_from_data(TensorData::new(vec![0.25f32], [1]), &Default::default());
        let scale = input::<NoCheckpointing>(vec![0.5]);
        let q = A::quantize(x, &QuantScheme::default().with_store(QuantStore::Native),
            QuantizationParametersPrimitive { scales: scale.clone() });
        let gradients = A::backward(A::float_sum(A::dequantize(q, FloatDType::F32)));
        assert_eq!(values(A::grad(&scale, &gradients).unwrap()), vec![0.5]);
    }
}
