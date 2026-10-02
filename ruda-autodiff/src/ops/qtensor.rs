//! Quantized storage remains quantized while a surrogate retains its graph.
//! The QAT input derivative is an **unclipped identity STE**, not the derivative
//! of rounding. Calibration/scales are constants; learned-scale QAT is rejected.
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

impl<B: Backend, C: CheckpointStrategy> QTensorOps<Self> for Autodiff<B, C> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        AutodiffQTensor::untracked(B::q_from_data(data, device))
    }
    fn quantize(tensor: FloatTensor<Self>, scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>) -> QuantizedTensor<Self> {
        assert!(!qparams.scales.is_tracked(),
            "identity-STE quantization differentiates inputs only; use untracked scales");
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
    #[should_panic(expected="scales")]
    fn learning_scales_is_rejected_instead_of_silently_losing_gradient() {
        A::quantize(input::<NoCheckpointing>(vec![1.0]), &QuantScheme::default(),
            QuantizationParametersPrimitive { scales: input::<NoCheckpointing>(vec![0.5]) });
    }
}
