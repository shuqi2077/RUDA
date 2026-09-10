use crate::{
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    graph::NodeId,
    ops::{Backward, Ops, OpsKind, broadcast_shape, unary},
    tensor::AutodiffTensor,
};
use ruda_tensor::{Backend, DType, FloatDType, Scalar, TensorMetadata, get_device_settings, tensor::FloatTensor};

fn gradient_operands<B: Backend>(
    input: FloatTensor<B>,
    grad: FloatTensor<B>,
) -> (FloatTensor<B>, FloatTensor<B>, FloatDType) {
    let dtype = input.dtype().into();
    if matches!(input.dtype(), DType::F16 | DType::BF16) {
        (B::float_cast(input, FloatDType::F32), B::float_cast(grad, FloatDType::F32), dtype)
    } else {
        (input, grad, dtype)
    }
}

#[derive(Debug)]
struct IntegerPowerScalar;

impl<B: Backend> Backward<B, 1> for IntegerPowerScalar {
    type State = (Option<NodeId>, Scalar);

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let (input_id, exponent) = ops.state;
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            if matches!(exponent, Scalar::Int(0) | Scalar::UInt(0)) {
                return B::float_zeros(grad.shape(), &B::float_device(&grad), grad.dtype().into());
            }
            if matches!(exponent, Scalar::Int(1) | Scalar::UInt(1)) {
                return grad;
            }
            let input = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(
                input_id.expect("nonconstant integer power requires its input"),
            );
            let (input, grad, dtype) = gradient_operands::<B>(input, grad);
            let previous = match exponent {
                Scalar::Int(value) => value.checked_sub(1).map(Scalar::Int),
                Scalar::UInt(value) => Some(Scalar::UInt(value - 1)),
                _ => unreachable!("integer power state must hold an integer"),
            };
            let power = match previous {
                Some(previous) => B::float_powi_scalar(input, previous),
                None => {
                    let power = B::float_powi_scalar(input.clone(), exponent);
                    B::float_div(power, input)
                }
            };
            let derivative = B::float_mul_scalar(power, exponent);
            let result = B::float_mul(grad, derivative);
            if result.dtype() == DType::from(dtype) { result } else { B::float_cast(result, dtype) }
        });
    }
}

#[derive(Debug)]
struct IntegerPowerTensor;

impl<B: Backend> Backward<B, 1> for IntegerPowerTensor {
    type State = (NodeId, B::IntTensorPrimitive);

    fn backward(
        self,
        ops: Ops<Self::State, 1>,
        grads: &mut Gradients,
        checkpointer: &mut Checkpointer,
    ) {
        let (input_id, exponent) = ops.state;
        let input = checkpointer.retrieve_node_output::<B::FloatTensorPrimitive>(input_id);
        let shape = input.shape();
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            let (input, grad, dtype) = gradient_operands::<B>(input, grad);
            let settings = get_device_settings::<B>(&B::float_device(&input));
            let zero = B::int_equal_elem(exponent.clone(), 0.into(), settings.bool_dtype);
            let mut adjusted = B::int_mask_fill(exponent.clone(), zero.clone(), 1.into());
            let minimum = match exponent.dtype() {
                DType::I8 => Some(i8::MIN as i64),
                DType::I16 => Some(i16::MIN as i64),
                DType::I32 => Some(i32::MIN as i64),
                DType::I64 => Some(i64::MIN),
                _ => None,
            };
            let minimum_mask = minimum.map(|minimum| {
                let mask = B::int_equal_elem(exponent.clone(), minimum.into(), settings.bool_dtype);
                adjusted = B::int_mask_fill(adjusted.clone(), mask.clone(), (minimum + 1).into());
                mask
            });
            let previous = B::int_sub_scalar(adjusted, 1.into());
            let mut power = B::float_powi(input.clone(), previous);
            if let Some(mask) = minimum_mask {
                let corrected = B::float_div(power.clone(), input.clone());
                power = B::float_mask_where(power, mask, corrected);
            }
            let coefficient = B::int_into_float(exponent, input.dtype().into());
            let derivative = B::float_mul(power, coefficient);
            let result = B::float_mul(grad, derivative);
            let result = B::float_mask_fill(result, zero, 0.into());
            let result = broadcast_shape::<B>(result, &shape);
            if result.dtype() == DType::from(dtype) { result } else { B::float_cast(result, dtype) }
        });
    }
}

pub(super) fn tensor<B: Backend, C: CheckpointStrategy>(
    input: AutodiffTensor<B>,
    exponent: B::IntTensorPrimitive,
) -> AutodiffTensor<B> {
    match IntegerPowerTensor.prepare::<C>([input.node.clone()]).compute_bound().stateful() {
        OpsKind::Tracked(mut prep) => {
            let input_id = prep.checkpoint(&input);
            let output = B::float_powi(input.primitive, exponent.clone());
            prep.finish((input_id, exponent), output)
        }
        OpsKind::UnTracked(prep) => prep.finish(B::float_powi(input.primitive, exponent)),
    }
}

pub(super) fn scalar<B: Backend, C: CheckpointStrategy>(
    tensor: AutodiffTensor<B>,
    value: Scalar,
    implementation: bool,
) -> AutodiffTensor<B> {
    let forward = |input| {
        if implementation {
            B::float_powi_scalar_impl(input, value)
        } else {
            B::float_powi_scalar(input, value)
        }
    };
    match IntegerPowerScalar.prepare::<C>([tensor.node.clone()]).compute_bound().stateful() {
        OpsKind::Tracked(mut prep) => {
            let exponent = match value {
                Scalar::UInt(_) => value,
                _ => Scalar::Int(value.elem::<i64>()),
            };
            let input_id = if matches!(exponent, Scalar::Int(0 | 1) | Scalar::UInt(0 | 1)) {
                None
            } else {
                Some(prep.checkpoint(&tensor))
            };
            prep.finish((input_id, exponent), forward(tensor.primitive))
        }
        OpsKind::UnTracked(prep) => prep.finish(forward(tensor.primitive)),
    }
}
