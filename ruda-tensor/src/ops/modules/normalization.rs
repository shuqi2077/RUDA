use crate::{Backend, DType, FloatDType, TensorMetadata, tensor::FloatTensor};
use ruda_core::tensor::Shape;
use super::{LayerNormBackward, LayerNormOutput};

/// Last-axis normalization on the current backend, retaining FP32/FP64 row statistics.
pub fn layer_norm_with_stats<B: Backend>(
    input: FloatTensor<B>, gamma: FloatTensor<B>, beta: Option<FloatTensor<B>>, epsilon: f64,
) -> LayerNormOutput<B> {
    let shape = input.shape();
    let width = *shape.last().expect("LayerNorm input must have an axis");
    assert!(width > 0, "LayerNorm final axis must be nonempty");
    assert_eq!(gamma.shape(), Shape::new([width]), "LayerNorm weight shape differs");
    if let Some(beta) = &beta { assert_eq!(beta.shape(), gamma.shape(), "LayerNorm bias shape differs"); }
    assert!(epsilon.is_finite() && epsilon > 0.0, "LayerNorm epsilon must be finite and positive");
    let rows = shape.num_elements() / width;
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    if rows == 0 {
        let device = B::float_device(&input);
        return LayerNormOutput { output: input,
            mean: B::float_zeros(Shape::new([0]), &device, compute),
            rstd: B::float_zeros(Shape::new([0]), &device, compute) };
    }
    let input = B::float_reshape(B::float_cast(input, compute), Shape::new([rows, width]));
    let mean = B::float_mean_dim(input.clone(), 1);
    let centered = B::float_sub(input, mean.clone());
    let variance = B::float_mean_dim(B::float_mul(centered.clone(), centered.clone()), 1);
    let rstd = B::float_recip(B::float_sqrt(B::float_add_scalar(variance, epsilon.into())));
    let gamma = B::float_reshape(B::float_cast(gamma, compute), Shape::new([1, width]));
    let mut output = B::float_mul(B::float_mul(centered, rstd.clone()), gamma);
    if let Some(beta) = beta {
        output = B::float_add(output, B::float_reshape(B::float_cast(beta, compute), Shape::new([1, width])));
    }
    LayerNormOutput {
        output: B::float_reshape(B::float_cast(output, storage), shape),
        mean: B::float_reshape(mean, Shape::new([rows])),
        rstd: B::float_reshape(rstd, Shape::new([rows])),
    }
}

/// Complete first-order derivatives from saved statistics, without repeating forward.
pub fn layer_norm_backward<B: Backend>(
    input: FloatTensor<B>, gamma: FloatTensor<B>, grad: FloatTensor<B>,
    mean: FloatTensor<B>, rstd: FloatTensor<B>,
) -> LayerNormBackward<B> {
    let shape = input.shape();
    let width = *shape.last().expect("LayerNorm input must have an axis");
    assert!(width > 0, "LayerNorm final axis must be nonempty");
    let rows = shape.num_elements() / width;
    assert_eq!(gamma.shape(), Shape::new([width]), "LayerNorm weight shape differs");
    assert_eq!(grad.shape(), shape, "LayerNorm gradient shape differs");
    assert_eq!(mean.shape(), Shape::new([rows]), "LayerNorm mean shape differs");
    assert_eq!(rstd.shape(), mean.shape(), "LayerNorm reciprocal deviation shape differs");
    let input_storage: FloatDType = input.dtype().into();
    let weight_storage: FloatDType = gamma.dtype().into();
    let compute = if [&input, &gamma, &grad, &mean, &rstd].iter().any(|value| value.dtype() == DType::F64) {
        FloatDType::F64
    } else { FloatDType::F32 };
    if rows == 0 {
        let device = B::float_device(&input);
        return LayerNormBackward { input,
            weight: B::float_zeros(Shape::new([width]), &device, weight_storage),
            bias: B::float_zeros(Shape::new([width]), &device, compute) };
    }
    let input = B::float_reshape(B::float_cast(input, compute), Shape::new([rows, width]));
    let grad = B::float_reshape(B::float_cast(grad, compute), Shape::new([rows, width]));
    let mean = B::float_reshape(B::float_cast(mean, compute), Shape::new([rows, 1]));
    let rstd = B::float_reshape(B::float_cast(rstd, compute), Shape::new([rows, 1]));
    let normalized = B::float_mul(B::float_sub(input, mean), rstd.clone());
    let gamma = B::float_reshape(B::float_cast(gamma, compute), Shape::new([1, width]));
    let scaled_grad = B::float_mul(grad.clone(), gamma);
    let average_grad = B::float_mean_dim(scaled_grad.clone(), 1);
    let average_product = B::float_mean_dim(B::float_mul(scaled_grad.clone(), normalized.clone()), 1);
    let input_grad = B::float_mul(rstd, B::float_sub(B::float_sub(scaled_grad, average_grad),
        B::float_mul(normalized.clone(), average_product)));
    let weight_grad = B::float_sum_dim(B::float_mul(grad.clone(), normalized), 0);
    let bias_grad = B::float_sum_dim(grad, 0);
    LayerNormBackward {
        input: B::float_reshape(B::float_cast(input_grad, input_storage), shape),
        weight: B::float_reshape(B::float_cast(weight_grad, weight_storage), Shape::new([width])),
        bias: B::float_reshape(bias_grad, Shape::new([width])),
    }
}
