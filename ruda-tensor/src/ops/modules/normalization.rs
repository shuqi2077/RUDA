use crate::{Backend, DType, FloatDType, TensorMetadata, tensor::FloatTensor};
use ruda_core::tensor::Shape;
use super::{LayerNormBackward, LayerNormOutput, RmsNormBackward, RmsNormOutput};

/// Last-axis RMSNorm retaining FP32/FP64 row statistics on the current backend.
pub fn rms_norm_with_stats<B: Backend>(input: FloatTensor<B>, gamma: FloatTensor<B>, epsilon: f64) -> RmsNormOutput<B> {
    let shape = input.shape();
    let width = *shape.last().expect("RMSNorm requires an axis");
    assert!(width > 0, "RMSNorm final axis must be nonempty");
    assert_eq!(gamma.shape(), Shape::new([width]), "RMSNorm weight shape differs");
    assert!(epsilon.is_finite() && epsilon > 0.0, "RMSNorm epsilon must be finite and positive");
    let rows = shape.num_elements() / width;
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    if rows == 0 {
        let device = B::float_device(&input);
        return RmsNormOutput { output: input, rstd: B::float_zeros(Shape::new([0]), &device, compute) };
    }
    let input = B::float_reshape(B::float_cast(input, compute), Shape::new([rows, width]));
    let square_mean = B::float_mean_dim(B::float_mul(input.clone(), input.clone()), 1);
    let rstd = B::float_recip(B::float_sqrt(B::float_add_scalar(square_mean, epsilon.into())));
    let gamma = B::float_reshape(B::float_cast(gamma, compute), Shape::new([1, width]));
    let output = B::float_mul(B::float_mul(input, rstd.clone()), gamma);
    RmsNormOutput { output: B::float_reshape(B::float_cast(output, storage), shape),
        rstd: B::float_reshape(rstd, Shape::new([rows])) }
}

/// RMSNorm derivatives from saved reciprocal norms and the original working values.
pub fn rms_norm_backward<B: Backend>(input: FloatTensor<B>, gamma: FloatTensor<B>, grad: FloatTensor<B>,
    rstd: FloatTensor<B>) -> RmsNormBackward<B> {
    let [input, weight] = rms_norm_backward_select::<B>(input, gamma, grad, rstd, [true; 2]);
    RmsNormBackward { input: input.expect("requested RMSNorm input gradient"),
        weight: weight.expect("requested RMSNorm weight gradient") }
}

/// Compute only requested input and weight derivatives on the current backend.
pub fn rms_norm_backward_select<B: Backend>(input: FloatTensor<B>, gamma: FloatTensor<B>, grad: FloatTensor<B>,
    rstd: FloatTensor<B>, mask: [bool; 2]) -> [Option<FloatTensor<B>>; 2] {
    if mask == [false; 2] { return [None, None]; }
    let shape = input.shape();
    let width = *shape.last().expect("RMSNorm requires an axis");
    assert!(width > 0, "RMSNorm final axis must be nonempty");
    let rows = shape.num_elements() / width;
    assert_eq!(gamma.shape(), Shape::new([width]), "RMSNorm weight shape differs");
    assert_eq!(grad.shape(), shape, "RMSNorm gradient shape differs");
    assert_eq!(rstd.shape(), Shape::new([rows]), "RMSNorm reciprocal norm shape differs");
    let storage: FloatDType = input.dtype().into();
    let weight_storage: FloatDType = gamma.dtype().into();
    let forward_compute = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    let compute = if [&input, &gamma, &grad, &rstd].iter().any(|value| value.dtype() == DType::F64) {
        FloatDType::F64
    } else { FloatDType::F32 };
    if rows == 0 {
        let device = B::float_device(&input);
        return [mask[0].then_some(input), mask[1].then(|| B::float_zeros(Shape::new([width]), &device, weight_storage))];
    }
    let input = B::float_reshape(B::float_cast(input, forward_compute), Shape::new([rows, width]));
    let rstd = B::float_reshape(B::float_cast(rstd, forward_compute), Shape::new([rows, 1]));
    let normalized = B::float_cast(B::float_mul(input, rstd.clone()), compute);
    let grad = B::float_reshape(B::float_cast(grad, compute), Shape::new([rows, width]));
    let input_grad = mask[0].then(|| {
        let rstd = B::float_cast(rstd, compute);
        let gamma = B::float_reshape(B::float_cast(B::float_cast(gamma, forward_compute), compute), Shape::new([1, width]));
        let scaled_grad = B::float_mul(grad.clone(), gamma);
        let correction = B::float_mean_dim(B::float_mul(scaled_grad.clone(), normalized.clone()), 1);
        let value = B::float_mul(rstd, B::float_sub(scaled_grad, B::float_mul(normalized.clone(), correction)));
        B::float_reshape(B::float_cast(value, storage), shape)
    });
    let weight_grad = mask[1].then(|| {
        let value = B::float_sum_dim(B::float_mul(grad, normalized), 0);
        B::float_reshape(B::float_cast(value, weight_storage), Shape::new([width]))
    });
    [input_grad, weight_grad]
}

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
    let [input, weight, bias] = layer_norm_backward_select::<B>(input, gamma, grad, mean, rstd, [true; 3]);
    LayerNormBackward { input: input.expect("requested LayerNorm input gradient"),
        weight: weight.expect("requested LayerNorm weight gradient"), bias: bias.expect("requested LayerNorm bias gradient") }
}

/// Compute requested input, weight and bias derivatives without building unrequested graphs.
pub fn layer_norm_backward_select<B: Backend>(input: FloatTensor<B>, gamma: FloatTensor<B>, grad: FloatTensor<B>,
    mean: FloatTensor<B>, rstd: FloatTensor<B>, mask: [bool; 3]) -> [Option<FloatTensor<B>>; 3] {
    if mask == [false; 3] { return [None, None, None]; }
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
    let forward_compute = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    let compute = if [&input, &gamma, &grad, &mean, &rstd].iter().any(|value| value.dtype() == DType::F64) {
        FloatDType::F64
    } else { FloatDType::F32 };
    if rows == 0 {
        let device = B::float_device(&input);
        return [mask[0].then_some(input),
            mask[1].then(|| B::float_zeros(Shape::new([width]), &device, weight_storage)),
            mask[2].then(|| B::float_zeros(Shape::new([width]), &device, compute))];
    }
    let grad = B::float_reshape(B::float_cast(grad, compute), Shape::new([rows, width]));
    if !mask[0] && !mask[1] {
        let bias = B::float_reshape(B::float_sum_dim(grad, 0), Shape::new([width]));
        return [None, None, Some(bias)];
    }
    let input = B::float_reshape(B::float_cast(input, forward_compute), Shape::new([rows, width]));
    let mean = B::float_reshape(B::float_cast(mean, forward_compute), Shape::new([rows, 1]));
    let rstd = B::float_reshape(B::float_cast(rstd, forward_compute), Shape::new([rows, 1]));
    let normalized = B::float_cast(B::float_mul(B::float_sub(input, mean), rstd.clone()), compute);
    let input_grad = mask[0].then(|| {
        let rstd = B::float_cast(rstd, compute);
        let gamma = B::float_reshape(B::float_cast(B::float_cast(gamma, forward_compute), compute), Shape::new([1, width]));
        let scaled_grad = B::float_mul(grad.clone(), gamma);
        let average_grad = B::float_mean_dim(scaled_grad.clone(), 1);
        let average_product = B::float_mean_dim(B::float_mul(scaled_grad.clone(), normalized.clone()), 1);
        let value = B::float_mul(rstd, B::float_sub(B::float_sub(scaled_grad, average_grad),
            B::float_mul(normalized.clone(), average_product)));
        B::float_reshape(B::float_cast(value, input_storage), shape)
    });
    let weight_grad = mask[1].then(|| {
        let value = B::float_sum_dim(B::float_mul(grad.clone(), normalized), 0);
        B::float_reshape(B::float_cast(value, weight_storage), Shape::new([width]))
    });
    let bias_grad = mask[2].then(|| B::float_reshape(B::float_sum_dim(grad, 0), Shape::new([width])));
    [input_grad, weight_grad, bias_grad]
}
