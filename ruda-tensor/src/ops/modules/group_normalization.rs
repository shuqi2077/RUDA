use super::LayerNormOutput;
use crate::{Backend, DType, FloatDType, Shape, TensorMetadata, tensor::FloatTensor};

/// Actual logical channel-group geometry, shared by backend dispatch and operation graphs.
pub struct GroupNormGeometry {
    /// Original batch extent, including empty batches.
    pub batch: usize,
    /// Original channel extent.
    pub channels: usize,
    /// Product of all original trailing spatial extents; rank-two input uses one.
    pub spatial: usize,
    /// Number of actual elements normalized in each channel group.
    pub width: usize,
    /// Number of actual independent batch/group rows.
    pub rows: usize,
}

/// Validate actual `[batch, channels, ...]` input and evenly divided nonempty groups.
pub fn geometry(shape: &Shape, groups: usize) -> GroupNormGeometry {
    assert!(shape.num_dims() >= 2, "GroupNorm requires batch and channel axes");
    let channels = shape[1];
    assert!(groups > 0 && channels > 0 && channels % groups == 0, "GroupNorm channels must divide into positive groups");
    let spatial = shape[2..].iter().try_fold(1usize, |size, &extent| size.checked_mul(extent)).expect("GroupNorm spatial overflow");
    let width = (channels / groups).checked_mul(spatial).expect("GroupNorm group width overflow");
    assert!(width > 0, "GroupNorm normalized groups must be nonempty");
    let rows = shape[0].checked_mul(groups).expect("GroupNorm row count overflow");
    rows.checked_mul(width).expect("GroupNorm element count overflow");
    GroupNormGeometry { batch: shape[0], channels, spatial, width, rows }
}

fn affine<B: Backend>(value: &FloatTensor<B>, channels: usize) {
    assert_eq!(value.shape(), Shape::new([channels]), "GroupNorm affine must match actual channels");
}

/// Same-device biased-variance GroupNorm with FP32/FP64 saved `[batch, groups]` statistics.
pub fn group_norm_with_stats<B: Backend>(input: FloatTensor<B>, gamma: Option<FloatTensor<B>>, beta: Option<FloatTensor<B>>,
    groups: usize, epsilon: f64) -> LayerNormOutput<B> {
    let shape = input.shape();
    let info = geometry(&shape, groups);
    for value in gamma.iter().chain(beta.iter()) { affine::<B>(value, info.channels); }
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    if info.rows == 0 {
        let device = B::float_device(&input);
        return LayerNormOutput { output: input, mean: B::float_zeros(Shape::new([0, groups]), &device, compute),
            rstd: B::float_zeros(Shape::new([0, groups]), &device, compute) };
    }
    let input = B::float_reshape(B::float_cast(input, compute), Shape::new([info.rows, info.width]));
    let mean = B::float_div_scalar(B::float_sum_dim(input.clone(), 1), (info.width as f64).into());
    let centered = B::float_sub(input, mean.clone());
    let variance = B::float_div_scalar(B::float_sum_dim(B::float_mul(centered.clone(), centered.clone()), 1), (info.width as f64).into());
    let rstd = B::float_recip(B::float_sqrt(B::float_add_scalar(variance, epsilon.into())));
    let mut output = B::float_reshape(B::float_mul(centered, rstd.clone()), Shape::new([info.batch, info.channels, info.spatial]));
    if let Some(gamma) = gamma {
        output = B::float_mul(output, B::float_reshape(B::float_cast(gamma, compute), Shape::new([1, info.channels, 1])));
    }
    if let Some(beta) = beta {
        output = B::float_add(output, B::float_reshape(B::float_cast(beta, compute), Shape::new([1, info.channels, 1])));
    }
    LayerNormOutput { output: B::float_reshape(B::float_cast(output, storage), shape),
        mean: B::float_reshape(mean, Shape::new([info.batch, groups])), rstd: B::float_reshape(rstd, Shape::new([info.batch, groups])) }
}

/// Independently requested input/actual-weight/bias derivatives on the same backend.
pub fn group_norm_backward_select<B: Backend>(input: FloatTensor<B>, gamma: Option<FloatTensor<B>>, grad: FloatTensor<B>,
    mean: FloatTensor<B>, rstd: FloatTensor<B>, groups: usize, mask: [bool; 3]) -> [Option<FloatTensor<B>>; 3] {
    if mask == [false; 3] { return [None, None, None]; }
    let shape = input.shape();
    let info = geometry(&shape, groups);
    if let Some(gamma) = &gamma { affine::<B>(gamma, info.channels); }
    assert!(!mask[1] || gamma.is_some(), "GroupNorm weight gradient requires an actual weight");
    assert_eq!(grad.shape(), shape, "GroupNorm gradient shape differs");
    assert_eq!(mean.shape(), Shape::new([info.batch, groups]), "GroupNorm mean shape differs");
    assert_eq!(rstd.shape(), mean.shape(), "GroupNorm reciprocal deviation shape differs");
    let storage: FloatDType = input.dtype().into();
    let forward_compute = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    let compute = if [&input, &grad, &mean, &rstd].into_iter().chain(gamma.iter()).any(|value| value.dtype() == DType::F64) {
        FloatDType::F64
    } else { FloatDType::F32 };
    if info.rows == 0 {
        let device = B::float_device(&input);
        return [mask[0].then_some(input), mask[1].then(|| B::float_zeros(Shape::new([info.channels]), &device,
            gamma.as_ref().expect("requested GroupNorm weight").dtype().into())),
            mask[2].then(|| B::float_zeros(Shape::new([info.channels]), &device, compute))];
    }
    let grad = B::float_reshape(B::float_cast(grad, compute), Shape::new([info.batch, info.channels, info.spatial]));
    let bias_grad = mask[2].then(|| B::float_reshape(B::float_sum_dim(B::float_sum_dim(grad.clone(), 0), 2), Shape::new([info.channels])));
    if !mask[0] && !mask[1] { return [None, None, bias_grad]; }
    let input = B::float_reshape(B::float_cast(input, forward_compute), Shape::new([info.rows, info.width]));
    let mean = B::float_reshape(B::float_cast(mean, forward_compute), Shape::new([info.rows, 1]));
    let rstd = B::float_reshape(B::float_cast(rstd, forward_compute), Shape::new([info.rows, 1]));
    let normalized = B::float_cast(B::float_mul(B::float_sub(input, mean), rstd.clone()), compute);
    let normalized_channels = B::float_reshape(normalized.clone(), Shape::new([info.batch, info.channels, info.spatial]));
    let input_grad = mask[0].then(|| {
        let scaled = if let Some(gamma) = &gamma {
            let gamma = B::float_cast(B::float_cast(gamma.clone(), forward_compute), compute);
            B::float_mul(grad.clone(), B::float_reshape(gamma, Shape::new([1, info.channels, 1])))
        } else { grad.clone() };
        let scaled = B::float_reshape(scaled, Shape::new([info.rows, info.width]));
        let average = B::float_div_scalar(B::float_sum_dim(scaled.clone(), 1), (info.width as f64).into());
        let product = B::float_div_scalar(B::float_sum_dim(B::float_mul(scaled.clone(), normalized.clone()), 1), (info.width as f64).into());
        let output = B::float_mul(B::float_cast(rstd, compute), B::float_sub(B::float_sub(scaled, average), B::float_mul(normalized, product)));
        B::float_reshape(B::float_cast(output, storage), shape)
    });
    let weight_grad = mask[1].then(|| {
        let output = B::float_sum_dim(B::float_sum_dim(B::float_mul(grad, normalized_channels), 0), 2);
        B::float_reshape(B::float_cast(output, gamma.as_ref().expect("requested GroupNorm weight").dtype().into()), Shape::new([info.channels]))
    });
    [input_grad, weight_grad, bias_grad]
}
