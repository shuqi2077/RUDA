use crate::{Backend, DType, FloatDType, Shape, TensorMetadata, get_device_settings, tensor::FloatTensor};

/// Actual PReLU channel geometry; the original slope remains a vector rather than a broadcast allocation.
pub struct PreluGeometry {
    /// Logical channel count, or one for rank-one shared-slope input.
    pub channels: usize,
    /// Product of trailing spatial extents.
    pub spatial: usize,
    /// Number of actual slope parameters, either one or the original channel count.
    pub parameters: usize,
    /// Actual input element count.
    pub elements: usize,
}

/// Validate shared or channel-wise slopes without replacing them with expanded copies.
pub fn geometry(input: &Shape, alpha: &Shape) -> PreluGeometry {
    assert!(input.num_dims() > 0, "PReLU requires at least one input axis");
    assert_eq!(alpha.num_dims(), 1, "PReLU slope must be a vector");
    let parameters = alpha[0];
    let channels = if input.num_dims() >= 2 { input[1] } else { 1 };
    assert!(parameters == 1 || (input.num_dims() >= 2 && parameters == channels), "PReLU slope must be shared or match actual channels");
    let spatial = if input.num_dims() >= 2 {
        input[2..].iter().try_fold(1usize, |size, &extent| size.checked_mul(extent)).expect("PReLU spatial overflow")
    } else { 1 };
    let elements = input.iter().try_fold(1usize, |size, &extent| size.checked_mul(extent)).expect("PReLU element count overflow");
    PreluGeometry { channels, spatial, parameters, elements }
}

fn broadcast(shape: &Shape, parameters: usize) -> Shape {
    let mut dims = alloc::vec![1; shape.num_dims()];
    if parameters != 1 { dims[1] = parameters; }
    Shape::from(dims)
}

/// Working-storage PReLU on the same backend, retaining the original storage only at output.
pub fn prelu_native<B: Backend>(input: FloatTensor<B>, alpha: FloatTensor<B>) -> FloatTensor<B> {
    let shape = input.shape();
    let info = geometry(&shape, &alpha.shape());
    if info.elements == 0 { return input; }
    let storage: FloatDType = input.dtype().into();
    let compute = if input.dtype() == DType::F64 || alpha.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    let input = B::float_cast(input, compute);
    let alpha = B::float_reshape(B::float_cast(alpha, compute), broadcast(&shape, info.parameters));
    B::float_cast(B::prelu(input, alpha), storage)
}

/// Independently selected input and slope derivatives; the zero boundary uses the nonnegative branch.
pub fn prelu_native_backward_select<B: Backend>(input: FloatTensor<B>, alpha: FloatTensor<B>, grad: FloatTensor<B>,
    mask: [bool; 2]) -> [Option<FloatTensor<B>>; 2] {
    if mask == [false; 2] { return [None, None]; }
    let shape = input.shape();
    let alpha_shape = alpha.shape();
    let info = geometry(&shape, &alpha_shape);
    assert_eq!(grad.shape(), shape, "PReLU gradient shape differs");
    let storage: FloatDType = input.dtype().into();
    let alpha_storage: FloatDType = alpha.dtype().into();
    if info.elements == 0 {
        let device = B::float_device(&alpha);
        return [mask[0].then_some(input), mask[1].then(|| B::float_zeros(alpha_shape, &device, alpha_storage))];
    }
    let compute = if [&input, &alpha, &grad].into_iter().any(|value| value.dtype() == DType::F64) { FloatDType::F64 } else { FloatDType::F32 };
    let input = B::float_cast(input, compute);
    let grad = B::float_cast(grad, compute);
    let bool_dtype = get_device_settings::<B>(&B::float_device(&input)).bool_dtype;
    let negative = B::float_lower_elem(input.clone(), 0f32.into(), bool_dtype);
    let input_grad = mask[0].then(|| {
        let alpha = B::float_reshape(B::float_cast(alpha, compute), broadcast(&shape, info.parameters));
        let scaled = B::float_mul(grad.clone(), alpha);
        B::float_cast(B::float_mask_where(grad.clone(), negative.clone(), scaled), storage)
    });
    let weight_grad = mask[1].then(|| {
        let nonnegative = B::bool_not(negative);
        let product = B::float_mask_fill(B::float_mul(input, grad), nonnegative, 0f32.into());
        let output = if info.parameters == 1 { B::float_sum(product) } else {
            let batch = shape[0];
            let product = B::float_reshape(product, Shape::new([batch, info.channels, info.spatial]));
            B::float_sum_dim(B::float_sum_dim(product, 0), 2)
        };
        B::float_reshape(B::float_cast(output, alpha_storage), alpha_shape)
    });
    [input_grad, weight_grad]
}
