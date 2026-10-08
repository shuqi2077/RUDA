use super::float::{FloatUnaryOp, FloatUnaryOpFamily, launch_unary_float};
use ruda_core::tensor::{DType, TensorMetadata};
use ruda_kernel::{dsl::prelude::*, tensor::RudaTensor};
use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{allocation::empty_device_dtype, layout::{address_type, max_vector_size}};

/// SiLU for F32/F16/BF16 inputs, using FP32 arithmetic with one final cast.
/// Retains the tensor's dtype and supports non-contiguous, non-overlapping views.
pub fn launch<R: Runtime>(tensor: RudaTensor<R>) -> RudaTensor<R> {
    assert!(matches!(tensor.dtype, DType::F32 | DType::F16 | DType::BF16));
    if tensor.meta.num_elements() == 0 { return tensor; }
    launch_unary_float::<R, Silu, _>(tensor, |_| {
        SiluOptionsLaunch::new(include_str!("silu.rs").to_owned())
    })
}

#[derive(RudaLaunch, RudaType)]
struct SiluOptions {
    #[ruda(comptime)]
    source: String,
}

struct Silu;

/// Shared SiLU arithmetic for standalone and fused F32/F16/BF16 kernels.
#[ruda]
pub fn apply<F: Float, N: Size>(input: Vector<F, N>) -> Vector<F, N> {
    let input = Vector::<f32, N>::cast_from(input);
    let denominator = Vector::new(1f32) + Vector::exp(-input);
    Vector::cast_from(input / denominator)
}

#[ruda]
impl<F: Float, N: Size> FloatUnaryOp<F, N> for Silu {
    type Options = SiluOptions;

    fn execute(input: Vector<F, N>, _options: &Self::Options) -> Vector<F, N> {
        apply(input)
    }
}

impl FloatUnaryOpFamily for Silu {
    type Options = SiluOptions;
    type Unary<F: Float, N: Size> = Self;
}

/// Independent first-order SiLU VJP with FP32 arithmetic and one final input-storage cast.
/// Input and upstream gradient may independently use F32/F16/BF16 storage.
/// Original primals/gradients are read-only, including non-contiguous views.
pub fn launch_backward<R: Runtime>(input: RudaTensor<R>, grad: RudaTensor<R>) -> RudaTensor<R> {
    assert!(matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16) && input.qparams.is_none(),
        "native SiLU input must use unquantized F32/F16/BF16");
    assert!(matches!(grad.dtype, DType::F32 | DType::F16 | DType::BF16) && grad.qparams.is_none(),
        "native SiLU gradient must use unquantized F32/F16/BF16");
    assert_eq!(input.meta.shape(), grad.meta.shape(), "SiLU gradient shape differs");
    assert_eq!(input.device.to_id(), grad.device.to_id(), "SiLU gradient device differs");
    assert!(input.client.same_execution_queue(&grad.client), "SiLU gradient queue differs");
    if input.meta.num_elements() == 0 { return input; }
    let output = empty_device_dtype(input.client.clone(), input.device.clone(), input.meta.shape().clone(), input.dtype);
    let vector_size = max_vector_size(&input).min(max_vector_size(&grad)).min(max_vector_size(&output));
    let units = input.meta.num_elements() / vector_size as usize;
    let dim = RudaDim::new(input.client.properties(), units);
    let count = calculate_ruda_count_elemwise(&input.client, units, dim);
    let dtypes = [input.dtype.into(), grad.dtype.into()];
    unsafe {
        backward::launch_unchecked::<R>(&output.client, count, dim, address_type!(input, grad, output), vector_size,
            input.into_linear_view_like(&output), grad.into_linear_view_like(&output), output.clone().into_linear_view(),
            include_str!("silu.rs").to_owned(), dtypes);
    }
    output
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn backward<F: Float, G: Float, N: Size>(input: &LinearView<Vector<F, N>>, grad: &LinearView<Vector<G, N>>,
    output: &mut LinearView<Vector<F, N>, ReadWrite>, #[comptime] _source: String,
    #[define(F, G)] _dtypes: [StorageType; 2]) {
    if !output.is_in_bounds(ABSOLUTE_POS) { terminate!(); }
    let input = Vector::<f32, N>::cast_from(input[ABSOLUTE_POS]);
    let grad = Vector::<f32, N>::cast_from(grad[ABSOLUTE_POS]);
    let one = Vector::new(1f32);
    let sigmoid = one / (one + Vector::exp(-input));
    output[ABSOLUTE_POS] = Vector::cast_from((grad * sigmoid) * (one + input * (one - sigmoid)));
}
