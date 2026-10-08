use super::float::{FloatUnaryOp, FloatUnaryOpFamily, launch_unary_float};
use ruda_core::{device::Device, tensor::{DType, TensorMetadata}};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, layout::{address_type, max_vector_size}};

/// FP32-computed erf GELU or the explicitly selected original tanh GELU approximation.
/// Original input storage and geometry are retained; non-contiguous views are supported.
pub fn launch<R: Runtime>(input: RudaTensor<R>, approximate: bool) -> RudaTensor<R> {
    assert!(matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16) && input.qparams.is_none(),
        "native GELU requires unquantized F32/F16/BF16");
    if input.meta.num_elements() == 0 { return input; }
    launch_unary_float::<R, Gelu, _>(input, |_| GeluOptionsLaunch::new(approximate, include_str!("gelu.rs").to_owned()))
}

#[derive(RudaLaunch, RudaType)]
struct GeluOptions {
    #[ruda(comptime)]
    approximate: bool,
    #[ruda(comptime)]
    source: String,
}
struct Gelu;

#[ruda]
impl<F: Float, N: Size> FloatUnaryOp<F, N> for Gelu {
    type Options = GeluOptions;
    fn execute(input: Vector<F, N>, options: &Self::Options) -> Vector<F, N> {
        let input = Vector::<f32, N>::cast_from(input);
        let one = Vector::new(1f32);
        let value = if comptime![options.approximate] {
            let cubic = Vector::powf(input, Vector::new(3f32));
            let inner = (input + cubic * Vector::new(0.044715f32)) * Vector::new(0.7978845608028654f32);
            (input * (Vector::tanh(inner) + one)) * Vector::new(0.5f32)
        } else {
            (input * (Vector::erf(input / Vector::new(core::f32::consts::SQRT_2)) + one)) / Vector::new(2f32)
        };
        Vector::cast_from(value)
    }
}
impl FloatUnaryOpFamily for Gelu {
    type Options = GeluOptions;
    type Unary<F: Float, N: Size> = Self;
}

/// Original erf/tanh mode's first-order VJP with FP32 arithmetic and one input-storage cast.
/// FP32 upstream gradients may be read directly alongside half/BF16 original inputs.
pub fn launch_backward<R: Runtime>(input: RudaTensor<R>, grad: RudaTensor<R>, approximate: bool) -> RudaTensor<R> {
    for value in [&input, &grad] {
        assert!(matches!(value.dtype, DType::F32 | DType::F16 | DType::BF16) && value.qparams.is_none(),
            "native GELU backward requires unquantized F32/F16/BF16");
    }
    assert_eq!(input.meta.shape(), grad.meta.shape(), "GELU gradient shape differs");
    assert_eq!(input.device.to_id(), grad.device.to_id(), "GELU gradient device differs");
    assert!(input.client.same_execution_queue(&grad.client), "GELU gradient queue differs");
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
            approximate, include_str!("gelu.rs").to_owned(), dtypes);
    }
    output
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn backward<F: Float, G: Float, N: Size>(input: &LinearView<Vector<F, N>>, grad: &LinearView<Vector<G, N>>,
    output: &mut LinearView<Vector<F, N>, ReadWrite>, #[comptime] approximate: bool, #[comptime] _source: String,
    #[define(F, G)] _dtypes: [StorageType; 2]) {
    if !output.is_in_bounds(ABSOLUTE_POS) { terminate!(); }
    let input = Vector::<f32, N>::cast_from(input[ABSOLUTE_POS]);
    let grad = Vector::<f32, N>::cast_from(grad[ABSOLUTE_POS]);
    let one = Vector::new(1f32);
    let derivative = if approximate {
        let cubic = Vector::powf(input, Vector::new(3f32));
        let inner = (input + cubic * Vector::new(0.044715f32)) * Vector::new(0.7978845608028654f32);
        let value = Vector::tanh(inner);
        let inner_grad = (one + (input * input) * Vector::new(0.134145f32)) * Vector::new(0.7978845608028654f32);
        (one + value) * Vector::new(0.5f32) + ((input * (one - value * value)) * inner_grad) * Vector::new(0.5f32)
    } else {
        let cdf = (Vector::erf(input / Vector::new(core::f32::consts::SQRT_2)) + one) * Vector::new(0.5f32);
        let pdf = Vector::exp((input * input) * Vector::new(-0.5f32));
        cdf + (input * pdf) * Vector::new(0.3989422804014327f32)
    };
    output[ABSOLUTE_POS] = Vector::cast_from(derivative * grad);
}
