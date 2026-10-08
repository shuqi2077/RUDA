use ruda_core::{device::Device, tensor::{DType, Shape}};
use ruda_kernel::{dsl::{calculate_ruda_count_elemwise, prelude::*},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous}};
use ruda_kernel::dsl as kernel_dsl;

struct Layout { channels: usize, spatial: usize, parameters: usize, elements: usize }

fn layout<R: Runtime>(input: &RudaTensor<R>, alpha: &RudaTensor<R>) -> Layout {
    let shape = input.meta.shape();
    assert!(shape.num_dims() > 0 && alpha.meta.shape().num_dims() == 1, "PReLU requires an input and an actual slope vector");
    let parameters = alpha.meta.shape()[0];
    let channels = if shape.num_dims() >= 2 { shape[1] } else { 1 };
    assert!(parameters == 1 || (shape.num_dims() >= 2 && parameters == channels), "PReLU slopes differ from actual channels");
    let spatial = if shape.num_dims() >= 2 {
        shape[2..].iter().try_fold(1usize, |size, &extent| size.checked_mul(extent)).expect("PReLU spatial overflow")
    } else { 1 };
    let elements = shape.iter().try_fold(1usize, |size, &extent| size.checked_mul(extent)).expect("PReLU element count overflow");
    assert!(elements <= u32::MAX as usize && parameters <= u32::MAX as usize && channels <= u32::MAX as usize
        && spatial <= u32::MAX as usize, "native PReLU index range exceeded");
    for value in [input, alpha] { binding(input, value); }
    Layout { channels, spatial, parameters, elements }
}

fn binding<R: Runtime>(input: &RudaTensor<R>, value: &RudaTensor<R>) {
    assert!(value.qparams.is_none() && matches!(value.dtype, DType::F32 | DType::F16 | DType::BF16), "native PReLU requires unquantized F32/F16/BF16");
    assert_eq!(value.device.to_id(), input.device.to_id(), "PReLU device differs");
    assert!(value.client.same_execution_queue(&input.client), "PReLU execution queue differs");
}

/// Native channel/shared-slope PReLU with FP32 arithmetic and one final original-storage cast.
pub fn launch<R: Runtime>(input: RudaTensor<R>, alpha: RudaTensor<R>) -> RudaTensor<R> {
    let info = layout(&input, &alpha);
    if info.elements == 0 { return input; }
    let input = into_contiguous(input);
    let alpha = into_contiguous(alpha);
    let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), input.meta.shape().clone(), input.dtype);
    let dim = RudaDim::new(input.client.properties(), info.elements);
    let count = calculate_ruda_count_elemwise(&input.client, info.elements, dim);
    forward::launch(&input.client, count, dim, input.clone().into_array_arg(), alpha.clone().into_array_arg(), output.clone().into_array_arg(),
        info.channels as u32, info.spatial as u32, info.parameters == 1, include_str!("prelu.rs").to_owned(), [input.dtype.into(), alpha.dtype.into()]);
    output
}

/// Selected PReLU VJPs; absent gradients allocate neither output nor reduction scratch.
pub fn launch_backward_select<R: Runtime>(input: RudaTensor<R>, alpha: RudaTensor<R>, grad: RudaTensor<R>,
    mask: [bool; 2]) -> [Option<RudaTensor<R>>; 2] {
    if mask == [false; 2] { return [None, None]; }
    let info = layout(&input, &alpha);
    binding(&input, &grad);
    assert_eq!(input.meta.shape(), grad.meta.shape(), "PReLU gradient shape differs");
    let client = input.client.clone();
    let device = input.device.clone();
    let allocate = |shape, dtype| empty_device_contiguous_dtype(client.clone(), device.clone(), shape, dtype);
    let dx = mask[0].then(|| allocate(input.meta.shape().clone(), input.dtype));
    let da = mask[1].then(|| allocate(alpha.meta.shape().clone(), alpha.dtype));
    let input = into_contiguous(input);
    let grad = into_contiguous(grad);
    if let Some(dx) = &dx {
        if info.elements > 0 {
            let alpha = into_contiguous(alpha.clone());
            let dim = RudaDim::new(client.properties(), info.elements);
            let count = calculate_ruda_count_elemwise(&client, info.elements, dim);
            input_backward::launch(&client, count, dim, input.clone().into_array_arg(), alpha.clone().into_array_arg(), grad.clone().into_array_arg(),
                dx.clone().into_array_arg(), info.channels as u32, info.spatial as u32, info.parameters == 1,
                [input.dtype.into(), alpha.dtype.into(), grad.dtype.into()]);
        }
    }
    if let Some(da) = &da {
        if info.parameters > 0 {
            let rows = info.elements / info.parameters;
            let parts = rows.div_ceil(32).clamp(1, 128).min(u32::MAX as usize / info.parameters);
            let work = parts * info.parameters;
            let partial = allocate(Shape::new([parts, info.parameters]), DType::F32);
            let dim = RudaDim::new(client.properties(), work);
            let count = calculate_ruda_count_elemwise(&client, work, dim);
            weight_partial::launch(&client, count, dim, input.clone().into_array_arg(), grad.clone().into_array_arg(), partial.clone().into_array_arg(),
                info.parameters as u32, info.spatial as u32, parts as u32, info.parameters == 1, [input.dtype.into(), grad.dtype.into()]);
            let dim = RudaDim::new(client.properties(), info.parameters);
            let count = calculate_ruda_count_elemwise(&client, info.parameters, dim);
            weight_merge::launch(&client, count, dim, partial.into_array_arg(), da.clone().into_array_arg(), parts as u32, da.dtype.into());
        }
    }
    [dx, da]
}

#[ruda(launch)]
fn forward<F: Float, W: Float>(input: &Array<F>, alpha: &Array<W>, output: &mut Array<F>, channels: u32, spatial: u32,
    #[comptime] shared: bool, #[comptime] _source: String, #[define(F, W)] _types: [StorageType; 2]) {
    let i = ABSOLUTE_POS as usize;
    if i >= output.len() { terminate!(); }
    let parameter = if shared { 0usize } else { (i / spatial as usize) % channels as usize };
    let value = f32::cast_from(input[i]);
    output[i] = F::cast_from(if value < 0f32 { value * f32::cast_from(alpha[parameter]) } else { value });
}

#[ruda(launch)]
fn input_backward<F: Float, W: Float, G: Float>(input: &Array<F>, alpha: &Array<W>, grad: &Array<G>, output: &mut Array<F>,
    channels: u32, spatial: u32, #[comptime] shared: bool, #[define(F, W, G)] _types: [StorageType; 3]) {
    let i = ABSOLUTE_POS as usize;
    if i >= output.len() { terminate!(); }
    let parameter = if shared { 0usize } else { (i / spatial as usize) % channels as usize };
    let value = f32::cast_from(grad[i]);
    output[i] = F::cast_from(if f32::cast_from(input[i]) < 0f32 { value * f32::cast_from(alpha[parameter]) } else { value });
}

#[ruda(launch)]
fn weight_partial<F: Float, G: Float>(input: &Array<F>, grad: &Array<G>, partial: &mut Array<f32>, parameters: u32, spatial: u32,
    parts: u32, #[comptime] shared: bool, #[define(F, G)] _types: [StorageType; 2]) {
    let i = ABSOLUTE_POS as usize;
    if i >= partial.len() { terminate!(); }
    let parameter = i % parameters as usize;
    let mut row = i / parameters as usize;
    let rows = input.len() / parameters as usize;
    let mut sum = 0f32;
    while row < rows {
        let index = if shared { row } else { (row / spatial as usize * parameters as usize + parameter) * spatial as usize + row % spatial as usize };
        let value = f32::cast_from(input[index]);
        if value < 0f32 { sum += value * f32::cast_from(grad[index]); }
        if rows - row <= parts as usize { break; }
        row += parts as usize;
    }
    partial[i] = sum;
}

#[ruda(launch)]
fn weight_merge<W: Float>(partial: &Array<f32>, output: &mut Array<W>, parts: u32, #[define(W)] _storage: StorageType) {
    let parameter = ABSOLUTE_POS as usize;
    if parameter >= output.len() { terminate!(); }
    let mut sum = 0f32;
    for part in 0u32..parts { sum += partial[part as usize * output.len() + parameter]; }
    output[parameter] = W::cast_from(sum);
}
