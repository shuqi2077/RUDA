use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::library::tensor::layout::linear::LinearLayout;
use ruda_kernel::library::tensor::layout::*;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;

use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::linear_layout;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::layout::max_vector_size;
use ruda_kernel::tensor::RudaTensor;

#[ruda]
pub(super) fn bicubic_coordinate(
    index: usize,
    input_size: usize,
    output_size: usize,
    #[comptime] align_corners: bool,
) -> f32 {
    let last = input_size - 1;
    if align_corners {
        let denominator = clamp_min(output_size - 1, 1) as f32;
        (index * last) as f32 / denominator
    } else {
        let in_size = (last + 1) as f32;
        let out_size = output_size as f32;
        (index as f32 + 0.5) * (in_size / out_size) - 0.5
    }
}

#[ruda(launch, address_type = "dynamic")]
fn interpolate_bicubic_kernel<F: Float, N: Size>(
    input: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    shape_out: Sequence<FastDivmod<usize>>,
    out_layout: LinearLayout,
    #[comptime] align_corners: bool,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let vector_size = input.vector_size();
    let out_idx = out_layout.to_source_pos(ABSOLUTE_POS);

    let (rem, c) = shape_out[3].div_mod(ABSOLUTE_POS * vector_size);
    let (rem, x) = shape_out[2].div_mod(rem);
    let (b, y) = shape_out[1].div_mod(rem);

    let input_height = input.shape(1) - 1;
    let input_height_f = input_height as f32;

    let frac = bicubic_coordinate(y, input.shape(1), output.shape(1), align_corners);
    let y_in_f = frac.floor();
    let yw = Vector::new(F::cast_from(frac - y_in_f));

    // Clamp indices in float space to handle negative coordinates from half_pixel
    let y0 = clamp(y_in_f - 1.0, 0.0, input_height_f) as usize;
    let y1 = clamp(y_in_f, 0.0, input_height_f) as usize;
    let y2 = clamp(y_in_f + 1.0, 0.0, input_height_f) as usize;
    let y3 = clamp(y_in_f + 2.0, 0.0, input_height_f) as usize;

    let input_width = input.shape(2) - 1;
    let input_width_f = input_width as f32;

    let frac = bicubic_coordinate(x, input.shape(2), output.shape(2), align_corners);
    let x_in_f = frac.floor();
    let xw = Vector::new(F::cast_from(frac - x_in_f));

    // Clamp indices in float space to handle negative coordinates from half_pixel
    let x0 = clamp(x_in_f - 1.0, 0.0, input_width_f) as usize;
    let x1 = clamp(x_in_f, 0.0, input_width_f) as usize;
    let x2 = clamp(x_in_f + 1.0, 0.0, input_width_f) as usize;
    let x3 = clamp(x_in_f + 2.0, 0.0, input_width_f) as usize;

    let index_base = b * input.stride(0) + c * input.stride(3);
    let in_stride_y = input.stride(1);
    let in_stride_x = input.stride(2);

    let y0_stride = y0 * in_stride_y;
    let y1_stride = y1 * in_stride_y;
    let y2_stride = y2 * in_stride_y;
    let y3_stride = y3 * in_stride_y;
    let x0_stride = x0 * in_stride_x;
    let x1_stride = x1 * in_stride_x;
    let x2_stride = x2 * in_stride_x;
    let x3_stride = x3 * in_stride_x;

    let inp_0 = input[(index_base + y0_stride + x0_stride) / vector_size];
    let inp_1 = input[(index_base + y0_stride + x1_stride) / vector_size];
    let inp_2 = input[(index_base + y0_stride + x2_stride) / vector_size];
    let inp_3 = input[(index_base + y0_stride + x3_stride) / vector_size];

    let coefficients0 = cubic_interp_1d(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[(index_base + y1_stride + x0_stride) / vector_size];
    let inp_1 = input[(index_base + y1_stride + x1_stride) / vector_size];
    let inp_2 = input[(index_base + y1_stride + x2_stride) / vector_size];
    let inp_3 = input[(index_base + y1_stride + x3_stride) / vector_size];

    let coefficients1 = cubic_interp_1d(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[(index_base + y2_stride + x0_stride) / vector_size];
    let inp_1 = input[(index_base + y2_stride + x1_stride) / vector_size];
    let inp_2 = input[(index_base + y2_stride + x2_stride) / vector_size];
    let inp_3 = input[(index_base + y2_stride + x3_stride) / vector_size];

    let coefficients2 = cubic_interp_1d(inp_0, inp_1, inp_2, inp_3, xw);

    let inp_0 = input[(index_base + y3_stride + x0_stride) / vector_size];
    let inp_1 = input[(index_base + y3_stride + x1_stride) / vector_size];
    let inp_2 = input[(index_base + y3_stride + x2_stride) / vector_size];
    let inp_3 = input[(index_base + y3_stride + x3_stride) / vector_size];

    let coefficients3 = cubic_interp_1d(inp_0, inp_1, inp_2, inp_3, xw);

    let val = cubic_interp_1d(
        coefficients0,
        coefficients1,
        coefficients2,
        coefficients3,
        yw,
    );

    output[out_idx] = val;
}

#[ruda]
fn cubic_interp_1d<F: Float, N: Size>(
    x0: Vector<F, N>,
    x1: Vector<F, N>,
    x2: Vector<F, N>,
    x3: Vector<F, N>,
    t: Vector<F, N>,
) -> Vector<F, N> {
    let coeffs0 = cubic_coefficient(t, 0usize);
    let coeffs1 = cubic_coefficient(t, 1usize);
    let coeffs2 = cubic_coefficient(t, 2usize);
    let coeffs3 = cubic_coefficient(t, 3usize);

    x0 * coeffs0 + x1 * coeffs1 + x2 * coeffs2 + x3 * coeffs3
}

#[ruda]
pub(crate) fn cubic_coefficient<F: Float, N: Size>(
    t: Vector<F, N>,
    #[comptime] tap: usize,
) -> Vector<F, N> {
    let a = float(-0.75);
    if tap == 0 {
        cubic_convolution_2(t + float(1.0), a)
    } else if tap == 1 {
        cubic_convolution_1(t, a)
    } else if tap == 2 {
        cubic_convolution_1(float(1.0) - t, a)
    } else {
        cubic_convolution_2(float(2.0) - t, a)
    }
}

#[ruda]
fn cubic_convolution_1<F: Float, N: Size>(x: Vector<F, N>, a: Vector<F, N>) -> Vector<F, N> {
    let conv = (a + float(2.0)) * x;
    let tmp = a + float(3.0);
    (conv - tmp) * x * x + float(1.0)
}

#[ruda]
fn cubic_convolution_2<F: Float, N: Size>(x: Vector<F, N>, a: Vector<F, N>) -> Vector<F, N> {
    let conv = a * x;
    let conv = (conv - float(5.0) * a) * x;
    let tmp = float(8.0) * a;
    let conv = (conv + tmp) * x;

    conv - float(4.0) * a
}

#[ruda]
fn float<F: Float, N: Size>(#[comptime] v: f32) -> Vector<F, N> {
    Vector::new(F::new(v))
}

pub fn interpolate_bicubic_launch<R: Runtime>(
    input: RudaTensor<R>,
    output: RudaTensor<R>,
    align_corners: bool,
) -> RudaTensor<R> {
    let vector_size = max_vector_size(&input);
    let out_shape = shape_divmod(&output);
    let out_layout = linear_layout(&output, vector_size);

    let working_units = output.meta.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(input.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&input.client, working_units, ruda_dim);

    interpolate_bicubic_kernel::launch(
        &output.client,
        ruda_count,
        ruda_dim,
        address_type!(input, output),
        vector_size,
        input.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        out_shape,
        out_layout,
        align_corners,
        output.dtype.into(),
    );

    output
}
