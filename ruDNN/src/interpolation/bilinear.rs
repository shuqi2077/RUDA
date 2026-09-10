use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::dsl::num_traits::Zero;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::library::tensor::layout::linear::LinearLayout;
use ruda_kernel::library::tensor::layout::*;

use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::linear_layout;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::layout::max_vector_size;
use ruda_kernel::tensor::RudaTensor;

#[ruda]
pub(super) fn bilinear_coordinate(
    index: usize,
    input_size: usize,
    output_size: usize,
    #[comptime] align_corners: bool,
) -> f32 {
    if align_corners {
        let numerator = (input_size - 1) as f32;
        let denominator = clamp_min(output_size - 1, 1) as f32;
        index as f32 * (numerator / denominator)
    } else {
        let in_size = input_size as f32;
        let out_size = output_size as f32;
        clamp(
            (index as f32 + 0.5) * (in_size / out_size) - 0.5,
            0.0,
            in_size - 1.0,
        )
    }
}

#[ruda(launch, address_type = "dynamic")]
fn interpolate_bilinear_kernel<F: Float, N: Size>(
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

    let frac = bilinear_coordinate(y, input.shape(1), output.shape(1), align_corners);

    let v0 = frac.floor();
    let v1 = frac.ceil();
    let yw = F::cast_from(frac - v0);
    let yw_ = Vector::new(F::one() - yw);
    let yw = Vector::new(yw);
    let y0_ok = v0 >= 0.0;
    let y0 = v0 as usize;
    let y1 = v1 as usize;

    let frac = bilinear_coordinate(x, input.shape(2), output.shape(2), align_corners);
    let v0 = frac.floor();
    let v1 = frac.ceil();
    let xw = F::cast_from(frac - v0);
    let xw_ = Vector::new(F::one() - xw);
    let xw = Vector::new(xw);
    let x0_ok = v0 >= 0.0;
    let x0 = v0 as usize;
    let x1 = v1 as usize;

    let index_base = b * input.stride(0) + c * input.stride(3);

    let in_stride_y = input.stride(1);
    let in_stride_x = input.stride(2);

    let y0_stride = y0 * in_stride_y;
    let y1_stride = y1 * in_stride_y;
    let x0_stride = x0 * in_stride_x;
    let x1_stride = x1 * in_stride_x;

    let height = input.shape(1);
    let width = input.shape(2);

    let y1_ok = y1 < height;
    let x1_ok = x1 < width;

    let zero = Vector::zero();

    let p_a = select(
        x0_ok && y0_ok,
        input[(index_base + y0_stride + x0_stride) / vector_size] * xw_ * yw_,
        zero,
    );
    let p_b = select(
        x1_ok && y0_ok,
        input[(index_base + y0_stride + x1_stride) / vector_size] * xw * yw_,
        zero,
    );
    let p_c = select(
        x0_ok && y1_ok,
        input[(index_base + y1_stride + x0_stride) / vector_size] * xw_ * yw,
        zero,
    );
    let p_d = select(
        x1_ok && y1_ok,
        input[(index_base + y1_stride + x1_stride) / vector_size] * xw * yw,
        zero,
    );

    output[out_idx] = p_a + p_b + p_c + p_d;
}

pub fn interpolate_bilinear_launch<R: Runtime>(
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

    interpolate_bilinear_kernel::launch(
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
