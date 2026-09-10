use ruda_kernel::dsl as kernel_dsl;
use ruda_core::tensor::DType;
use ruda_core::tensor::TensorMetadata;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::linear::LinearView;

use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::max_vector_size_many;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::RudaTensor;

#[ruda(launch_unchecked, address_type = "dynamic")]
fn mask_fill_kernel<T: Numeric, B: Int, N: Size>(
    input: &LinearView<Vector<T, N>>,
    mask: &LinearView<Vector<B, N>>,
    output: &mut LinearView<Vector<T, N>, ReadWrite>,
    value: InputScalar,
    #[define(T, B)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let mask = Vector::cast_from(mask[ABSOLUTE_POS]);
    let input = input[ABSOLUTE_POS];
    let value = Vector::new(value.get::<T>());

    output[ABSOLUTE_POS] = select_many(mask, value, input);
}

#[derive(Clone, Copy, Debug)]
/// Define how to run the mask fill kernel.
///
/// # Notes
///
/// All assertions should be done before choosing the strategy.
pub enum MaskFillStrategy {
    /// Don't mutate any input.
    Readonly,
    /// Reuse the input tensor inplace.
    Inplace,
}

/// Execute the mask fill kernel with the given strategy.
pub fn mask_fill<R: Runtime>(
    input: RudaTensor<R>,
    mask: RudaTensor<R>,
    value: InputScalar,
    strategy: MaskFillStrategy,
    dtype_bool: DType,
) -> RudaTensor<R> {
    let ndims = input.meta.num_dims();
    let output = match strategy {
        MaskFillStrategy::Readonly => empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            input.shape(),
            input.dtype,
        ),
        MaskFillStrategy::Inplace => input.clone(),
    };

    let vector_size = max_vector_size_many(&[&input, &mask], ndims - 1);
    let working_units = input.meta.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(input.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&input.client, working_units, ruda_dim);

    let out_arg = match strategy {
        MaskFillStrategy::Readonly => output.clone().into_linear_view(),
        MaskFillStrategy::Inplace => output.as_linear_view_alias(0),
    };

    let at = address_type!(input, mask, output);
    let mask = mask.into_linear_view_like(&input);

    unsafe {
        mask_fill_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            at,
            vector_size,
            input.into_linear_view(),
            mask,
            out_arg,
            value,
            [output.dtype.into(), dtype_bool.into()],
        );
    }

    output
}
