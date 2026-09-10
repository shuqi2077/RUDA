use ruda_kernel::dsl as kernel_dsl;
use ruda_core::tensor::DType;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::linear::LinearView;

use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::broadcast_shape;
use ruda_kernel::tensor::layout::max_vector_size_many;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::RudaTensor;

#[ruda(launch, address_type = "dynamic")]
fn mask_where_kernel<T: Numeric, B: Int, N: Size>(
    input: &LinearView<Vector<T, N>>,
    value: &LinearView<Vector<T, N>>,
    mask: &LinearView<Vector<B, N>>,
    output: &mut LinearView<Vector<T, N>, ReadWrite>,
    #[define(T, B)] _dtypes: [StorageType; 2],
) {
    let pos = ABSOLUTE_POS;
    if !output.is_in_bounds(pos) {
        terminate!();
    }

    output[pos] = select_many(Vector::cast_from(mask[pos]), value[pos], input[pos]);
}

#[derive(Clone, Copy, Debug)]
/// Define how to run the mask where kernel.
///
/// # Notes
///
/// All assertions should be done before choosing the strategy.
pub enum MaskWhereStrategy {
    /// Don't mutate any input.
    Readonly,
    /// Reuse the lhs tensor inplace.
    InplaceLhs,
    /// Reuse the rhs tensor inplace.
    InplaceRhs,
}

/// Execute the mask where kernel with the given strategy.
pub fn mask_where<R: Runtime>(
    input: RudaTensor<R>,
    mask: RudaTensor<R>,
    value: RudaTensor<R>,
    strategy: MaskWhereStrategy,
    dtype_bool: DType,
) -> RudaTensor<R> {
    let vector_size = max_vector_size_many(&[&input, &mask, &value], input.meta.num_dims() - 1);

    let working_units = input.meta.num_elements() / vector_size as usize;
    let ruda_dim = RudaDim::new(input.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&input.client, working_units, ruda_dim);

    let out_shape = broadcast_shape(&[&input, &mask, &value]);

    let output = match strategy {
        MaskWhereStrategy::Readonly => empty_device_dtype(
            input.client.clone(),
            input.device.clone(),
            out_shape,
            input.dtype,
        ),
        MaskWhereStrategy::InplaceLhs => input.clone(),
        MaskWhereStrategy::InplaceRhs => value.clone(),
    };

    let out = match strategy {
        MaskWhereStrategy::Readonly => output.clone().into_linear_view(),
        MaskWhereStrategy::InplaceLhs => output.as_linear_view_alias(0),
        MaskWhereStrategy::InplaceRhs => output.as_linear_view_alias(1),
    };

    mask_where_kernel::launch(
        &output.client,
        ruda_count,
        ruda_dim,
        address_type!(input, value, mask, output),
        vector_size,
        input.into_linear_view_like(&output),
        value.into_linear_view_like(&output),
        mask.into_linear_view_like(&output),
        out,
        [output.dtype.into(), dtype_bool.into()],
    );

    output
}
