use ruda_kernel::dsl as kernel_dsl;
use crate::elementwise::binary::numeric::AddOp;
use crate::elementwise::binary::numeric::BinaryOp;
use crate::elementwise::binary::numeric::BinaryOpFamily;
use crate::elementwise::binary::numeric::OrOp;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::RudaTensor;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;

/// Uses checked launch mode because user-provided `indices` may contain out-of-bounds values
/// that would cause invalid writes into `tensor`. Checked mode clamps these accesses rather
/// than producing undefined behavior.
#[ruda(launch, address_type = "dynamic")]
fn select_assign_kernel<F: Numeric, I: Numeric, Op: BinaryOpFamily>(
    tensor: &mut Tensor<F>,
    indices: &LinearView<I>,
    value: &Tensor<F>,
    value_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[comptime] axis: usize,
    #[define(F, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let rank = value_shape.len().comptime();

    let mut offset = ABSOLUTE_POS;
    let mut offset_tensor = 0;
    let mut offset_value = 0;

    // Calculate offsets and num_elems
    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        if i != axis {
            let (rem, local_pos) = value_shape[i].div_mod(offset);
            offset = rem;

            offset_tensor += local_pos * tensor.stride(i);
            offset_value += local_pos * value.stride(i);
        }
    }

    let strides_tensor_dim = tensor.stride(axis);
    let strides_value_dim = value.stride(axis);

    // Main operation
    for i in 0..value.shape(axis) {
        let index_tensor = usize::cast_from(indices[i]) * strides_tensor_dim + offset_tensor;
        let index_value = i * strides_value_dim + offset_value;

        let value = Op::BinaryOp::<F, Const<1>>::execute(
            Vector::cast_from(tensor[index_tensor]),
            Vector::cast_from(value[index_value]),
        );
        tensor[index_tensor] = F::cast_from(value);
    }
}

pub fn select_assign<R: Runtime>(
    tensor: RudaTensor<R>,
    axis: usize,
    indices: RudaTensor<R>,
    value: RudaTensor<R>,
    is_bool: bool,
) -> RudaTensor<R> {
    if value.meta.num_elements() == 0 {
        return tensor;
    }

    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    let working_units = value.meta.num_elements() / value.meta.shape()[axis];
    let ruda_dim = RudaDim::new(indices.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&indices.client, working_units, ruda_dim);

    let launch = match is_bool {
        true => select_assign_kernel::launch::<OrOp, R>,
        false => select_assign_kernel::launch::<AddOp, R>,
    };

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    let shape = shape_divmod(&value);
    launch(
        &tensor.client,
        ruda_count,
        ruda_dim,
        address_type!(tensor, indices, value),
        tensor.clone().into_tensor_arg(),
        indices.into_linear_view(),
        value.into_tensor_arg(),
        shape,
        working_units,
        axis,
        [tensor_dtype.into(), indices_dtype.into()],
    );

    tensor
}
