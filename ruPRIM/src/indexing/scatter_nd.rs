use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use crate::elementwise::binary::numeric::AssignOp;
use crate::elementwise::binary::numeric::BinaryMaxOp;
use crate::elementwise::binary::numeric::BinaryMinOp;
use crate::elementwise::binary::numeric::BinaryOp;
use crate::elementwise::binary::numeric::BinaryOpFamily;
use crate::elementwise::binary::numeric::MulOp;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::layout::shape_divmod_range;
use ruda_kernel::tensor::RudaTensor;
use ruda_core::tensor::indexing::IndexingUpdateOp;
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;

mod add;

/// scatter_nd GPU kernel.
///
/// Each thread handles one element across all update slices.
/// Work items = num_updates * slice_size.
#[ruda(launch_unchecked, address_type = "dynamic")]
fn scatter_nd_kernel<T: Numeric, I: Int, Op: BinaryOpFamily>(
    data: &mut Tensor<T>,
    indices: &LinearView<I>,
    values: &Tensor<T>,
    data_slice_shape: Sequence<FastDivmod<usize>>,
    values_shape: Sequence<FastDivmod<usize>>,
    slice_size: usize,
    k: usize,
    working_units: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let slice_offset = ABSOLUTE_POS % slice_size;
    let update_idx = ABSOLUTE_POS / slice_size;

    let idx_base = update_idx * k;
    let mut base_offset = 0usize;
    for j in 0..k {
        let idx_val = usize::cast_from(indices[idx_base + j]);
        base_offset += idx_val * data.stride(j);
    }

    // Decompose slice_offset over data's trailing dims (k..n)
    let slice_rank = data_slice_shape.len().comptime();
    let mut data_slice_offset = 0usize;
    let mut remainder = slice_offset;
    #[unroll]
    for i in 0..slice_rank {
        let dim = slice_rank - i - 1;
        let (rem, coord) = data_slice_shape[dim].div_mod(remainder);
        remainder = rem;
        data_slice_offset += coord * data.stride(k + dim);
    }

    let val_rank = values_shape.len().comptime();
    let mut val_offset = 0usize;
    let mut remainder_v = ABSOLUTE_POS;
    #[unroll]
    for i in 0..val_rank {
        let dim = val_rank - i - 1;
        let (rem, coord) = values_shape[dim].div_mod(remainder_v);
        remainder_v = rem;
        val_offset += coord * values.stride(dim);
    }

    let data_idx = base_offset + data_slice_offset;
    let result = Op::BinaryOp::<T, Const<1>>::execute(
        Vector::cast_from(data[data_idx]),
        Vector::cast_from(values[val_offset]),
    );
    data[data_idx] = result[0];
}

pub fn scatter_nd<R: Runtime>(
    tensor: RudaTensor<R>,
    indices: RudaTensor<R>,
    values: RudaTensor<R>,
    reduction: IndexingUpdateOp,
) -> RudaTensor<R> {
    if values.meta.num_elements() == 0 {
        return tensor;
    }

    // Ensure we can write in-place
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    let launch = match reduction {
        IndexingUpdateOp::Assign => scatter_nd_kernel::launch_unchecked::<AssignOp, R>,
        IndexingUpdateOp::Add => return add::scatter_nd_add(tensor, indices, values),
        IndexingUpdateOp::Mul => scatter_nd_kernel::launch_unchecked::<MulOp, R>,
        IndexingUpdateOp::Min => scatter_nd_kernel::launch_unchecked::<BinaryMinOp, R>,
        IndexingUpdateOp::Max => scatter_nd_kernel::launch_unchecked::<BinaryMaxOp, R>,
    };

    let data_shape = &tensor.meta.shape;
    let idx_shape = &indices.meta.shape;
    let m = idx_shape.num_dims();
    let k = idx_shape[m - 1];

    // num_updates = product of first M-1 dims of indices
    let num_updates: usize = idx_shape.as_slice()[..m - 1].iter().product();
    // slice_size = product of data.shape[K..]
    let slice_size: usize = data_shape.as_slice()[k..].iter().product();
    let working_units = num_updates * slice_size;

    let ruda_dim = RudaDim::new(indices.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&indices.client, working_units, ruda_dim);

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    let data_slice_shape = shape_divmod_range(&tensor, k..data_shape.num_dims());
    let values_shape = shape_divmod(&values);

    unsafe {
        launch(
            &tensor.client.clone(),
            ruda_count,
            ruda_dim,
            address_type!(tensor, indices, values),
            tensor.clone().into_tensor_arg(),
            indices.into_linear_view(),
            values.into_tensor_arg(),
            data_slice_shape,
            values_shape,
            slice_size,
            k,
            working_units,
            [tensor_dtype.into(), indices_dtype.into()],
        )
    }

    tensor
}
