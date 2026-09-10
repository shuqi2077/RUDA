use ruda_core::ir::features::AtomicUsage;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::library::{FastDivmod, tensor::layout::linear::LinearView};
use ruda_kernel::tensor::{RudaTensor, layout::{address_type, shape_divmod_range}};

pub(super) fn scatter_nd_add<R: Runtime>(
    tensor: RudaTensor<R>,
    indices: RudaTensor<R>,
    values: RudaTensor<R>,
) -> RudaTensor<R> {
    let idx_shape = indices.meta.shape();
    let m = idx_shape.len();
    let k = idx_shape[m - 1];
    let num_updates: usize = idx_shape[..m - 1].iter().product();
    let slice_size: usize = tensor.meta.shape()[k..].iter().product();
    let total = num_updates * slice_size;
    let supports_atomic_add = tensor.client.properties()
        .atomic_type_usage(Type::new(StorageType::Atomic(tensor.dtype.into())))
        .contains(AtomicUsage::Add);
    let working_units = if supports_atomic_add { total } else { slice_size };
    let ruda_dim = RudaDim::new(tensor.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&tensor.client, working_units, ruda_dim);
    let launch = if supports_atomic_add {
        atomic_add_kernel::launch_unchecked::<R>
    } else {
        ordered_add_kernel::launch_unchecked::<R>
    };
    let dtypes = [tensor.dtype.into(), indices.dtype.into()];
    let data_slice_shape = shape_divmod_range(&tensor, k..tensor.meta.num_dims());
    let address_type = address_type!(tensor, indices, values).max(AddressType::from_len(
        total.max(indices.meta.num_elements()),
    ));

    unsafe {
        launch(
            &tensor.client,
            ruda_count,
            ruda_dim,
            address_type,
            tensor.clone().into_tensor_arg(),
            indices.into_linear_view(),
            values.into_linear_view(),
            data_slice_shape,
            slice_size,
            k,
            num_updates,
            dtypes,
        );
    }
    tensor
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn atomic_add_kernel<T: Numeric, I: Int>(
    data: &mut Tensor<Atomic<T>>,
    indices: &LinearView<I>,
    values: &LinearView<T>,
    data_slice_shape: Sequence<FastDivmod<usize>>,
    slice_size: usize,
    k: usize,
    num_updates: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= num_updates * slice_size {
        terminate!();
    }
    let update_idx = ABSOLUTE_POS / slice_size;
    let mut remainder = ABSOLUTE_POS % slice_size;
    let slice_rank = data_slice_shape.len().comptime();
    let mut data_idx = 0usize;
    #[unroll]
    for i in 0..slice_rank {
        let dim = slice_rank - i - 1;
        let (rem, coord) = data_slice_shape[dim].div_mod(remainder);
        remainder = rem;
        data_idx += coord * data.stride(k + dim);
    }
    for j in 0..k {
        data_idx += usize::cast_from(indices[update_idx * k + j]) * data.stride(j);
    }
    data[data_idx].fetch_add(values[ABSOLUTE_POS]);
}

#[ruda(launch_unchecked, address_type = "dynamic")]
fn ordered_add_kernel<T: Numeric, I: Int>(
    data: &mut Tensor<T>,
    indices: &LinearView<I>,
    values: &LinearView<T>,
    data_slice_shape: Sequence<FastDivmod<usize>>,
    slice_size: usize,
    k: usize,
    num_updates: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= slice_size {
        terminate!();
    }
    let mut remainder = ABSOLUTE_POS;
    let slice_rank = data_slice_shape.len().comptime();
    let mut slice_offset = 0usize;
    #[unroll]
    for i in 0..slice_rank {
        let dim = slice_rank - i - 1;
        let (rem, coord) = data_slice_shape[dim].div_mod(remainder);
        remainder = rem;
        slice_offset += coord * data.stride(k + dim);
    }
    for update_idx in 0..num_updates {
        let mut data_idx = slice_offset;
        for j in 0..k {
            data_idx += usize::cast_from(indices[update_idx * k + j]) * data.stride(j);
        }
        data[data_idx] = data[data_idx] + values[update_idx * slice_size + ABSOLUTE_POS];
    }
}
