use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use crate::elementwise::binary::numeric::AddOp;
use crate::elementwise::binary::numeric::AssignOp;
use crate::elementwise::binary::numeric::BinaryOp;
use crate::elementwise::binary::numeric::BinaryOpFamily;
use crate::elementwise::binary::numeric::OrOp;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::shape_divmod;
use ruda_kernel::tensor::RudaTensor;
use ruda_kernel::dsl::RudaDim;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;

#[ruda(launch_unchecked, address_type = "dynamic")]
fn scatter_kernel<T: Numeric, I: Int, Op: BinaryOpFamily>(
    input: &mut Tensor<T>,
    indices: &Tensor<I>,
    value: &Tensor<T>,
    in_shape: Sequence<FastDivmod<usize>>,
    #[comptime] dim: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    let rank = in_shape.len().comptime();
    let stride_input = input.stride(dim);
    let stride_value = value.stride(dim);
    let stride_indices = indices.stride(dim);
    let shape_value = value.shape(dim);

    let mut offset = ABSOLUTE_POS;
    let mut offset_input = 0;
    let mut offset_indices = 0;
    let mut offset_value = 0;
    let mut num_elems = 1;

    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        if i != dim {
            let shape_input_loop = input.shape(i);

            let (rem, local_pos) = in_shape[i].div_mod(offset);
            offset = rem;

            offset_input += local_pos * input.stride(i);
            offset_indices += local_pos * indices.stride(i);
            offset_value += local_pos * value.stride(i);

            num_elems *= shape_input_loop;
        }
    }

    let should_stop = ABSOLUTE_POS >= num_elems;
    if should_stop {
        terminate!();
    }

    for i in 0..shape_value {
        let value_idx = (stride_value * i) + offset_value;
        let index_idx = (stride_indices * i) + offset_indices;

        let value = value[value_idx];
        let index = usize::cast_from(indices[index_idx]);

        let input_idx = (stride_input * index) + offset_input;

        let value = Op::BinaryOp::<T, Const<1>>::execute(
            Vector::cast_from(input[input_idx]),
            Vector::cast_from(value),
        );
        input[input_idx] = value[0];
    }
}

pub fn scatter<R: Runtime>(
    dim: usize,
    tensor: RudaTensor<R>,
    indices: RudaTensor<R>,
    value: RudaTensor<R>,
    is_bool: bool,
) -> RudaTensor<R> {
    scatter_impl(dim, tensor, indices, value, is_bool, false)
}

pub fn scatter_assign<R: Runtime>(
    dim: usize,
    tensor: RudaTensor<R>,
    indices: RudaTensor<R>,
    value: RudaTensor<R>,
) -> RudaTensor<R> {
    scatter_impl(dim, tensor, indices, value, false, true)
}

fn scatter_impl<R: Runtime>(
    dim: usize,
    tensor: RudaTensor<R>,
    indices: RudaTensor<R>,
    value: RudaTensor<R>,
    is_bool: bool,
    assign: bool,
) -> RudaTensor<R> {
    if value.meta.num_elements() == 0 {
        return tensor;
    }

    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    let num_elems = tensor.meta.num_elements() / tensor.meta.shape()[dim];

    let working_units = num_elems;
    let ruda_dim = RudaDim::new(indices.client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&indices.client, working_units, ruda_dim);

    let launch = match (assign, is_bool) {
        (true, _) => scatter_kernel::launch_unchecked::<AssignOp, R>,
        (false, true) => scatter_kernel::launch_unchecked::<OrOp, R>,
        (false, false) => scatter_kernel::launch_unchecked::<AddOp, R>,
    };

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    unsafe {
        launch(
            &tensor.client.clone(),
            ruda_count,
            ruda_dim,
            address_type!(tensor, indices, value),
            tensor.clone().into_tensor_arg(),
            indices.into_tensor_arg(),
            value.into_tensor_arg(),
            shape_divmod(&tensor),
            dim,
            [tensor_dtype.into(), indices_dtype.into()],
        )
    }
    tensor
}
