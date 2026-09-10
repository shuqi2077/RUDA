use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{Runtime, calculate_ruda_count_elemwise, prelude::*};
use ruda_kernel::library::{FastDivmod, tensor::layout::linear::LinearView};
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, layout::{address_type, shape_divmod}};
use ruda_core::tensor::TensorMetadata;
use super::operation::{CumulativeOp, CumulativeOpFamily, SumOp, ProdOp, MaxOp, MinOp};

/// Generic cumulative operation kernel
///
/// # Limitations
///
/// This is a **naive sequential implementation** along the cumulative dimension:
/// - Each output element sequentially reads all previous elements along the dimension
/// - Computational complexity: O(n^2) memory reads where n is the size of the cumulative dimension
/// - **Performance:** Suitable for small tensors or small dimensions. For large tensors,
///   performance will degrade significantly compared to an optimized parallel scan algorithm.
///
/// # TODO
///
/// Implement an efficient GPU-optimized parallel scan algorithm.
#[ruda(launch_unchecked, address_type = "dynamic")]
fn cumulative_kernel<C: Numeric, O: CumulativeOpFamily>(
    input: &Tensor<C>,
    output: &mut LinearView<C, ReadWrite>,
    shape: Sequence<FastDivmod<usize>>,
    #[comptime] dim: usize,
    #[define(C)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let rank = comptime![shape.len()];
    let dim_stride = input.stride(dim);

    let mut remainder = ABSOLUTE_POS;
    let mut offset = 0;
    let mut dim_idx = 0;

    #[unroll]
    for i in 0..shape.len() {
        let i = comptime![rank - i - 1];
        let (rem, local_idx) = shape.index(i).div_mod(remainder);
        remainder = rem;
        if i == dim {
            dim_idx = local_idx;
        } else {
            offset += local_idx * input.stride(i);
        }
    }

    // Read first element
    let first_read_idx = offset + dim_idx * dim_stride;
    let first_elem = input[first_read_idx];

    // Initialize accumulator
    let mut result = O::CumulativeOp::<C>::init_value(first_elem);

    // Accumulate values
    for i in 0..=dim_idx {
        let read_idx = offset + i * dim_stride;
        result = O::CumulativeOp::<C>::execute(result, input[read_idx]);
    }
    output[ABSOLUTE_POS] = result;
}

/// Compute the cumulative sum along a dimension
pub fn cumsum<R: Runtime>(input: RudaTensor<R>, dim: usize) -> RudaTensor<R> {
    cumulative_op::<R, SumOp>(input, dim)
}

/// Compute the cumulative product along a dimension
pub fn cumprod<R: Runtime>(input: RudaTensor<R>, dim: usize) -> RudaTensor<R> {
    cumulative_op::<R, ProdOp>(input, dim)
}

/// Compute the cumulative minimum along a dimension
pub fn cummin<R: Runtime>(input: RudaTensor<R>, dim: usize) -> RudaTensor<R> {
    cumulative_op::<R, MinOp>(input, dim)
}

/// Compute the cumulative maximum along a dimension
pub fn cummax<R: Runtime>(input: RudaTensor<R>, dim: usize) -> RudaTensor<R> {
    cumulative_op::<R, MaxOp>(input, dim)
}

/// Generic cumulative operation function
fn cumulative_op<R: Runtime, O: CumulativeOpFamily>(
    input: RudaTensor<R>,
    dim: usize,
) -> RudaTensor<R> {
    let client = input.client.clone();
    let device = input.device.clone();

    let output = empty_device_dtype(client.clone(), device, input.shape(), input.dtype);

    let num_elems = output.meta.num_elements();
    let working_units = num_elems;
    let ruda_dim = RudaDim::new(client.properties(), working_units);
    let ruda_count = calculate_ruda_count_elemwise(&client, working_units, ruda_dim);
    let shape = shape_divmod(&input);

    unsafe {
        cumulative_kernel::launch_unchecked::<O, R>(
            &client,
            ruda_count,
            ruda_dim,
            address_type!(input, output),
            input.into_tensor_arg(),
            output.clone().into_linear_view(),
            shape,
            dim,
            output.dtype.into(),
        );
    }

    output
}
