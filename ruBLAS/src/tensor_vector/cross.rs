use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::contiguous::into_contiguous;
use ruda_kernel::tensor::layout::address_type;
use ruda_kernel::tensor::layout::broadcast_shape;
use ruda_kernel::tensor::allocation::empty_device_dtype;
use ruda_kernel::tensor::permutation::swap_dims;
use ruda_kernel::tensor::RudaTensor;
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::dsl::calculate_ruda_count_elemwise;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch_unchecked, address_type = "dynamic")]
fn cross_kernel<E: Float>(
    lhs: &LinearView<E>,
    rhs: &LinearView<E>,
    output: &mut LinearView<E, ReadWrite>,
    #[define(E)] _dtype: StorageType,
) {
    // Each thread processes one 3-element vector
    let vector_idx = ABSOLUTE_POS;
    let base_pos = vector_idx * 3;

    if !output.is_in_bounds(base_pos) {
        terminate!();
    }

    // Extract vectors
    let a0 = lhs[base_pos];
    let a1 = lhs[base_pos + 1];
    let a2 = lhs[base_pos + 2];
    let b0 = rhs[base_pos];
    let b1 = rhs[base_pos + 1];
    let b2 = rhs[base_pos + 2];

    // Compute cross product: a × b
    let x = a1 * b2 - a2 * b1;
    let y = a2 * b0 - a0 * b2;
    let z = a0 * b1 - a1 * b0;

    // Store result
    output[base_pos] = x;
    output[base_pos + 1] = y;
    output[base_pos + 2] = z;
}

pub fn cross<R: Runtime>(
    lhs: RudaTensor<R>,
    rhs: RudaTensor<R>,
    dim: usize,
) -> RudaTensor<R> {
    let ndims = lhs.meta.num_dims();

    // Validate that the cross dimension has size 3
    if lhs.meta.shape()[dim] != 3 || rhs.meta.shape()[dim] != 3 {
        panic!(
            "Cross product requires dimension {} to have size 3, but got {} and {}",
            dim,
            lhs.meta.shape()[dim],
            rhs.meta.shape()[dim]
        );
    }

    // The kernel reads each 3-vector from contiguous memory, so it expects the
    // cross dimension to be the last (innermost) and physically contiguous.
    // For non-last dims we permute the cross dim to the last position, run the
    // kernel, then permute the result back. swap_dims only updates strides, so
    // make the permuted operands contiguous before launch.
    if dim != ndims - 1 {
        let last = ndims - 1;
        let lhs = into_contiguous(swap_dims(lhs, dim, last));
        let rhs = into_contiguous(swap_dims(rhs, dim, last));
        let result = cross(lhs, rhs, last);
        return swap_dims(result, dim, last);
    }

    let output_shape = broadcast_shape(&[&lhs, &rhs]);

    let output = empty_device_dtype(
        lhs.client.clone(),
        lhs.device.clone(),
        output_shape.clone(),
        lhs.dtype,
    );

    // Number of vectors to process
    let num_vectors = output_shape.num_elements() / 3;

    let ruda_dim = RudaDim::new(lhs.client.properties(), num_vectors);
    let ruda_count = calculate_ruda_count_elemwise(&lhs.client, num_vectors, ruda_dim);
    let dtype = lhs.dtype;

    unsafe {
        cross_kernel::launch_unchecked(
            &output.client,
            ruda_count,
            ruda_dim,
            address_type!(lhs, rhs, output),
            lhs.into_linear_view_like(&output),
            rhs.into_linear_view_like(&output),
            output.clone().into_linear_view(),
            dtype.into(),
        );
    };

    output
}
