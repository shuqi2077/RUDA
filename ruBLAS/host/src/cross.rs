use alloc::vec;
use ruda_core::tensor::{Slice, host::HostTensor};

pub fn cross(
    lhs: HostTensor,
    rhs: HostTensor,
    dim: usize,
) -> HostTensor {
    let shape = lhs.layout().shape();
    let ndims = shape.num_dims();
    assert_eq!(
        shape[dim], 3,
        "cross product requires dimension {} to have size 3, got {}",
        dim, shape[dim]
    );

    // Helper to create slices that select index `idx` along `dim`
    let make_slices = |idx: usize| -> alloc::vec::Vec<Slice> {
        (0..ndims)
            .map(|d| {
                if d == dim {
                    Slice::new(idx as isize, Some((idx + 1) as isize), 1)
                } else {
                    Slice::new(0, None, 1)
                }
            })
            .collect()
    };

    // Extract components along the dimension
    // a = [a0, a1, a2], b = [b0, b1, b2]
    let a0 = ruprim_host::slice::slice(lhs.clone(), &make_slices(0));
    let a1 = ruprim_host::slice::slice(lhs.clone(), &make_slices(1));
    let a2 = ruprim_host::slice::slice(lhs, &make_slices(2));

    let b0 = ruprim_host::slice::slice(rhs.clone(), &make_slices(0));
    let b1 = ruprim_host::slice::slice(rhs.clone(), &make_slices(1));
    let b2 = ruprim_host::slice::slice(rhs, &make_slices(2));

    // Cross product: c = a × b
    // c0 = a1*b2 - a2*b1
    // c1 = a2*b0 - a0*b2
    // c2 = a0*b1 - a1*b0
    let c0 = ruprim_host::binary::dispatch_float::float_sub(
        ruprim_host::binary::dispatch_float::float_mul(a1.clone(), b2.clone()),
        ruprim_host::binary::dispatch_float::float_mul(a2.clone(), b1.clone()),
    );
    let c1 = ruprim_host::binary::dispatch_float::float_sub(
        ruprim_host::binary::dispatch_float::float_mul(a2, b0.clone()),
        ruprim_host::binary::dispatch_float::float_mul(a0.clone(), b2),
    );
    let c2 = ruprim_host::binary::dispatch_float::float_sub(ruprim_host::binary::dispatch_float::float_mul(a0, b1), ruprim_host::binary::dispatch_float::float_mul(a1, b0));

    // Concatenate along the dimension
    ruprim_host::cat::cat(vec![c0, c1, c2], dim)
}

