use super::{RudaTensor, element::TensorElement};
use crate::dsl::Runtime;
use alloc::{vec, vec::Vec};

/// Create a vector containing the dimension, strides and shape of tensors.
///
/// # Example
///
/// With two tensors (lhs, rhs)
///
/// | Indexes                  | Value       |
/// |:------------------------:|:-----------:|
/// |           0..1           | D           |
/// |           1..D + 1       | lhs strides |
/// |     (D + 1)..(2 * D + 1) | rhs strides |
/// | (2 * D + 1)..(3 * D + 1) | lhs shape   |
/// | (3 * D + 1)..(4 * D + 1) | rhs shape   |
pub fn build_info<R: Runtime, E: TensorElement>(tensors: &[&RudaTensor<R>]) -> Vec<u32> {
    let ndims = tensors[0].meta.num_dims();
    let mut info: Vec<u32> = vec![0; tensors.len() * 2 * ndims + 1];
    info[0] = ndims as u32;

    let mut current = 1;
    for tensor in tensors.iter() {
        for d in 0..ndims {
            info[current] = tensor.meta.strides()[d] as u32;
            current += 1;
        }
    }
    for tensor in tensors.iter() {
        for d in 0..ndims {
            info[current] = tensor.meta.shape()[d] as u32;
            current += 1;
        }
    }
    info
}
