//! Flip operation for reversing tensor elements along axes.
//!
//! With signed strides, flip is a zero-copy operation that simply negates
//! the stride and adjusts the start offset for each flipped axis.

use ruda_core::tensor::host::HostTensor;

/// Flip tensor elements along specified axes.
///
/// This is a zero-copy operation using negative strides.
pub fn flip(tensor: HostTensor, axes: &[usize]) -> HostTensor {
    if axes.is_empty() {
        return tensor;
    }

    let new_layout = tensor.layout().flip(axes);
    tensor.with_layout(new_layout)
}

// Tests kept here exercise flex-specific behavior: flip is a zero-copy
// stride-only operation in the flex backend, and the test below verifies
// the underlying buffer pointer is shared across the flip. Correctness
// tests for flip along various axes live in
// crates/ruda-backend-tests/tests/tensor/{float,int,bool}/ops/flip.rs and
// run against every backend.
#[cfg(test)]
mod tests;
