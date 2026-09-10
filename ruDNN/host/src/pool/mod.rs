//! Pooling operations using a unified 3D implementation.
//!
//! All pooling (1D, 2D, 3D) uses a unified 3D implementation:
//! - pool1d: adds two size-1 dimensions, calls pool3d, squeezes output
//! - pool2d: adds one size-1 dimension, calls pool3d, squeezes output
//! - pool3d: native implementation
//!
//! Supported dtypes: f32, f64, f16 (native), bf16 (via f32 conversion)

use alloc::vec;
use alloc::vec::Vec;
use ruda_core::tensor::{DType, element::Element};
use ruda_core::{bytes::Bytes, tensor::Shape};
use half::{bf16, f16};

use ruda_core::tensor::host::{HostTensor, Layout};

// ============================================================================
// Macros for dtype wrappers
// ============================================================================

/// Generates max_pool3d_with_indices typed dispatchers.
macro_rules! max_pool3d_with_indices_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $neg_inf:expr) => {
        pub fn $fn_name(
            x: HostTensor,
            kernel_size: [usize; 3],
            stride: [usize; 3],
            padding: [usize; 3],
            dilation: [usize; 3],
            ceil_mode: bool,
        ) -> (HostTensor, HostTensor) {
            max_pool3d_with_indices_impl::<$T>(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
                $dtype,
                $neg_inf,
            )
        }
    };
}

/// Generates avg_pool3d typed dispatchers.
macro_rules! avg_pool3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr, $div_fn:expr) => {
        pub fn $fn_name(
            x: HostTensor,
            kernel_size: [usize; 3],
            stride: [usize; 3],
            padding: [usize; 3],
            count_include_pad: bool,
            ceil_mode: bool,
        ) -> HostTensor {
            avg_pool3d_impl::<$T>(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
                $dtype,
                $zero,
                $add_fn,
                $div_fn,
            )
        }
    };
}

/// Generates adaptive_avg_pool3d typed dispatchers.
macro_rules! adaptive_avg_pool3d_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr, $div_fn:expr) => {
        pub fn $fn_name(x: HostTensor, output_size: [usize; 3]) -> HostTensor {
            adaptive_avg_pool3d_impl::<$T>(x, output_size, $dtype, $zero, $add_fn, $div_fn)
        }
    };
}

/// Generates max_pool3d_backward typed dispatchers.
macro_rules! max_pool3d_backward_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr) => {
        pub fn $fn_name(x: HostTensor, grad: HostTensor, indices: HostTensor) -> HostTensor {
            max_pool3d_backward_impl::<$T>(x, grad, indices, $dtype, $zero, $add_fn)
        }
    };
}

/// Generates avg_pool3d_backward typed dispatchers.
macro_rules! avg_pool3d_backward_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr, $div_fn:expr) => {
        pub fn $fn_name(
            x: HostTensor,
            grad: HostTensor,
            kernel_size: [usize; 3],
            stride: [usize; 3],
            padding: [usize; 3],
            count_include_pad: bool,
        ) -> HostTensor {
            avg_pool3d_backward_impl::<$T>(
                x,
                grad,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                $dtype,
                $zero,
                $add_fn,
                $div_fn,
            )
        }
    };
}

/// Generates adaptive_avg_pool3d_backward typed dispatchers.
macro_rules! adaptive_avg_pool3d_backward_typed {
    ($fn_name:ident, $T:ty, $dtype:expr, $zero:expr, $add_fn:expr, $div_fn:expr) => {
        pub fn $fn_name(x: HostTensor, grad: HostTensor) -> HostTensor {
            adaptive_avg_pool3d_backward_impl::<$T>(x, grad, $dtype, $zero, $add_fn, $div_fn)
        }
    };
}

// ============================================================================
// Output size calculation
// ============================================================================

/// Calculate pooling output size for a single dimension.
fn pool_output_size(
    input: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    ceil_mode: bool,
) -> usize {
    assert!(kernel > 0, "pool: kernel size must be > 0");
    assert!(stride > 0, "pool: stride must be > 0");
    let effective_kernel = dilation * (kernel - 1) + 1;
    let padded = input + 2 * padding;
    if padded < effective_kernel {
        return if ceil_mode { 1 } else { 0 };
    }
    let numerator = padded - effective_kernel;
    if ceil_mode {
        numerator.div_ceil(stride) + 1
    } else {
        numerator / stride + 1
    }
}

mod max;
pub use max::*;

mod average;
pub use average::*;

mod adaptive;
pub use adaptive::*;

mod backward;
pub use backward::*;

// ============================================================================
// Dimension expansion/squeeze helpers
// ============================================================================

/// Expand 2D tensor [N, C, H, W] to 3D [N, C, 1, H, W].
fn expand_2d_to_3d(x: &HostTensor) -> HostTensor {
    let shape = x.layout().shape();
    x.reshape(Shape::from(vec![shape[0], shape[1], 1, shape[2], shape[3]]))
}

/// Squeeze 3D tensor [N, C, 1, H, W] to 2D [N, C, H, W].
fn squeeze_3d_to_2d(x: HostTensor) -> HostTensor {
    let shape = x.layout().shape();
    x.reshape(Shape::from(vec![shape[0], shape[1], shape[3], shape[4]]))
}

/// Expand 1D tensor [N, C, L] to 3D [N, C, 1, 1, L].
fn expand_1d_to_3d(x: &HostTensor) -> HostTensor {
    let shape = x.layout().shape();
    x.reshape(Shape::from(vec![shape[0], shape[1], 1, 1, shape[2]]))
}

/// Squeeze 3D tensor [N, C, 1, 1, L] to 1D [N, C, L].
fn squeeze_3d_to_1d(x: HostTensor) -> HostTensor {
    let shape = x.layout().shape();
    x.reshape(Shape::from(vec![shape[0], shape[1], shape[4]]))
}

// ============================================================================
// bf16 conversion helpers
// ============================================================================

fn convert_bf16_to_f32(tensor: &HostTensor) -> HostTensor {
    let tensor = tensor.to_contiguous();
    let data: &[bf16] = tensor.storage();
    let f32_data: Vec<f32> = data.iter().map(|x| x.to_f32()).collect();
    HostTensor::new(
        Bytes::from_elems(f32_data),
        Layout::contiguous(tensor.layout().shape().clone()),
        DType::F32,
    )
}

fn convert_f32_to_bf16(tensor: &HostTensor) -> HostTensor {
    let data: &[f32] = tensor.storage();
    let bf16_data: Vec<bf16> = data.iter().map(|x| bf16::from_f32(*x)).collect();
    HostTensor::new(
        Bytes::from_elems(bf16_data),
        Layout::contiguous(tensor.layout().shape().clone()),
        DType::BF16,
    )
}

// ============================================================================
// Tests
// ============================================================================

// Tests kept here probe flex internals: the `pool_output_size` helper
// (including zero-kernel/stride panics), dtype storage paths for max_pool2d
// (f16/bf16/f64), the backward kernels (max/avg/adaptive), and flex's
// count_include_pad avg_pool2d semantics. Plain forward-pass pool tests
// (max/avg/adaptive 2d, pool1d/3d delegation) live in
// crates/ruda-backend-tests/tests/tensor/float/module/{maxpool,avgpool,
// adaptive_avgpool}*.rs and run on every backend.
#[cfg(test)]
mod tests;

pub mod dispatch;
