use ruda_core::tensor::{DType, host::HostTensor};
use crate::pool;

pub fn avg_pool2d(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    ceil_mode: bool,
) -> HostTensor {
    match x.dtype() {
        DType::F32 => pool::avg_pool2d_f32(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        ),
        DType::F64 => pool::avg_pool2d_f64(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        ),
        DType::F16 => pool::avg_pool2d_f16(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        ),
        DType::BF16 => pool::avg_pool2d_bf16(
            x,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode,
        ),
        dtype => panic!("avg_pool2d: unsupported dtype {:?}", dtype),
    }
}

pub fn avg_pool2d_backward(
    x: HostTensor,
    grad: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    count_include_pad: bool,
    _divisor_override: bool,
) -> HostTensor {
    match x.dtype() {
        DType::F32 => pool::avg_pool2d_backward_f32(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
        ),
        DType::F64 => pool::avg_pool2d_backward_f64(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
        ),
        DType::F16 => pool::avg_pool2d_backward_f16(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
        ),
        DType::BF16 => pool::avg_pool2d_backward_bf16(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
        ),
        dtype => panic!("avg_pool2d_backward: unsupported dtype {:?}", dtype),
    }
}

pub fn adaptive_avg_pool2d(x: HostTensor, output_size: [usize; 2]) -> HostTensor {
    match x.dtype() {
        DType::F32 => pool::adaptive_avg_pool2d_f32(x, output_size),
        DType::F64 => pool::adaptive_avg_pool2d_f64(x, output_size),
        DType::F16 => pool::adaptive_avg_pool2d_f16(x, output_size),
        DType::BF16 => pool::adaptive_avg_pool2d_bf16(x, output_size),
        dtype => panic!("adaptive_avg_pool2d: unsupported dtype {:?}", dtype),
    }
}

pub fn adaptive_avg_pool2d_backward(
    x: HostTensor,
    grad: HostTensor,
) -> HostTensor {
    match x.dtype() {
        DType::F32 => pool::adaptive_avg_pool2d_backward_f32(x, grad),
        DType::F64 => pool::adaptive_avg_pool2d_backward_f64(x, grad),
        DType::F16 => pool::adaptive_avg_pool2d_backward_f16(x, grad),
        DType::BF16 => pool::adaptive_avg_pool2d_backward_bf16(x, grad),
        dtype => panic!(
            "adaptive_avg_pool2d_backward: unsupported dtype {:?}",
            dtype
        ),
    }
}

pub fn max_pool2d(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> HostTensor {
    match x.dtype() {
        DType::F32 => {
            pool::max_pool2d_f32(x, kernel_size, stride, padding, dilation, ceil_mode)
        }
        DType::F64 => {
            pool::max_pool2d_f64(x, kernel_size, stride, padding, dilation, ceil_mode)
        }
        DType::F16 => {
            pool::max_pool2d_f16(x, kernel_size, stride, padding, dilation, ceil_mode)
        }
        DType::BF16 => {
            pool::max_pool2d_bf16(x, kernel_size, stride, padding, dilation, ceil_mode)
        }
        dtype => panic!("max_pool2d: unsupported dtype {:?}", dtype),
    }
}

pub fn max_pool2d_with_indices(
    x: HostTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: bool,
) -> (HostTensor, HostTensor) {
    let (output, indices) = match x.dtype() {
        DType::F32 => pool::max_pool2d_with_indices_f32(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        ),
        DType::F64 => pool::max_pool2d_with_indices_f64(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        ),
        DType::F16 => pool::max_pool2d_with_indices_f16(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        ),
        DType::BF16 => pool::max_pool2d_with_indices_bf16(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
        ),
        dtype => panic!("max_pool2d_with_indices: unsupported dtype {:?}", dtype),
    };
    (output, indices)
}

pub fn max_pool2d_with_indices_backward(
    x: HostTensor,
    _kernel_size: [usize; 2],
    _stride: [usize; 2],
    _padding: [usize; 2],
    _dilation: [usize; 2],
    _ceil_mode: bool,
    output_grad: HostTensor,
    indices: HostTensor,
) -> HostTensor {
    let x_grad = match x.dtype() {
        DType::F32 => pool::max_pool2d_backward_f32(x, output_grad, indices),
        DType::F64 => pool::max_pool2d_backward_f64(x, output_grad, indices),
        DType::F16 => pool::max_pool2d_backward_f16(x, output_grad, indices),
        DType::BF16 => pool::max_pool2d_backward_bf16(x, output_grad, indices),
        dtype => panic!(
            "max_pool2d_with_indices_backward: unsupported dtype {:?}",
            dtype
        ),
    };
    x_grad
}

