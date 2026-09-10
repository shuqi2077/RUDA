use ruda_core::tensor::{DType, host::HostTensor};
use half::{bf16, f16};

pub fn float_gather(
    dim: usize,
    tensor: HostTensor,
    indices: HostTensor,
) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => crate::gather_scatter::gather::<f32>(tensor, dim, indices),
        DType::F64 => crate::gather_scatter::gather::<f64>(tensor, dim, indices),
        DType::F16 => crate::gather_scatter::gather::<f16>(tensor, dim, indices),
        DType::BF16 => crate::gather_scatter::gather::<bf16>(tensor, dim, indices),
        _ => panic!("float_gather: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn float_scatter_add(
    dim: usize,
    tensor: HostTensor,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => {
            crate::gather_scatter::scatter_add::<f32>(tensor, dim, indices, value)
        }
        DType::F64 => {
            crate::gather_scatter::scatter_add::<f64>(tensor, dim, indices, value)
        }
        DType::F16 => {
            crate::gather_scatter::scatter_add::<f16>(tensor, dim, indices, value)
        }
        DType::BF16 => {
            crate::gather_scatter::scatter_add::<bf16>(tensor, dim, indices, value)
        }
        _ => panic!("float_scatter_add: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn float_scatter_nd(
    data: HostTensor,
    indices: HostTensor,
    values: HostTensor,
    reduction: ruda_core::tensor::indexing::IndexingUpdateOp,
) -> HostTensor {
    match data.dtype() {
        DType::F32 => {
            crate::gather_scatter::scatter_nd::<f32>(data, indices, values, reduction)
        }
        DType::F64 => {
            crate::gather_scatter::scatter_nd::<f64>(data, indices, values, reduction)
        }
        DType::F16 => {
            crate::gather_scatter::scatter_nd::<f16>(data, indices, values, reduction)
        }
        DType::BF16 => {
            crate::gather_scatter::scatter_nd::<bf16>(data, indices, values, reduction)
        }
        _ => panic!("float_scatter_nd: unsupported dtype {:?}", data.dtype()),
    }
}

pub fn float_gather_nd(data: HostTensor, indices: HostTensor) -> HostTensor {
    match data.dtype() {
        DType::F32 => crate::gather_scatter::gather_nd::<f32>(data, indices),
        DType::F64 => crate::gather_scatter::gather_nd::<f64>(data, indices),
        DType::F16 => crate::gather_scatter::gather_nd::<f16>(data, indices),
        DType::BF16 => crate::gather_scatter::gather_nd::<bf16>(data, indices),
        _ => panic!("float_gather_nd: unsupported dtype {:?}", data.dtype()),
    }
}

pub fn float_select(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => crate::gather_scatter::select::<f32>(tensor, dim, indices),
        DType::F64 => crate::gather_scatter::select::<f64>(tensor, dim, indices),
        DType::F16 => crate::gather_scatter::select::<f16>(tensor, dim, indices),
        DType::BF16 => crate::gather_scatter::select::<bf16>(tensor, dim, indices),
        _ => panic!("float_select: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn float_select_add(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    match tensor.dtype() {
        DType::F32 => {
            crate::gather_scatter::select_add::<f32>(tensor, dim, indices, value)
        }
        DType::F64 => {
            crate::gather_scatter::select_add::<f64>(tensor, dim, indices, value)
        }
        DType::F16 => {
            crate::gather_scatter::select_add::<f16>(tensor, dim, indices, value)
        }
        DType::BF16 => {
            crate::gather_scatter::select_add::<bf16>(tensor, dim, indices, value)
        }
        _ => panic!("float_select_add: unsupported dtype {:?}", tensor.dtype()),
    }
}

