use ruda_core::tensor::{DType, host::HostTensor};

pub fn int_gather(
    dim: usize,
    tensor: HostTensor,
    indices: HostTensor,
) -> HostTensor {
    match tensor.dtype() {
        DType::I64 => crate::gather_scatter::gather::<i64>(tensor, dim, indices),
        DType::I32 => crate::gather_scatter::gather::<i32>(tensor, dim, indices),
        DType::I16 => crate::gather_scatter::gather::<i16>(tensor, dim, indices),
        DType::I8 => crate::gather_scatter::gather::<i8>(tensor, dim, indices),
        DType::U64 => crate::gather_scatter::gather::<u64>(tensor, dim, indices),
        DType::U32 => crate::gather_scatter::gather::<u32>(tensor, dim, indices),
        DType::U16 => crate::gather_scatter::gather::<u16>(tensor, dim, indices),
        DType::U8 => crate::gather_scatter::gather::<u8>(tensor, dim, indices),
        dt => panic!("int_gather: unsupported dtype {:?}", dt),
    }
}

pub fn int_scatter_add(
    dim: usize,
    tensor: HostTensor,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    debug_assert_eq!(
        tensor.dtype(),
        value.dtype(),
        "int_scatter_add: dtype mismatch"
    );
    match tensor.dtype() {
        DType::I64 => {
            crate::gather_scatter::scatter_add::<i64>(tensor, dim, indices, value)
        }
        DType::I32 => {
            crate::gather_scatter::scatter_add::<i32>(tensor, dim, indices, value)
        }
        DType::I16 => {
            crate::gather_scatter::scatter_add::<i16>(tensor, dim, indices, value)
        }
        DType::I8 => crate::gather_scatter::scatter_add::<i8>(tensor, dim, indices, value),
        DType::U64 => {
            crate::gather_scatter::scatter_add::<u64>(tensor, dim, indices, value)
        }
        DType::U32 => {
            crate::gather_scatter::scatter_add::<u32>(tensor, dim, indices, value)
        }
        DType::U16 => {
            crate::gather_scatter::scatter_add::<u16>(tensor, dim, indices, value)
        }
        DType::U8 => crate::gather_scatter::scatter_add::<u8>(tensor, dim, indices, value),
        dt => panic!("int_scatter_add: unsupported dtype {:?}", dt),
    }
}

pub fn int_scatter_nd(
    data: HostTensor,
    indices: HostTensor,
    values: HostTensor,
    reduction: ruda_core::tensor::indexing::IndexingUpdateOp,
) -> HostTensor {
    match data.dtype() {
        DType::I64 => {
            crate::gather_scatter::scatter_nd::<i64>(data, indices, values, reduction)
        }
        DType::I32 => {
            crate::gather_scatter::scatter_nd::<i32>(data, indices, values, reduction)
        }
        DType::I16 => {
            crate::gather_scatter::scatter_nd::<i16>(data, indices, values, reduction)
        }
        DType::I8 => {
            crate::gather_scatter::scatter_nd::<i8>(data, indices, values, reduction)
        }
        DType::U64 => {
            crate::gather_scatter::scatter_nd::<u64>(data, indices, values, reduction)
        }
        DType::U32 => {
            crate::gather_scatter::scatter_nd::<u32>(data, indices, values, reduction)
        }
        DType::U16 => {
            crate::gather_scatter::scatter_nd::<u16>(data, indices, values, reduction)
        }
        DType::U8 => {
            crate::gather_scatter::scatter_nd::<u8>(data, indices, values, reduction)
        }
        dt => panic!("int_scatter_nd: unsupported dtype {:?}", dt),
    }
}

pub fn int_gather_nd(data: HostTensor, indices: HostTensor) -> HostTensor {
    match data.dtype() {
        DType::I64 => crate::gather_scatter::gather_nd::<i64>(data, indices),
        DType::I32 => crate::gather_scatter::gather_nd::<i32>(data, indices),
        DType::I16 => crate::gather_scatter::gather_nd::<i16>(data, indices),
        DType::I8 => crate::gather_scatter::gather_nd::<i8>(data, indices),
        DType::U64 => crate::gather_scatter::gather_nd::<u64>(data, indices),
        DType::U32 => crate::gather_scatter::gather_nd::<u32>(data, indices),
        DType::U16 => crate::gather_scatter::gather_nd::<u16>(data, indices),
        DType::U8 => crate::gather_scatter::gather_nd::<u8>(data, indices),
        dt => panic!("int_gather_nd: unsupported dtype {:?}", dt),
    }
}

pub fn int_select(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
) -> HostTensor {
    match tensor.dtype() {
        DType::I64 => crate::gather_scatter::select::<i64>(tensor, dim, indices),
        DType::I32 => crate::gather_scatter::select::<i32>(tensor, dim, indices),
        DType::I16 => crate::gather_scatter::select::<i16>(tensor, dim, indices),
        DType::I8 => crate::gather_scatter::select::<i8>(tensor, dim, indices),
        DType::U64 => crate::gather_scatter::select::<u64>(tensor, dim, indices),
        DType::U32 => crate::gather_scatter::select::<u32>(tensor, dim, indices),
        DType::U16 => crate::gather_scatter::select::<u16>(tensor, dim, indices),
        DType::U8 => crate::gather_scatter::select::<u8>(tensor, dim, indices),
        dt => panic!("int_select: unsupported dtype {:?}", dt),
    }
}

pub fn int_select_add(
    tensor: HostTensor,
    dim: usize,
    indices: HostTensor,
    value: HostTensor,
) -> HostTensor {
    debug_assert_eq!(
        tensor.dtype(),
        value.dtype(),
        "int_select_add: dtype mismatch"
    );
    match tensor.dtype() {
        DType::I64 => {
            crate::gather_scatter::select_add::<i64>(tensor, dim, indices, value)
        }
        DType::I32 => {
            crate::gather_scatter::select_add::<i32>(tensor, dim, indices, value)
        }
        DType::I16 => {
            crate::gather_scatter::select_add::<i16>(tensor, dim, indices, value)
        }
        DType::I8 => crate::gather_scatter::select_add::<i8>(tensor, dim, indices, value),
        DType::U64 => {
            crate::gather_scatter::select_add::<u64>(tensor, dim, indices, value)
        }
        DType::U32 => {
            crate::gather_scatter::select_add::<u32>(tensor, dim, indices, value)
        }
        DType::U16 => {
            crate::gather_scatter::select_add::<u16>(tensor, dim, indices, value)
        }
        DType::U8 => crate::gather_scatter::select_add::<u8>(tensor, dim, indices, value),
        dt => panic!("int_select_add: unsupported dtype {:?}", dt),
    }
}

