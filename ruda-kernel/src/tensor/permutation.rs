use super::RudaTensor;
use crate::dsl::Runtime;
use ruda_core::tensor::{DType, Shape, TensorMetadata, quantization::{BlockSize, QuantLevel, QuantScheme, QuantStore}};

pub fn swap_dims<R: Runtime>(
    mut tensor: RudaTensor<R>,
    dim1: usize,
    dim2: usize,
) -> RudaTensor<R> {
    tensor.meta.swap(dim1, dim2);

    if let DType::QFloat(scheme) = tensor.dtype {
        tensor.dtype = DType::QFloat(swap_dims_scheme(scheme, tensor.rank(), dim1, dim2));
        if matches!(scheme.level, QuantLevel::Block(_)) {
            tensor.qparams.as_mut().unwrap().scales.metadata.swap(dim1, dim2);
        }
    }

    tensor
}

pub fn swap_dims_scheme(
    mut scheme: QuantScheme,
    rank: usize,
    dim1: usize,
    dim2: usize,
) -> QuantScheme {
    if let QuantLevel::Block(block_size) = scheme.level {
        let mut block_size = block_size.to_dim_vec(rank);
        block_size.swap(dim1, dim2);

        // Truncate unit dims from the start
        let first = block_size.iter().position(|dim| *dim != 1).unwrap_or(0);
        let block_size = &block_size[first..];
        if block_size.len() > BlockSize::MAX_DIMS {
            panic!("Swapped block size would exceed max dims");
        }

        scheme.level = QuantLevel::Block(BlockSize::new(block_size));
    }

    if let QuantStore::PackedU32(packed_dim) | QuantStore::PackedNative(packed_dim) =
        &mut scheme.store
    {
        if *packed_dim == rank - dim1 - 1 {
            *packed_dim = rank - dim2 - 1;
        } else if *packed_dim == rank - dim2 - 1 {
            *packed_dim = rank - dim1 - 1;
        }
    }

    scheme
}

/// Permute a tensor's dimensions
pub fn permute<R: Runtime>(mut tensor: RudaTensor<R>, axes: &[usize]) -> RudaTensor<R> {
    tensor.meta.permute(axes).unwrap();

    if let DType::QFloat(scheme) = tensor.dtype {
        tensor.dtype = DType::QFloat(permute_scheme(scheme, axes));
        if matches!(scheme.level, QuantLevel::Block(_)) {
            tensor.qparams.as_mut().unwrap().scales.metadata.permute(axes).unwrap();
        }
    }

    tensor
}

pub fn permute_scheme(mut scheme: QuantScheme, axes: &[usize]) -> QuantScheme {
    let rank = axes.len();
    if let QuantLevel::Block(block_size) = scheme.level {
        let mut block_size = block_size.to_dim_vec(rank);
        block_size = axes.iter().map(|i| block_size[*i]).collect();

        // Truncate unit dims from the start
        let block_size = block_size
            .into_iter()
            .skip_while(|it| *it == 1)
            .collect::<Vec<_>>();
        if block_size.len() > BlockSize::MAX_DIMS {
            panic!("Swapped block size would exceed max dims");
        }

        scheme.level = QuantLevel::block(&block_size);
    }

    if let QuantStore::PackedU32(packed_dim) | QuantStore::PackedNative(packed_dim) =
        &mut scheme.store
    {
        let new_pos = axes
            .iter()
            .position(|axis| *axis == rank - *packed_dim - 1)
            .unwrap_or(0);
        *packed_dim = rank - new_pos - 1;
    }

    scheme
}

/// Permute a tensor's dimensions from NCHW to NHWC, or the N-dimensional equivalent
pub fn permute_nchw_to_nhwc<R: Runtime>(tensor: RudaTensor<R>) -> RudaTensor<R> {
    let rank = tensor.meta.num_dims();
    let c_dim = 1;

    let mut dims = vec![0];
    dims.extend(2..rank);
    dims.push(c_dim);

    permute(tensor, &dims)
}

/// Permute a shape's dimensions from NCHW to NHWC, or the N-dimensional equivalent
pub fn permute_nchw_to_nhwc_shape(shape: Shape) -> Shape {
    let rank = shape.num_dims();
    let c_dim = 1;

    let mut dims = vec![0];
    dims.extend(2..rank);
    dims.push(c_dim);

    shape.permuted(&dims).expect("Shape permute should succeed")
}

/// Permute a tensor's dimensions from NHWC to NCHW, or the N-dimensional equivalent
pub fn permute_nhwc_to_nchw<R: Runtime>(tensor: RudaTensor<R>) -> RudaTensor<R> {
    let rank = tensor.meta.num_dims();
    let c_dim = rank - 1;

    let mut dims = vec![0];
    dims.push(c_dim);
    dims.extend(1..c_dim);

    permute(tensor, &dims)
}

/// Permute a shape's dimensions from NHWC to NCHW, or the N-dimensional equivalent
pub fn permute_nhwc_to_nchw_shape(shape: Shape) -> Shape {
    let rank = shape.num_dims();
    let c_dim = rank - 1;

    let mut dims = vec![0];
    dims.push(c_dim);
    dims.extend(1..c_dim);

    shape.permuted(&dims).expect("Shape permute should succeed")
}

