use crate::dsl::prelude::*;
use crate::library::{FastDivmod, FastDivmodInt, tensor::layout::linear::{LinearLayoutLaunch, LinearViewLayoutLaunch}};
use ruda_core::tensor::Shape;
use super::RudaTensor;
use crate::dsl::{Runtime, tensor_vector_size_parallel, ir::{AddressType, UIntKind, VectorSize}};

pub fn max_vector_size<R: Runtime>(tensor: &RudaTensor<R>) -> VectorSize {
    tensor_vector_size_parallel(
        tensor.client.io_optimized_vector_sizes(tensor.dtype.size()),
        tensor.meta.shape(),
        tensor.meta.strides(),
        tensor.meta.num_dims() - 1,
    )
}

pub trait RequiredAddrType {
    fn required_address_type(&self) -> AddressType;
}

impl<R: Runtime> RequiredAddrType for RudaTensor<R> {
    fn required_address_type(&self) -> AddressType {
        self.required_address_type()
    }
}
impl<R: Runtime> RequiredAddrType for Option<RudaTensor<R>> {
    fn required_address_type(&self) -> AddressType {
        self.as_ref()
            .map(|it| it.required_address_type())
            .unwrap_or_default()
    }
}

#[macro_export]
macro_rules! tensor_address_type {
    ($($tensor: tt),*) => {
        [$($crate::tensor::layout::RequiredAddrType::required_address_type(&$tensor)),*]
        .into_iter()
        .max()
        .unwrap_or_default()
    };
}
pub use crate::tensor_address_type as address_type;

pub fn broadcast_shape<R: Runtime>(tensors: &[&RudaTensor<R>]) -> Shape {
    let rank = tensors[0].meta.num_dims();
    debug_assert!(
        tensors.iter().all(|it| it.meta.num_dims() == rank),
        "Broadcast tensors must have the same rank"
    );

    let dims = (0..rank).map(|dim| {
        let max = tensors.iter().map(|it| it.meta.shape()[dim]).max();
        let max = max.unwrap_or(1);
        debug_assert!(
            tensors
                .iter()
                .all(|it| it.meta.shape()[dim] == max || it.meta.shape()[dim] == 1),
            "Broadcast dims must be size 1"
        );
        max
    });

    Shape::from(dims)
}

pub fn max_vector_size_many<R: Runtime>(
    tensors: &[&RudaTensor<R>],
    axis: usize,
) -> VectorSize {
    let vec = tensors
        .iter()
        .map(|tensor| {
            tensor_vector_size_parallel(
                tensor.client.io_optimized_vector_sizes(tensor.dtype.size()),
                tensor.meta.shape(),
                tensor.meta.strides(),
                axis,
            )
        })
        .min();

    vec.unwrap_or(0)
}

pub fn shape_divmod<R: Runtime>(tensor: &RudaTensor<R>) -> SequenceArg<R, FastDivmod<usize>> {
    let mut arg = SequenceArg::new();
    for dim in tensor.meta.shape().iter() {
        arg.push(*dim);
    }
    arg
}

pub fn shape_divmod_range<R: Runtime>(
    tensor: &RudaTensor<R>,
    range: core::ops::Range<usize>,
) -> SequenceArg<R, FastDivmod<usize>> {
    let mut arg = SequenceArg::new();
    let shape = &tensor.meta.shape;
    for i in range {
        arg.push(shape[i]);
    }
    arg
}

pub fn linear_layout<R: Runtime>(
    tensor: &RudaTensor<R>,
    vector_size: VectorSize,
) -> LinearLayoutLaunch<R> {
    LinearLayoutLaunch::from_shape_strides(
        tensor.meta.shape().clone(),
        tensor.meta.strides().clone(),
        // Don't care about type size, only vector size
        Type::new(UIntKind::U32.into()).with_vector_size(vector_size),
        LinearViewLayoutLaunch::new(),
    )
}

pub fn split_dim<R: Runtime>(
    mut tensor: RudaTensor<R>,
    dim: usize,
    shape: &[usize],
) -> RudaTensor<R> {
    let mut stride = tensor.meta.strides()[dim];
    tensor.meta.remove(dim);

    for size in shape.iter().rev() {
        tensor.meta.insert(dim, *size, stride);
        stride *= size;
    }

    tensor
}

pub fn broadcast_strides<R: Runtime>(
    reference: &RudaTensor<R>,
    tensor: &RudaTensor<R>,
) -> SequenceArg<R, usize> {
    if reference.meta.shape() != tensor.meta.shape() {
        tensor
            .meta
            .strides()
            .iter()
            .zip(
                tensor
                    .meta
                    .shape()
                    .iter()
                    .zip(reference.meta.shape().iter()),
            )
            .map(|(stride, (shape, ref_shape))| if *shape == *ref_shape { *stride } else { 0 })
            .collect()
    } else {
        tensor.meta.strides().iter().copied().collect()
    }
}

#[ruda]
pub fn decompose_linear<I: FastDivmodInt>(
    pos: I,
    shape: &Sequence<FastDivmod<I>>,
) -> (I, Sequence<I>) {
    let rank = comptime![shape.len()];
    let mut offs = pos;
    let mut out = Sequence::new();

    #[unroll]
    for i in 0..rank {
        let dim = comptime![rank - i - 1];
        let (rem, offs_local) = shape.index(dim).div_mod(offs);
        out.push(offs_local);
        offs = rem;
    }

    (offs, out.rev())
}
