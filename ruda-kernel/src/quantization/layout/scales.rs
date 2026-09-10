use crate::library::{
    FastDivmod,
    tensor::{
        View,
        layout::{Coords1d, Layout, LayoutExpand},
    },
};
use crate::dsl::{prelude::*};
use crate::library::{tensor::launch::ViewArg};

use crate::quantization::scheme::{QuantLevel, QuantScheme};
use ruda_core::tensor::Shape;

/// Layout for quantization scales, indexed by quant element index and returns the corresponding
/// scale based on the quantization type.
#[derive(RudaType, RudaLaunch)]
pub enum ScalesLayout {
    PerTensor(PerTensorLayout),
    BlockScaled(BlockScaledLayout),
}

#[ruda]
impl Layout for ScalesLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        match self {
            ScalesLayout::PerTensor(layout) => layout.to_source_pos(pos),
            ScalesLayout::BlockScaled(layout) => layout.to_source_pos(pos),
        }
    }

    fn shape(&self) -> Self::Coordinates {
        match self {
            ScalesLayout::PerTensor(layout) => layout.shape(),
            ScalesLayout::BlockScaled(layout) => layout.shape(),
        }
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        match self {
            ScalesLayout::PerTensor(layout) => layout.is_in_bounds(pos),
            ScalesLayout::BlockScaled(layout) => layout.is_in_bounds(pos),
        }
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        match self {
            ScalesLayout::PerTensor(layout) => layout.to_source_pos_checked(pos),
            ScalesLayout::BlockScaled(layout) => layout.to_source_pos_checked(pos),
        }
    }
}

#[ruda]
impl ScalesLayout {
    /// Whether the position is at the start of a new block. Used for electing a unit to write each
    /// scale.
    pub fn is_block_start(&self, pos: usize) -> bool {
        match self {
            ScalesLayout::PerTensor(layout) => layout.is_block_start(pos),
            ScalesLayout::BlockScaled(layout) => layout.is_block_start(pos),
        }
    }
}

#[derive(RudaType, RudaLaunch)]
pub struct PerTensorLayout {
    tensor_len: usize,
}

#[ruda]
impl PerTensorLayout {
    pub fn new(tensor_len: usize) -> Self {
        PerTensorLayout { tensor_len }
    }
}

#[ruda]
impl Layout for PerTensorLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, _pos: Self::Coordinates) -> Self::SourceCoordinates {
        0usize.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        self.tensor_len
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.tensor_len
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

#[ruda]
impl PerTensorLayout {
    /// Whether the position is at the start of a new block. Used for electing a unit to write each
    /// scale.
    pub fn is_block_start(&self, pos: usize) -> bool {
        pos == 0
    }
}

#[derive(RudaType, RudaLaunch)]
pub struct BlockScaledLayout {
    tensor_shape: Sequence<FastDivmod<usize>>,
    tensor_len: usize,
    scales_strides: Sequence<usize>,
    #[ruda(comptime)]
    block_size: Vec<u8>,
    #[ruda(comptime)]
    scales_vector_size: usize,
}

#[ruda]
impl BlockScaledLayout {
    pub fn new(
        tensor_shape: Sequence<FastDivmod<usize>>,
        tensor_len: usize,
        scales_strides: Sequence<usize>,
        #[comptime] block_size: Vec<u8>,
        #[comptime] scales_vector_size: usize,
    ) -> Self {
        BlockScaledLayout {
            tensor_shape,
            tensor_len,
            scales_strides,
            block_size,
            scales_vector_size,
        }
    }
}

#[ruda]
impl Layout for BlockScaledLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let rank = self.scales_strides.len().comptime();
        let mut offs = pos;
        let mut scale_offs = 0;

        #[unroll]
        for i in 0..rank {
            let dim = rank - i - 1;
            let block_size_local = comptime![self.block_size[dim] as usize];
            let (rem, offs_local) = self.tensor_shape[dim].div_mod(offs);

            offs = rem;
            scale_offs += (offs_local / block_size_local) * self.scales_strides[dim];
        }

        scale_offs / self.scales_vector_size
    }

    fn shape(&self) -> Self::Coordinates {
        self.tensor_len
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.tensor_len
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

#[ruda]
impl BlockScaledLayout {
    /// Whether the position is at the start of a new block. Used for electing a unit to write each
    /// scale.
    pub fn is_block_start(&self, pos: usize) -> bool {
        let rank = self.scales_strides.len().comptime();
        let mut offs = pos;
        let mut is_start = true;

        #[unroll]
        for i in 0..rank {
            let dim = rank - i - 1;
            let block_size_local = comptime![self.block_size[dim] as usize];
            let (rem, offs_local) = self.tensor_shape[dim].div_mod(offs);
            offs = rem;
            is_start &= offs_local.is_multiple_of(block_size_local);
        }

        is_start
    }
}

/// TensorView with a linear layout inferred from the shape/strides at launch.
/// Useful for elementwise kernels.
pub type ScalesView<E, IO = ReadOnly> = View<E, Coords1d, IO>;
/// Launch type for LinearTensorView.
pub type ScalesViewLaunch<R> = ViewArg<Coords1d, R>;

/// Create a scales view from the values and scales handle, vector size and quantization scheme.
/// `values` should be *the quantized tensor*, and will be adjusted by `num_quants`.
pub fn scales_view<R: Runtime>(
    values: TensorBinding<R>,
    scales: TensorBinding<R>,
    scales_vector_size: usize,
    quant_scheme: &QuantScheme,
) -> ScalesViewLaunch<R> {
    let shape = unpacked_shape(&values.shape, quant_scheme);
    scales_view_with_shape(&shape, scales, scales_vector_size, quant_scheme)
}

/// Create a scales view indexed by the exact logical, unpadded tensor shape.
pub fn scales_view_with_shape<R: Runtime>(
    logical_shape: &Shape,
    scales: TensorBinding<R>,
    scales_vector_size: usize,
    quant_scheme: &QuantScheme,
) -> ScalesViewLaunch<R> {
    let layout = scales_layout_with_shape(logical_shape, &scales, scales_vector_size, quant_scheme);
    let len = if scales.shape.contains(&0) {
        0
    } else {
        scales.shape.iter().zip(scales.strides.iter()).fold(1usize, |span, (size, stride)| {
            span + (size - 1) * stride
        })
    };
    let buffer = unsafe { ArrayArg::from_raw_parts_binding(scales.handle, len) };
    ScalesViewLaunch::new_array::<ScalesLayout>(buffer, layout)
}

pub fn scales_layout<R: Runtime>(
    values: &TensorBinding<R>,
    scales: &TensorBinding<R>,
    scales_vector_size: usize,
    scheme: &QuantScheme,
) -> ScalesLayoutArgs<R> {
    let shape = unpacked_shape(&values.shape, scheme);
    scales_layout_with_shape(&shape, scales, scales_vector_size, scheme)
}

/// Create a scale layout from the exact logical, unpadded tensor shape.
pub fn scales_layout_with_shape<R: Runtime>(
    logical_shape: &Shape,
    scales: &TensorBinding<R>,
    scales_vector_size: usize,
    scheme: &QuantScheme,
) -> ScalesLayoutArgs<R> {
    let values_len = logical_shape.num_elements();

    match &scheme.level {
        QuantLevel::Tensor => ScalesLayoutArgs::PerTensor(PerTensorLayoutLaunch::new(values_len)),
        QuantLevel::Block(block_size) => {
            let tensor_shape = shape_divmod(logical_shape);
            let scales_strides = strides_seq(&scales.strides);
            ScalesLayoutArgs::BlockScaled(BlockScaledLayoutLaunch::new(
                tensor_shape,
                values_len,
                scales_strides,
                block_size.to_dim_vec(logical_shape.len()),
                scales_vector_size,
            ))
        }
    }
}

fn unpacked_shape(shape: &Shape, scheme: &QuantScheme) -> Shape {
    let mut shape = shape.clone();
    if let Some(dim) = scheme.packing_dim() {
        let axis = shape.len() - dim - 1;
        shape[axis] *= scheme.num_quants();
    }
    shape
}

fn shape_divmod<R: Runtime>(shape: &[usize]) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for s in shape {
        out_seq.push(*s);
    }
    out_seq
}

fn strides_seq<R: Runtime>(strides: &[usize]) -> SequenceArg<R, usize> {
    let mut out_seq = SequenceArg::new();
    for s in strides {
        out_seq.push(*s);
    }
    out_seq
}
