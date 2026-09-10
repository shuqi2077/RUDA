use crate::dsl::prelude::*;
use ruda_kernel::dsl::{zspace::Shape};

use crate::library::tensor::layout::{Coords1d, Layout, LayoutExpand};

/// Layout for contiguous tensors, indexed in individual elements.
/// Differs from `PlainLayout` because `PlainLayout` expects vector indices, not element indices.
#[derive(RudaType, RudaLaunch, Clone)]
pub struct SimpleLayout {
    len: usize,
    #[ruda(comptime)]
    vector_size: VectorSize,
}

#[ruda]
impl SimpleLayout {
    /// Create a new simple layout with a length and vector size.
    ///
    /// # Note
    /// Length should be in elements, not vectors!
    pub fn new(len: usize, #[comptime] vector_size: VectorSize) -> Self {
        SimpleLayout { len, vector_size }
    }
}

impl<R: Runtime> SimpleLayoutLaunch<R> {
    pub fn from_shape(shape: &Shape, vector_size: VectorSize) -> Self {
        let len = shape.iter().product::<usize>();
        Self::new(len, vector_size)
    }

    pub fn from_handle(handle: TensorBinding<R>, vector_size: VectorSize) -> Self {
        Self::from_shape(&handle.shape, vector_size)
    }
}

#[ruda]
impl Layout for SimpleLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> usize {
        pos / self.vector_size
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (usize, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.len
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.len
    }
}
