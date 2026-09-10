use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;
use ruda_kernel::library::tensor::layout::Coords2d;

#[derive(Clone, RudaType)]
/// An iterator over global memory, advancing along k.
pub struct GlobalIterator<EI: RudaPrimitive> {
    global_view: View<EI, Coords2d>,
    offset: RuntimeCell<u32>,
    /// The amount to advance by on each iteration
    step: u32,
    first_step: RuntimeCell<u32>,
    view_size: Coords2d,
    #[ruda(comptime)]
    view_direction: ViewDirection,
    #[ruda(comptime)]
    checked: bool,
}

unsafe impl<EG: RudaPrimitive> Sync for GlobalIterator<EG> {}
unsafe impl<EG: RudaPrimitive> Send for GlobalIterator<EG> {}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Default)]
pub enum ViewDirection {
    Row,
    Col,
    /// Cannot advance if direction is none
    #[default]
    None,
}

#[ruda]
impl<EG: RudaPrimitive> GlobalIterator<EG> {
    /// Instantiate a read iterator over the given global view, which should be sliced to the size
    /// of one `m`/`n` stage and the full range of `k` handled by this matmul instance.
    ///
    /// `step` is the amount advanced in `view_direction` each iteration.
    /// `checked` determines whether the slices should be created as checked or unchecked.
    pub fn new(
        global_view: View<EG, Coords2d>,
        step: u32,
        #[comptime] view_direction: ViewDirection,
        #[comptime] checked: bool,
    ) -> Self {
        let (size_row, size_col) = global_view.shape();
        let view_size = match view_direction {
            ViewDirection::Row => (step, size_col),
            ViewDirection::Col => (size_row, step),
            ViewDirection::None => (size_row, size_col),
        };

        GlobalIterator::<EG> {
            global_view,
            offset: RuntimeCell::new(0),
            step,
            first_step: RuntimeCell::new(step),
            view_size,
            view_direction,
            checked,
        }
    }

    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn advance(&self) {
        let offset = self.offset.read();
        let step = if offset == 0 { self.first_step.read() } else { self.step };
        self.offset.store(offset + step);
    }

    pub fn set_first_step(&mut self, first: u32) {
        self.first_step.store(first);
    }

    /// Returns the current view slice of the iterator
    pub fn view(&self) -> View<EG, Coords2d> {
        let offset = match self.view_direction.comptime() {
            ViewDirection::Row => (self.offset.read(), 0u32),
            ViewDirection::Col => (0u32, self.offset.read()),
            ViewDirection::None => (0u32, 0u32).runtime(),
        };
        let step = if self.offset.read() == 0 { self.first_step.read() } else { self.step };
        let size = match self.view_direction.comptime() {
            ViewDirection::Row => (step, self.view_size.1),
            ViewDirection::Col => (self.view_size.0, step),
            ViewDirection::None => self.view_size,
        };
        if self.checked.comptime() {
            self.global_view.slice(offset, size)
        } else {
            self.global_view.slice_unchecked(offset, size)
        }
    }

    /// Returns the vector size of the global view
    pub fn vector_size(&self) -> comptime_type!(VectorSize) {
        self.global_view.vector_size()
    }

    pub fn offset(&self) -> u32 {
        self.offset.read()
    }
}
