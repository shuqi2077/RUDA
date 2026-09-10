use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::layout::Coords2d;

#[derive(RudaType, Debug, Clone, Copy, PartialEq, Eq)]
/// Events that occur during the process of storing tiles to
/// a stage and executing writes
pub enum WriteEvent {
    /// Before any step
    Begin,
    /// After each tile is stored into the stage
    TileStored { tile: Coords2d },
    /// After the last step
    Finish,
}

#[ruda]
/// Function that is called at each [WriteEvent]
pub trait WriteEventListener: RudaType {
    fn on_event(this: &mut Self, event: WriteEvent);
}
