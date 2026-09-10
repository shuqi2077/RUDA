use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Events that occur during the process of loading tiles to
/// registers and executing inner Tile Matmuls
pub enum StageEvent {
    /// Before any step
    Begin,
    /// After loading LHS
    LhsLoaded { current: u32, total: u32 },
    /// After X RHS loads are completed
    RhsLoaded { current: u32, total: u32 },
    /// After X tile matmul operations are completed
    TileMatmulCompleted { current: u32, total: u32 },
    /// After the last step
    Finish,
}

#[ruda]
/// Function that is called at each [StageEvent]
pub trait StageEventListener: RudaType {
    fn on_event(this: &mut Self, #[comptime] event: StageEvent);
}

#[derive(RudaType)]
/// Use when there is no event listening to do
pub struct NoEvent {}

#[ruda]
impl StageEventListener for NoEvent {
    fn on_event(_this: &mut Self, #[comptime] _event: StageEvent) {
        // Nothing to do
    }
}

impl Default for NoEvent {
    fn default() -> Self {
        Self::new()
    }
}

#[ruda]
impl NoEvent {
    pub fn new() -> NoEvent {
        NoEvent {}
    }
}
