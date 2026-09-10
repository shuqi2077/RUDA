use ruda_kernel::dsl as kernel_dsl;
use std::marker::PhantomData;

use ruda_kernel::dsl::RudaType;
use ruda_kernel::dsl::prelude::*;

use crate::kernel_ir::{components::stage::FilledStage, definition::MatrixTypes};

#[derive(RudaType)]
/// Accumulator reader that zeros the accumulator
pub struct ZeroGlobalReader<IP: MatrixTypes> {
    #[ruda(comptime)]
    _ty: PhantomData<IP>,
}

#[ruda]
impl<IP: MatrixTypes> ZeroGlobalReader<IP> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        ZeroGlobalReader::<IP> { _ty: PhantomData }
    }

    /// Give a reader to the loaded data.
    pub fn stage(&self) -> FilledStage<IP::Stage> {
        FilledStage::new(IP::Stage::from_int(0))
    }
}
