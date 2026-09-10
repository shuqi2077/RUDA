use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;

use crate::convolution::components::ConvolutionOperation;

#[derive(RudaType, RudaLaunch, Clone)]
pub struct RuntimeArgs {
    pub shape_k: u32,
    pub channels: u32,
    pub padded_channels: FastDivmod<u32>,
    #[ruda(comptime)]
    pub operation: ConvolutionOperation,
}
