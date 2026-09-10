use alloc::rc::Rc;
use crate::dsl::prelude::*;
use ruda_core::ir::{DeviceProperties, HardwareProperties};
use ruda_kernel_macros::intrinsic;

/// Retrieves the [`device_properties`](DeviceProperties).
#[ruda]
#[allow(unused_variables)]
pub fn device_properties() -> comptime_type!(Rc<DeviceProperties>) {
    intrinsic!(|scope| scope.properties.as_ref().unwrap().clone())
}

/// Retrieves the [`hardware_properties`](HardwareProperties).
#[ruda]
#[allow(unused_variables)]
pub fn hardware_properties() -> comptime_type!(HardwareProperties) {
    let props = &device_properties().comptime().hardware;
    comptime!(props.clone())
}
