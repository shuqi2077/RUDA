#![cfg_attr(docsrs, feature(doc_cfg))]
extern crate alloc;

use ruda_tensor_device::DeviceBackend;

pub use ruda_driver_hip::AmdDevice as RocmDevice;

use ruda_driver_hip::HipRuntime;

#[cfg(not(feature = "fusion"))]
pub type Rocm<F = f32, I = i32, B = u8> = DeviceBackend<HipRuntime, F, I, B>;

#[cfg(feature = "fusion")]
pub type Rocm<F = f32, I = i32, B = u8> = ruda_fusion::Fusion<DeviceBackend<HipRuntime, F, I, B>>;
