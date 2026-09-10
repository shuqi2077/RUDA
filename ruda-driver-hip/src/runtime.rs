use crate::{
    HipWmmaCompiler,
    execution::{HipServer, context::HipContext},
    device::AmdDevice,
};
use core::ffi::c_int;
use ruda_core::{
    device::{Device, DeviceService},
    profile::TimingMethod,
};
use ruda_kernel::dsl::{
    MemoryConfiguration, Runtime,
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        ContiguousElements, DeviceProperties, HardwareProperties, MatrixLayout,
        MemoryDeviceProperties, MmaProperties, TargetProperties, VectorSize, features::Plane,
    },
    server::ServerUtilities,
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use ruda_compiler::cpp::{
    ComputeKernel,
    hip::{HipDialect, arch::AMDArchitecture, mma::contiguous_elements_rdna3},
    register_supported_types,
    shared::{
        Architecture, CompilationOptions, CppSupportedFeatures, DialectWmmaCompiler,
        register_mma_features, register_scaled_mma_features, register_wmma_features,
    },
};
use ruda_hip_sys::{HIP_SUCCESS, hipDeviceScheduleSpin, hipGetDeviceCount, hipSetDeviceFlags};
use ruda::runtime::{
    allocator::PitchedMemoryLayoutPolicy, client::ComputeClient, logging::ServerLogger,
};
use std::{ffi::CStr, mem::MaybeUninit, sync::Arc};

/// The values that control how a HIP Runtime will perform its calculations.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug, Clone)]
pub struct HipRuntime;

pub type HipCompiler = ruda_kernel::dsl::lowering::cpp::HipCompiler<HipWmmaCompiler>;
pub type HipComputeKernel = ComputeKernel<HipDialect<HipWmmaCompiler>>;

mod device_service;

impl Runtime for HipRuntime {
    type Compiler = HipCompiler;
    type Server = HipServer;
    type Device = AmdDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        "hip"
    }

    fn autotune_driver_fingerprint(client: &ComputeClient<Self>) -> Option<String> {
        use std::{collections::BTreeMap, sync::{Mutex, OnceLock}};
        static IDS: OnceLock<Mutex<BTreeMap<u16, Option<String>>>> = OnceLock::new();
        let index = client.device_id().index_id;
        let mut ids = IDS.get_or_init(|| Mutex::new(BTreeMap::new())).lock().unwrap_or_else(|p| p.into_inner());
        ids.entry(index).or_insert_with(|| {
            let mut version = 0i32; let mut name = [0i8; 256];
            // SAFETY: both HIP queries write bounded stack outputs and retain no pointers.
            let (vs, ns) = unsafe {
                (ruda_hip_sys::hipRuntimeGetVersion(&mut version),
                 ruda_hip_sys::hipDeviceGetName(name.as_mut_ptr(), name.len() as i32, index as i32))
            };
            if vs != HIP_SUCCESS || ns != HIP_SUCCESS || version <= 0 { return None; }
            let module = std::fs::read_to_string("/sys/module/amdgpu/srcversion").ok()?;
            let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease").ok()?;
            let name: Vec<u8> = name.iter().take_while(|&&v| v != 0).map(|&v| v as u8).collect();
            Some(format!("device={};hip-runtime={version};amdgpu={module};kernel={kernel}", String::from_utf8_lossy(&name)))
        }).clone()
    }

    fn require_array_lengths() -> bool {
        true
    }

    fn max_ruda_count() -> (u32, u32, u32) {
        (i32::MAX as u32, u16::MAX as u32, u16::MAX as u32)
    }

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        if shape.is_empty() {
            return true;
        }
        has_pitched_row_major_strides(shape, strides)
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: MmaProperties {
                register_size_bits: 32,
                const_plane_size: 32,
                register_layout_a: MatrixLayout::RowMajor,
                register_layout_b: MatrixLayout::ColMajor,
                register_layout_acc: MatrixLayout::ColMajor,
                register_duplication_a: 2,
                register_duplication_b: 2,
                register_duplication_acc: 1,
                contiguous_elements: ContiguousElements::new(contiguous_elements_rdna3),
            },
        }
    }

    fn enumerate_devices(
        _: u16,
        _: &<Self::Server as ruda_kernel::dsl::server::ComputeServer>::Info,
    ) -> Vec<ruda_kernel::dsl::device::DeviceId> {
        fn device_count() -> usize {
            let mut device_count: c_int = 0;
            let result;
            // SAFETY: Calling HIP FFI to get the number of available devices.
            // `device_count` is a valid mutable pointer to a stack-allocated `c_int`.
            unsafe {
                result = hipGetDeviceCount(&mut device_count);
            }
            if result == HIP_SUCCESS {
                device_count.try_into().unwrap_or(0)
            } else {
                0
            }
        }
        (0..device_count())
            .map(|i| DeviceId::new(0, i as u16))
            .collect()
    }
}

/// Checks whether the GPU with the given device ID is an integrated (APU) device.
///
/// # Safety
///
/// Calls HIP FFI functions. The caller must ensure `device_id` is a valid HIP device index.
unsafe fn is_integrated_gpu(device_id: i32) -> bool {
    // SAFETY: `hipDeviceProp_tR0600` is a plain-old-data struct; zeroing it is valid.
    let mut props = unsafe { std::mem::zeroed::<ruda_hip_sys::hipDeviceProp_tR0600>() };
    // SAFETY: `props` is a valid mutable reference and `device_id` is assumed valid by the caller.
    let status = unsafe { ruda_hip_sys::hipGetDevicePropertiesR0600(&mut props, device_id) };
    if status != HIP_SUCCESS {
        return false; // assume discrete if we can't tell
    }
    props.integrated != 0
}
