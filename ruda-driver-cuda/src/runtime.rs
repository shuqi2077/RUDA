use crate::{
    WmmaCompiler,
    execution::{CudaServer, context::CudaContext},
    device::CudaDevice,
};
use ruda_core::{
    device::{Device, DeviceService},
    profile::TimingMethod,
};
use ruda_kernel::dsl::{
    MemoryConfiguration, Runtime,
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        BarrierLevel, ContiguousElements, DeviceProperties, ElemType, FloatKind,
        HardwareProperties, MatrixLayout, MemoryDeviceProperties, MmaProperties, OpaqueType,
        SemanticType, StorageType, TargetProperties, Type, VectorSize,
        features::{AtomicUsage, Plane, Tma, TypeUsage},
    },
    server::ServerUtilities,
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use ruda_compiler::cpp::{
    ComputeKernel, DialectWmmaCompiler,
    cuda::{CudaDialect, arch::CudaArchitecture, mma::contiguous_elements_cuda},
    register_supported_types,
    shared::{
        CompilationOptions, CppSupportedFeatures, register_mma_features,
        register_scaled_mma_features, register_wmma_features,
    },
};
use ruda::runtime::{
    allocator::PitchedMemoryLayoutPolicy, client::ComputeClient, logging::ServerLogger,
};
use cudarc::driver::sys::{CUDA_VERSION, cuDeviceTotalMem_v2};
use std::{mem::MaybeUninit, sync::Arc};

/// Options configuring the CUDA runtime.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug, Clone)]
pub struct CudaRuntime;

mod device_service;

pub type CudaCompiler = ruda_kernel::dsl::lowering::cpp::CudaCompiler<WmmaCompiler>;
pub type CudaComputeKernel = ComputeKernel<CudaDialect<WmmaCompiler>>;

fn tensor_cores_per_sm(version: u32) -> Option<u32> {
    match version {
        70 | 75 => Some(8),                           // Volta, Turing
        80 | 86 | 89 | 90 | 91 | 92 | 100 => Some(4), // Ampere, Hopper, Blackwell
        _ => None,                                    // Unknown or unsupported architecture
    }
}

impl Runtime for CudaRuntime {
    type Compiler = CudaCompiler;
    type Server = CudaServer;
    type Device = CudaDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        "cuda"
    }

    fn require_array_lengths() -> bool {
        true
    }

    fn max_ruda_count() -> (u32, u32, u32) {
        (i32::MAX as u32, u16::MAX as u32, u16::MAX as u32)
    }

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        shape.contains(&0) || has_pitched_row_major_strides(shape, strides)
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: MmaProperties {
                register_size_bits: 32,
                const_plane_size: 32,
                register_layout_a: MatrixLayout::RowMajor,
                register_layout_b: MatrixLayout::ColMajor,
                register_layout_acc: MatrixLayout::RowMajor,
                register_duplication_a: 1,
                register_duplication_b: 1,
                register_duplication_acc: 1,
                contiguous_elements: ContiguousElements::new(contiguous_elements_cuda),
            },
        }
    }

    fn enumerate_devices(
        _: u16,
        _: &<Self::Server as ruda_kernel::dsl::server::ComputeServer>::Info,
    ) -> Vec<ruda_kernel::dsl::device::DeviceId> {
        let count = cudarc::driver::CudaContext::device_count().unwrap_or(0) as usize;
        (0..count)
            .map(|i| DeviceId {
                type_id: 0,
                index_id: i as u16,
            })
            .collect()
    }
}
