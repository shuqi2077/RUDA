use super::wgsl;
use crate::AutoRepresentationRef;
use crate::WgpuServer;
use ruda_kernel::dsl::MemoryConfiguration;
use ruda_kernel::dsl::{
    ExecutionMode, WgpuCompilationOptions, hash::StableHash, server::KernelArguments,
};
use ruda_core::ir::DeviceProperties;
use ruda::runtime::{compiler::CompilationError, id::KernelId};
use std::{borrow::Cow, sync::Arc};
use wgpu::{
    Adapter, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    ComputePipeline, Device, PipelineLayoutDescriptor, Queue, ShaderModule, ShaderModuleDescriptor,
    ShaderStages,
};

#[cfg(feature = "spirv")]
use super::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
use super::metal;
#[cfg(all(feature = "msl", target_os = "macos"))]
use ruda_kernel::dsl::lowering::cpp::metal as cpp_metal;

mod pipeline;

pub async fn request_device(adapter: &Adapter) -> (Device, Queue) {
    if let Some(result) = request_vulkan_device(adapter).await {
        return result;
    }
    if let Some(result) = request_metal_device(adapter).await {
        return result;
    }
    wgsl::request_device(adapter).await
}

#[cfg(feature = "spirv")]
async fn request_vulkan_device(adapter: &Adapter) -> Option<(Device, Queue)> {
    if is_vulkan(adapter) {
        vulkan::request_vulkan_device(adapter).await
    } else {
        None
    }
}

#[cfg(not(feature = "spirv"))]
async fn request_vulkan_device(_adapter: &Adapter) -> Option<(Device, Queue)> {
    None
}

#[cfg(all(feature = "msl", target_os = "macos"))]
async fn request_metal_device(adapter: &Adapter) -> Option<(Device, Queue)> {
    if is_metal(adapter) {
        Some(metal::request_metal_device(adapter).await)
    } else {
        None
    }
}

#[cfg(not(all(feature = "msl", target_os = "macos")))]
async fn request_metal_device(_adapter: &Adapter) -> Option<(Device, Queue)> {
    None
}

pub fn register_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
    memory_config: &MemoryConfiguration,
) {
    if register_vulkan_features(adapter, props, comp_options, memory_config) {
        return;
    }
    if register_metal_features(adapter, props, comp_options, memory_config) {
        return;
    }
    wgsl::register_wgsl_features(adapter, props, comp_options);
}

#[cfg(feature = "spirv")]
pub fn register_vulkan_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
    memory_config: &MemoryConfiguration,
) -> bool {
    if is_vulkan(adapter) {
        vulkan::register_vulkan_features(adapter, props, comp_options, memory_config)
    } else {
        false
    }
}

#[cfg(not(feature = "spirv"))]
pub fn register_vulkan_features(
    _adapter: &Adapter,
    _props: &mut DeviceProperties,
    _comp_options: &mut WgpuCompilationOptions,
    _memory_config: &MemoryConfiguration,
) -> bool {
    false
}

#[cfg(all(feature = "msl", target_os = "macos"))]
pub fn register_metal_features(
    adapter: &Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
    _memory_config: &MemoryConfiguration,
) -> bool {
    if is_metal(adapter) {
        metal::register_metal_features(adapter, props, comp_options);
        true
    } else {
        false
    }
}

#[cfg(not(all(feature = "msl", target_os = "macos")))]
pub fn register_metal_features(
    _adapter: &Adapter,
    _props: &mut DeviceProperties,
    _comp_options: &mut WgpuCompilationOptions,
    _memory_config: &MemoryConfiguration,
) -> bool {
    false
}

#[cfg(feature = "spirv")]
fn is_vulkan(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>().is_some() }
}

#[cfg(all(feature = "msl", target_os = "macos"))]
fn is_metal(adapter: &Adapter) -> bool {
    unsafe { adapter.as_hal::<wgpu::hal::api::Metal>().is_some() }
}
