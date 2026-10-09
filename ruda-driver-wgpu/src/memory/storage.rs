use ruda_kernel::dsl::server::IoError;
use ruda::runtime::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use hashbrown::HashMap;
use std::num::NonZeroU64;
use wgpu::BufferUsages;

/// Minimum buffer size in bytes. The WebGPU spec requires buffer sizes > 0, and shaders
/// declare typed arrays (e.g. `array<vec4<f32>>`) that impose a minimum binding size.
/// 32 bytes covers the largest possible binding type (`vec4<f64>`).
const MIN_BUFFER_SIZE: u64 = 32;

/// Buffer storage for wgpu.
pub struct WgpuStorage {
    memory: HashMap<StorageId, wgpu::Buffer>,
    device: wgpu::Device,
    buffer_usages: BufferUsages,
    mem_alignment: usize,
    relocation_queue: Option<wgpu::Queue>,
}

impl core::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// The memory resource that can be allocated for wgpu.
#[derive(new, Debug)]
pub struct WgpuResource {
    #[new(default)]
    pin: Option<ruda::runtime::memory_management::MemoryResourcePin>,
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer offset.
    pub offset: u64,
    /// The size of the resource.
    ///
    /// # Notes
    ///
    /// The result considers the offset.
    pub size: u64,
}

impl WgpuResource {
    pub(crate) fn address_pin(&self) -> Option<ruda::runtime::memory_management::MemoryResourcePin> {
        self.pin.clone()
    }

    /// Return the binding view of the buffer.
    pub fn as_wgpu_bind_resource(&self) -> wgpu::BindingResource<'_> {
        // wgpu enforces 4-byte alignment for buffer binding sizes per the WebGPU spec.
        // - https://github.com/gfx-rs/wgpu/pull/8041
        //
        // This padding is safe because:
        // 1. In checked mode, bounds checks prevent reading beyond the logical size.
        // 2. In unchecked mode, OOB access is already undefined behavior.
        //
        // For zero-sized resources, pass None (use rest of buffer from offset).
        // The allocator guarantees the buffer is at least MIN_BUFFER_SIZE bytes.
        let size = NonZeroU64::new(self.size.next_multiple_of(4));

        let binding = wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size,
        };
        wgpu::BindingResource::Buffer(binding)
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
impl WgpuStorage {
    /// Create a new storage on the given [device](wgpu::Device).
    pub fn new(mem_alignment: usize, device: wgpu::Device, usages: BufferUsages) -> Self {
        Self {
            memory: HashMap::new(),
            device,
            buffer_usages: usages,
            mem_alignment,
            relocation_queue: None,
        }
    }

    pub(crate) fn with_relocation_queue(mut self, queue: wgpu::Queue) -> Self {
        self.relocation_queue = Some(queue);
        self
    }

    fn wait_relocation(&self) -> Result<(), IoError> {
        #[cfg(not(target_family = "wasm"))]
        {
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
                .map(|_| ()).map_err(|error| IoError::Unknown {
                    description: format!("WGPU relocation wait: {error}"),
                    backtrace: ruda_core::backtrace::BackTrace::capture(),
                })
        }
        #[cfg(target_family = "wasm")]
        {
            Err(IoError::UnsupportedIoOperation { backtrace: ruda_core::backtrace::BackTrace::capture() })
        }
    }
}

impl ComputeStorage for WgpuStorage {
    type Resource = WgpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get_pinned(&mut self, handle: &StorageHandle, binding: ruda::runtime::memory_management::ManagedMemoryBinding) -> Self::Resource {
        let mut resource = self.get(handle);
        resource.pin = Some(binding.pin());
        resource
    }

    fn supports_relocation(&self) -> bool {
        self.relocation_queue.is_some() && cfg!(not(target_family = "wasm"))
    }

    fn relocation_barrier(&mut self) -> Result<(), IoError> {
        let queue = self.relocation_queue.as_ref().expect("relocation queue");
        // Submit pending write_buffer work too, even when no compute encoder
        // had commands. The stream submits its encoder before reserve.
        queue.submit([]);
        self.wait_relocation()
    }

    fn relocation_copy(&mut self, source: &StorageHandle, target: &StorageHandle) -> Result<(), IoError> {
        let src = self.memory.get(&source.id).expect("relocation source");
        let dst = self.memory.get(&target.id).expect("relocation target");
        let size = source.size().next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        if source.offset() % wgpu::COPY_BUFFER_ALIGNMENT != 0 || target.offset() % wgpu::COPY_BUFFER_ALIGNMENT != 0
            || source.offset().checked_add(size).is_none_or(|end| end > src.size())
            || target.offset().checked_add(size).is_none_or(|end| end > dst.size()) {
            return Err(IoError::UnsupportedIoOperation { backtrace: ruda_core::backtrace::BackTrace::capture() });
        }
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let allocation = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("RUDA adaptive memory copies") });
        encoder.copy_buffer_to_buffer(src, source.offset(), dst, target.offset(), size);
        self.relocation_queue.as_ref().expect("relocation queue").submit([encoder.finish()]);
        let errors = [internal.pop(), allocation.pop(), validation.pop()];
        #[cfg(not(target_family = "wasm"))]
        {
            let mut first = None;
            for error in errors {
                if let Some(error) = ruda_core::future::block_on(error) {
                    if first.is_none() { first = Some(error); }
                }
            }
            if let Some(error) = first {
                return Err(IoError::Unknown { description: format!("WGPU relocation copy: {error}"),
                    backtrace: ruda_core::backtrace::BackTrace::capture() });
            }
        }
        #[cfg(target_family = "wasm")]
        let _ = errors;
        Ok(())
    }

    fn relocation_complete(&mut self) -> Result<(), IoError> { self.wait_relocation() }

    fn get(&mut self, handle: &StorageHandle) -> Self::Resource {
        let buffer = self.memory.get(&handle.id).unwrap();
        WgpuResource::new(buffer.clone(), handle.offset(), handle.size())
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();

        let alloc_size = size.max(MIN_BUFFER_SIZE);

        let scopes = self.relocation_queue.as_ref().map(|_| [
            self.device.push_error_scope(wgpu::ErrorFilter::Validation),
            self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
            self.device.push_error_scope(wgpu::ErrorFilter::Internal),
        ]);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: alloc_size,
            usage: self.buffer_usages,
            mapped_at_creation: false,
        });

        if let Some(scopes) = scopes {
            let errors = scopes.into_iter().rev().map(|scope| scope.pop()).collect::<Vec<_>>();
            #[cfg(not(target_family = "wasm"))]
            {
                let mut first = None;
                for error in errors {
                    if let Some(error) = ruda_core::future::block_on(error) {
                        if first.is_none() { first = Some(error); }
                    }
                }
                if let Some(error) = first {
                    return Err(IoError::Unknown { description: format!("WGPU adaptive allocation: {error}"),
                        backtrace: ruda_core::backtrace::BackTrace::capture() });
                }
            }
            #[cfg(target_family = "wasm")]
            let _ = errors;
        }

        self.memory.insert(id, buffer);
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        self.memory.remove(&id);
    }

    fn flush(&mut self) {
        // We don't wait for dealloc
    }
}
