use super::*;

impl DeviceService for CpuServer {
    fn init(_device_id: ruda_core::device::DeviceId) -> Self {
        let options = RuntimeOptions::default();
        let max_ruda_dim = (u32::MAX, u32::MAX, u32::MAX);
        let max_ruda_count = (u32::MAX, u32::MAX, u32::MAX);
        let system = System::new_all();
        let max_shared_memory_size = system
            .cgroup_limits()
            .map(|g| g.total_memory)
            .unwrap_or(system.total_memory()) as usize;
        let logger = ruda_core::stub::Arc::new(ServerLogger::default());

        let available_parallelism = std::thread::available_parallelism()
            .expect("Can't get available parallelism on this platform")
            .get();

        let topology = HardwareProperties {
            load_width: 512,
            plane_size_min: 1,
            plane_size_max: 1,
            max_bindings: u32::MAX,
            max_shared_memory_size,
            max_ruda_count,
            num_cpu_cores: Some(available_parallelism as u32),
            max_units_per_ruda: u32::MAX,
            max_ruda_dim,
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            max_vector_size: VectorSize::MAX,
        };

        const ALIGNMENT: u64 = 8;

        let mem_properties = MemoryDeviceProperties {
            max_page_size: max_shared_memory_size as u64,
            alignment: ALIGNMENT,
        };

        let mut device_props = DeviceProperties::new(
            Features {
                unaligned_io: true,
                ..Default::default()
            },
            mem_properties.clone(),
            topology,
            TimingMethod::Device,
        );
        register_supported_types(&mut device_props);

        let utilities = ServerUtilities::new(
            device_props,
            logger,
            (),
            ContiguousMemoryLayoutPolicy::new(ALIGNMENT as usize),
        );
        CpuServer::new(mem_properties, options.memory_config, Arc::new(utilities))
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities() as ServerUtilitiesHandle
    }
}
