use alloc::format;
use alloc::string::String;
use ruda_tensor::{
    DType, Shape, TensorData,
    backend::{Backend, DeviceId, DeviceOps, ExecutionError},
    try_read_sync,
};
use ruda_tensor::graph::{BackendIr, OperationIr, TensorHandle, TensorId, TensorIr};
use ruda_core::future::DynFut;

use crate::{
    ByteBridge, DirectChannel, MultiBackendBridge, RouterTensor, Runner, RunnerChannel,
    RunnerClient,
};

/// Implement multi backend types, with enums having one variant per backend.
macro_rules! impl_multi_backend_types {
    // Match the default backend and at least one other backend, with rest being optional
    ($module_name:ident, $DefaultBackend:ident, $($OtherBackend:ident),+) => {
        /// Module containing the essential types for multi-backend operations.
        ///
        /// - `Handle`: the type used to point to a tensor (defined for all backends).
        /// - `MultiRunnerClient`: a client for multiple runners (each responsible to execute tensor operations on a given backend).
        /// - `DirectChannel`: a local channel with direct connection to the backend runner clients.
        /// - `ByteBridge`: a simple multi-backend bridge that transfers tensors via the underlying [tensor data](ruda_tensor::TensorData).
        ///
        /// Each enum type is defined with backend identifiers as variant names (e.g., `B1` and `B2` for dual backends).
        pub mod $module_name {
            use super::*;

            /// The type that can be used to point to a tensor of any kind.
            /// Each backend has its own variant.
            pub enum Handle<$DefaultBackend: BackendIr, $($OtherBackend: BackendIr),+> {
                #[allow(missing_docs)]
                $DefaultBackend($DefaultBackend::Handle),
                $(
                    #[allow(missing_docs)]
                    $OtherBackend($OtherBackend::Handle),
                )+
            }

            /// The device type used by a backend.
            /// Each backend has its own variant.
            #[derive(Clone, Debug)]
            pub enum MultiDevice<$DefaultBackend: Backend, $($OtherBackend: Backend),+> {
                #[allow(missing_docs)]
                $DefaultBackend($DefaultBackend::Device),
                $(
                    #[allow(missing_docs)]
                    $OtherBackend($OtherBackend::Device),
                )+
            }
            impl<$DefaultBackend: Backend, $($OtherBackend: Backend),+> PartialEq for MultiDevice<$DefaultBackend, $($OtherBackend),+> {
                fn eq(&self, other: &Self) -> bool {
                    match (self, other) {
                        (Self::$DefaultBackend(lhs), Self::$DefaultBackend(rhs)) => lhs == rhs,
                        $(
                            (Self::$OtherBackend(lhs), Self::$OtherBackend(rhs)) => lhs == rhs,
                        )+
                        _ => false,
                    }
                }
            }

            // Default implementation always returns the first backend's device
            impl<$DefaultBackend: Backend, $($OtherBackend: Backend),+> Default for MultiDevice<$DefaultBackend, $($OtherBackend),+> {
                fn default() -> Self {
                    Self::$DefaultBackend($DefaultBackend::Device::default())
                }
            }

            impl<$DefaultBackend: Backend, $($OtherBackend: Backend),+> ruda_core::device::Device for MultiDevice<$DefaultBackend, $($OtherBackend),+> {
                fn from_id(device_id: DeviceId) -> Self {
                    let backends = [stringify!($DefaultBackend), $(stringify!($OtherBackend)),+];
                    let slot = (device_id.type_id >> 14) as usize;
                    let native = DeviceId::new(device_id.type_id & 0x3fff, device_id.index_id);
                    match *backends.get(slot).expect("invalid router backend slot") {
                        stringify!($DefaultBackend) => Self::$DefaultBackend(
                            <$DefaultBackend::Device as ruda_core::device::Device>::from_id(native)),
                        $(stringify!($OtherBackend) => Self::$OtherBackend(
                            <$OtherBackend::Device as ruda_core::device::Device>::from_id(native)),)+
                        _ => unreachable!(),
                    }
                }
                fn to_id(&self) -> DeviceId {
                    let backends = [stringify!($DefaultBackend), $(stringify!($OtherBackend)),+];
                    let (native, slot) = match self {
                        Self::$DefaultBackend(device) => (device.id(), 0),
                        $(Self::$OtherBackend(device) => (device.id(),
                            backends.iter().position(|name| *name == stringify!($OtherBackend)).unwrap() as u16),)+
                    };
                    assert!(native.type_id < 0x4000, "native device type uses reserved router slot bits");
                    DeviceId::new(native.type_id | (slot << 14), native.index_id)
                }
            }

            impl<$DefaultBackend: Backend, $($OtherBackend: Backend),+> DeviceOps for MultiDevice<$DefaultBackend, $($OtherBackend),+> {}

            /// A local client with multiple runners (each responsible to execute tensor operations on a given backend).
            #[derive(Clone)]
            pub enum MultiRunnerClient<$DefaultBackend: BackendIr, $($OtherBackend: BackendIr),+> {
                #[allow(missing_docs)]
                $DefaultBackend(Runner<$DefaultBackend>),
                $(
                    #[allow(missing_docs)]
                    $OtherBackend(Runner<$OtherBackend>),
                )+
            }

            impl<$DefaultBackend: BackendIr, $($OtherBackend: BackendIr),+> RunnerClient for MultiRunnerClient<$DefaultBackend, $($OtherBackend),+>
            {
               type Device = MultiDevice<$DefaultBackend, $($OtherBackend),+>;

                fn register_op(&self, op: OperationIr) {
                    match self {
                        Self::$DefaultBackend(runner) => runner.register_op(op),
                        $(
                            Self::$OtherBackend(runner) => runner.register_op(op),
                        )+
                    }
                }

                fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
                    match self {
                        Self::$DefaultBackend(runner) => runner.read_tensor_async(tensor),
                        $(
                            Self::$OtherBackend(runner) => runner.read_tensor_async(tensor),
                        )+
                    }
                }

                fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
                    match self {
                        Self::$DefaultBackend(runner) => {
                            let desc = runner.register_tensor_data_desc(data);
                            RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
                        }
                        $(
                            Self::$OtherBackend(runner) => {
                                let desc = runner.register_tensor_data_desc(data);
                                RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
                            }
                        )+
                    }
                }

                fn device(&self) -> Self::Device {
                    match self {
                        Self::$DefaultBackend(runner) => MultiDevice::$DefaultBackend(runner.device()),
                        $(
                            Self::$OtherBackend(runner) => MultiDevice::$OtherBackend(runner.device()),
                        )+
                    }
                }

                fn sync(&self) -> Result<(), ExecutionError> {
                    match self {
                        Self::$DefaultBackend(runner) => runner.sync(),
                        $(
                            Self::$OtherBackend(runner) => runner.sync(),
                        )+
                    }
                }

                fn seed(&self, seed: u64) {
                    match self {
                        Self::$DefaultBackend(runner) => runner.seed(seed),
                        $(
                            Self::$OtherBackend(runner) => runner.seed(seed),
                        )+
                    }
                }

                fn create_empty_handle(&self) -> TensorId {
                    match self {
                        Self::$DefaultBackend(runner) => runner.create_empty_handle(),
                        $(
                            Self::$OtherBackend(runner) => runner.create_empty_handle(),
                        )+
                    }
                }

                fn dtype_usage(&self, dtype: ruda_core::tensor::DType) -> ruda_tensor::DTypeUsageSet {
                    match self {
                        Self::$DefaultBackend(runner) => runner.dtype_usage(dtype),
                        $(
                            Self::$OtherBackend(runner) => runner.dtype_usage(dtype),
                        )+
                    }
                }
            }

            impl<$DefaultBackend: BackendIr, $($OtherBackend: BackendIr),+, Br> RunnerChannel for DirectChannel<($DefaultBackend, $($OtherBackend),+), Br>
            where
                Br: MultiBackendBridge<TensorHandle = Handle<$DefaultBackend, $($OtherBackend),+>, Device = MultiDevice<$DefaultBackend, $($OtherBackend),+>>,
            {
                type Device = Br::Device;

                type Bridge = Br;

                type FloatElem = $DefaultBackend::FloatElem;
                type IntElem = $DefaultBackend::IntElem;
                type BoolElem = $DefaultBackend::BoolElem;

                type Client = MultiRunnerClient<$DefaultBackend, $($OtherBackend),+>;

                fn init_client(device: &Self::Device) -> Self::Client {
                    match device {
                        MultiDevice::$DefaultBackend(device) => MultiRunnerClient::$DefaultBackend(Runner::new(device.clone())),
                        $(
                            MultiDevice::$OtherBackend(device) => MultiRunnerClient::$OtherBackend(Runner::new(device.clone())),
                        )+
                    }
                }

                fn get_tensor_handle(
                    tensor: &TensorIr,
                    client: &Self::Client,
                ) -> <Self::Bridge as MultiBackendBridge>::TensorHandle {
                    match client {
                        MultiRunnerClient::$DefaultBackend(runner) => Handle::$DefaultBackend(runner.get_tensor_handle(tensor)),
                        $(
                            MultiRunnerClient::$OtherBackend(runner) => Handle::$OtherBackend(runner.get_tensor_handle(tensor)),
                        )+
                    }
                }

                fn register_tensor(
                    client: &Self::Client,
                    handle: <Self::Bridge as MultiBackendBridge>::TensorHandle,
                    shape: Shape,
                    dtype: DType,
                ) -> RouterTensor<Self::Client> {
                    match client {
                        MultiRunnerClient::$DefaultBackend(runner) => match handle {
                            Handle::$DefaultBackend(handle) => runner.register_tensor(handle, shape, dtype, client.clone()),
                            _ => unreachable!("Can't register tensor handle for another backend."),
                        },
                        $(
                            MultiRunnerClient::$OtherBackend(runner) =>  match handle {
                                Handle::$OtherBackend(handle) => runner.register_tensor(handle, shape, dtype, client.clone()),
                                _ => unreachable!("Can't register tensor handle for another backend."),
                            },
                        )+
                    }
                }

                fn name(_device: &Self::Device) -> String {
                    let mut name = format!("{}", $DefaultBackend::name(&<$DefaultBackend::Device as Default>::default()));
                    $(
                        name.push_str(&format!(", {}", $OtherBackend::name(&<$OtherBackend::Device as Default>::default())));
                    )+
                    format!("direct<({})>", name)
                }
            }

            #[cfg(feature = "distributed")]
            impl<$DefaultBackend: BackendIr + ruda_tensor::distributed::DistributedBackend,
                $($OtherBackend: BackendIr + ruda_tensor::distributed::DistributedBackend),+, Br>
                crate::NativeCollectiveChannel for DirectChannel<($DefaultBackend, $($OtherBackend),+), Br>
            where
                Br: MultiBackendBridge<TensorHandle = Handle<$DefaultBackend, $($OtherBackend),+>,
                    Device = MultiDevice<$DefaultBackend, $($OtherBackend),+>>,
            {
                fn init_native_client(device: &Self::Device) -> Self::Client {
                    match device {
                        MultiDevice::$DefaultBackend(device) => MultiRunnerClient::$DefaultBackend(Runner::new_distributed(device.clone())),
                        $(MultiDevice::$OtherBackend(device) => MultiRunnerClient::$OtherBackend(Runner::new_distributed(device.clone())),)+
                    }
                }
                fn sync_native(client: &Self::Client) {
                    match client {
                        MultiRunnerClient::$DefaultBackend(runner) => runner.sync_native_collective(),
                        $(MultiRunnerClient::$OtherBackend(runner) => runner.sync_native_collective(),)+
                    }
                }
                fn native_group(device: &Self::Device, devices: &[DeviceId]) -> alloc::vec::Vec<DeviceId> {
                    let slot = device.id().type_id >> 14;
                    devices.iter().map(|id| {
                        assert_eq!(id.type_id >> 14, slot,
                            "native Router collectives require one backend slot; move tensors explicitly before reducing");
                        DeviceId::new(id.type_id & 0x3fff, id.index_id)
                    }).collect()
                }
            }

            impl<$DefaultBackend: BackendIr, $($OtherBackend: BackendIr),+> MultiBackendBridge for ByteBridge<($DefaultBackend, $($OtherBackend),+)> {
                type TensorHandle = Handle<$DefaultBackend, $($OtherBackend),+>;
                type Device = MultiDevice<$DefaultBackend, $($OtherBackend),+>;

                fn change_backend_float(
                    tensor: Self::TensorHandle,
                    shape: Shape,
                    target_device: &Self::Device,
                ) -> Self::TensorHandle {
                    multi_backend_match!(shape, (tensor, target_device) : $DefaultBackend, $($OtherBackend),+)
                }

                fn change_backend_int(
                    tensor: Self::TensorHandle,
                    shape: Shape,
                    target_device: &Self::Device,
                ) -> Self::TensorHandle {
                    multi_backend_match!(bridge_int; shape, (tensor, target_device) : $DefaultBackend, $($OtherBackend),+)
                }

                fn change_backend_bool(
                    tensor: Self::TensorHandle,
                    shape: Shape,
                    target_device: &Self::Device,
                ) -> Self::TensorHandle {
                    multi_backend_match!(bridge_bool; shape, (tensor, target_device) : $DefaultBackend, $($OtherBackend),+)
                }

                fn change_backend_quantized(
                    tensor: Self::TensorHandle,
                    shape: Shape,
                    target_device: &Self::Device,
                ) -> Self::TensorHandle {
                    multi_backend_match!(bridge_quantized; shape, (tensor, target_device) : $DefaultBackend, $($OtherBackend),+)
                }

            }
        }
    };
}

macro_rules! bridge {
    ($Backend:ident, $handle:expr, $device:expr, $shape:expr) => {{
        // Bridge for the same backend
        let tensor = $Backend::float_tensor(TensorHandle {
            handle: $handle,
            shape: $shape,
        });
        let tensor = $Backend::float_to_device(tensor, $device);
        let handle = $Backend::float_tensor_handle(tensor);
        Handle::$Backend(handle)
    }};
    ($BackendA:ident, $BackendB:ident, $handle:expr, $device:expr, $shape:expr) => {{
        // Byte bridge between two backends
        let tensor = $BackendA::float_tensor(TensorHandle { handle: $handle, shape: $shape });
        let data = try_read_sync($BackendA::float_into_data(tensor)).unwrap().expect(
            "Failed to read tensor data synchronously. This can happen on platforms that don't support blocking futures like WASM."
        );
        let tensor = $BackendB::float_from_data(data, $device);
        let handle = $BackendB::float_tensor_handle(tensor);
        Handle::$BackendB(handle)
    }};
}

macro_rules! bridge_kind {
    ($from_handle:ident, $to_device:ident, $into_data:ident, $from_data:ident, $to_handle:ident;
        $Backend:ident, $handle:expr, $device:expr, $shape:expr) => {{
        let tensor = $Backend::$from_handle(TensorHandle { handle: $handle, shape: $shape });
        let tensor = $Backend::$to_device(tensor, $device);
        Handle::$Backend($Backend::$to_handle(tensor))
    }};
    ($from_handle:ident, $to_device:ident, $into_data:ident, $from_data:ident, $to_handle:ident;
        $BackendA:ident, $BackendB:ident, $handle:expr, $device:expr, $shape:expr) => {{
        let tensor = $BackendA::$from_handle(TensorHandle { handle: $handle, shape: $shape });
        let data = try_read_sync($BackendA::$into_data(tensor)).unwrap().expect(
            "Failed to read tensor data synchronously. Blocking futures may be unsupported on this platform."
        );
        let tensor = $BackendB::$from_data(data, $device);
        Handle::$BackendB($BackendB::$to_handle(tensor))
    }};
}

macro_rules! bridge_int {
    ($($args:tt)*) => {
        bridge_kind!(int_tensor, int_to_device, int_into_data, int_from_data, int_tensor_handle; $($args)*)
    };
}

macro_rules! bridge_bool {
    ($($args:tt)*) => {
        bridge_kind!(bool_tensor, bool_to_device, bool_into_data, bool_from_data, bool_tensor_handle; $($args)*)
    };
}

macro_rules! bridge_quantized {
    ($Backend:ident, $handle:expr, $device:expr, $shape:expr) => {{
        let tensor = $Backend::quantized_tensor(TensorHandle { handle: $handle, shape: $shape });
        let tensor = $Backend::q_to_device(tensor, $device);
        Handle::$Backend($Backend::quantized_tensor_handle(tensor))
    }};
    ($BackendA:ident, $BackendB:ident, $handle:expr, $device:expr, $shape:expr) => {{
        let tensor = $BackendA::quantized_tensor(TensorHandle { handle: $handle, shape: $shape });
        let data = try_read_sync($BackendA::q_into_data(tensor)).unwrap().expect(
            "Failed to read quantized tensor data synchronously. Blocking futures may be unsupported on this platform."
        );
        let tensor = $BackendB::q_from_data(data, $device);
        Handle::$BackendB($BackendB::quantized_tensor_handle(tensor))
    }};
}

macro_rules! multi_backend_match {
    ($shape:expr, ($handle:expr, $device:expr) : $DefaultBackend:ident, $($OtherBackend:ident),+) => {
        multi_backend_match!(bridge; $shape, ($handle, $device) : $DefaultBackend, $($OtherBackend),+)
    };
    ($bridge:ident; $shape:expr, ($handle:expr, $device:expr) : $DefaultBackend:ident, $($OtherBackend:ident),+) => {
        multi_backend_match! (
            @step
            $bridge;
            $shape,
            ($handle, $device);
            {
                (Handle::$DefaultBackend(handle), MultiDevice::$DefaultBackend(device)) => $bridge!($DefaultBackend, handle, device, $shape),
                $(
                    (Handle::$DefaultBackend(handle), MultiDevice::$OtherBackend(device)) => $bridge!($DefaultBackend, $OtherBackend, handle, device, $shape),
                    (Handle::$OtherBackend(handle), MultiDevice::$DefaultBackend(device)) => $bridge!($OtherBackend, $DefaultBackend, handle, device, $shape),
                    (Handle::$OtherBackend(handle), MultiDevice::$OtherBackend(device)) => $bridge!($OtherBackend, handle, device, $shape),
                )+
            };
            $($OtherBackend),+
        )
    };

    (@step
        $bridge:ident;
        $shape:expr,
        $pats:tt;
        { $($arms:tt)* };
        $BackendA:ident,
        $($OtherBackend:ident),+
    ) => {
        multi_backend_match! (
            @step
            $bridge;
            $shape,
            $pats;
            {
                $($arms)*
                $(
                    (Handle::$BackendA(handle), MultiDevice::$OtherBackend(device)) => $bridge!($BackendA, $OtherBackend, handle, device, $shape),
                    (Handle::$OtherBackend(handle), MultiDevice::$BackendA(device)) => $bridge!($OtherBackend, $BackendA, handle, device, $shape),
                )*
            };
            $($OtherBackend),*
        )
    };

    (@step
        $bridge:ident;
        $shape:expr,
        ($handle:expr, $device:expr);
        { $($arms:tt)* };
        $($BackendA:ident)?
    ) => {
        match ($handle, $device) {
            $($arms)*
        }
    };
}

// Implement multi-backend types and byte bridge for up to 4 backends
impl_multi_backend_types!(duo, B1, B2);
impl_multi_backend_types!(trio, B1, B2, B3);
impl_multi_backend_types!(quad, B1, B2, B3, B4);

#[cfg(not(target_os = "windows"))] // cannot find a wgpu adapter on windows CI
#[cfg(test)]
mod tests {
    use ruda_tensor::api::Tensor;

    use super::*;
    use crate::tests::TestBackend;

    #[test]
    fn should_support_dual_byte_bridge() {
        let device1 = duo::MultiDevice::B1(Default::default());
        let device2 = duo::MultiDevice::B2(Default::default());
        let tensor1 = Tensor::<TestBackend, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device1);
        let tensor2 = Tensor::<TestBackend, 1>::from_floats([5.0, 6.0, 7.0, 8.0], &device2);

        let tensor1_2 = tensor1.clone().to_device(&device2);
        tensor1.into_data().assert_eq(&tensor1_2.into_data(), true);

        let tensor2_1 = tensor2.clone().to_device(&device1);
        tensor2.into_data().assert_eq(&tensor2_1.into_data(), true);
    }
}

#[cfg(test)]
mod device_id_tests {
    use super::*;
    use ruda_core::device::Device;
    use ruda_tensor_host::{Host, HostDevice};
    type D = duo::MultiDevice<Host,Host>;
    type C = crate::DirectByteChannel<(Host,Host)>;

    #[test]
    fn identical_native_ids_in_different_backend_slots_do_not_collide() {
        let first = D::B1(HostDevice);
        let second = D::B2(HostDevice);
        assert_ne!(first.id(),second.id());
        assert_eq!(D::from_id(first.id()),first);
        assert_eq!(D::from_id(second.id()),second);
        assert!(matches!(crate::get_client::<C>(&first),duo::MultiRunnerClient::B1(_)));
        assert!(matches!(crate::get_client::<C>(&second),duo::MultiRunnerClient::B2(_)));
    }
}
