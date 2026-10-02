//! Opt-in native collective routing. Every group stays within one backend slot.
//! There is no hidden CPU rendezvous or cross-runtime transport emulation.
use alloc::{format, string::String, vec::Vec};
use core::marker::PhantomData;
use ruda_tensor::{DType, DeviceId, DeviceOps, Shape,
    distributed::{CollectiveTensor, DistributedBackend, ReduceOperation},
    graph::{AllReduceOpIr, DistributedOperationIr, OperationIr, TensorIr},
    tensor::{Device, FloatTensor}};
use crate::{BackendRouter, MultiBackendBridge, RouterTensor, RunnerChannel, RunnerClient, get_client};

/// A channel whose runners can enqueue and synchronize native collectives.
/// Implementations must preserve the ordering/lifetime contract of CollectiveTensor.
pub trait NativeCollectiveChannel: RunnerChannel {
    /// Construct runners with their native collective execution capability.
    fn init_native_client(device: &Self::Device) -> Self::Client;
    /// Wait for previously enqueued collective work on this runner.
    fn sync_native(client: &Self::Client);
    /// Validate the router group and translate IDs into native device IDs.
    fn native_group(device: &Self::Device, devices: &[DeviceId]) -> Vec<DeviceId>;
}

/// Explicitly enable native collective routing for a supporting local channel.
///
/// All backends in the underlying channel must implement DistributedBackend;
/// any individual collective must use devices from a single backend slot.
/// Different slots still support explicit tensor transfers via their bridge.
#[derive(Clone)]
pub struct NativeDistributedChannel<R: NativeCollectiveChannel>(PhantomData<R>);

impl<R: NativeCollectiveChannel> RunnerChannel for NativeDistributedChannel<R> {
    type Device = R::Device;
    type Bridge = R::Bridge;
    type Client = R::Client;
    type FloatElem = R::FloatElem;
    type IntElem = R::IntElem;
    type BoolElem = R::BoolElem;
    fn name(device: &Self::Device) -> String { format!("native-distributed<{}>", R::name(device)) }
    fn init_client(device: &Self::Device) -> Self::Client { R::init_native_client(device) }
    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client)
        -> <Self::Bridge as MultiBackendBridge>::TensorHandle
    { R::get_tensor_handle(tensor, client) }
    fn register_tensor(client: &Self::Client, handle: <Self::Bridge as MultiBackendBridge>::TensorHandle,
        shape: Shape, dtype: DType) -> RouterTensor<Self::Client>
    { R::register_tensor(client, handle, shape, dtype) }
}

impl<R: NativeCollectiveChannel> DistributedBackend for BackendRouter<NativeDistributedChannel<R>> {
    fn all_reduce(tensor: FloatTensor<Self>, op: ReduceOperation, device_ids: Vec<DeviceId>) -> CollectiveTensor<Self> {
        let client = tensor.client.clone();
        let mut unique = device_ids.clone(); unique.sort(); unique.dedup();
        assert!(!device_ids.is_empty() && unique.len() == device_ids.len(), "collective participants must be nonempty and unique");
        assert!(device_ids.contains(&client.device().id()), "caller must belong to the collective group");
        // Canonical order prevents different caller orderings from constructing
        // different native communicator keys for the same participant set.
        let native = R::native_group(&client.device(), &unique);
        let desc = AllReduceOpIr::create(tensor.into_ir(), op,
            native.iter().map(|id| (id.type_id,id.index_id)).collect(), || client.create_empty_handle());
        let mut output = client.register(OperationIr::Distributed(DistributedOperationIr::AllReduce(desc)));
        CollectiveTensor::new(output.pop().expect("all-reduce must produce one tensor"))
    }
    fn sync_collective(device: &Device<Self>) {
        let client = get_client::<NativeDistributedChannel<R>>(device);
        R::sync_native(&client);
    }
}
