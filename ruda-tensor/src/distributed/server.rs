use std::collections::{HashMap, HashSet};

use crate::{DeviceId, DeviceOps, tensor::Device};

use crate::distributed::{
    DistributedBackend, DistributedConfig, DistributedParamId, DistributedParams, TensorRef,
    client::DistributedSyncMessage,
};

pub(crate) struct DistributedSyncServer<B: DistributedBackend> {
    config: DistributedConfig,
    all_reduce_ops_queue: HashMap<DistributedParamId, Vec<TensorRef<B>>>,
    param_required_map: HashMap<DistributedParamId, usize>,
    devices: HashSet<DeviceId>,
    devices_registered: HashSet<DeviceId>,
    collective_devices: HashSet<DeviceId>,
    syncing_devices: Vec<Device<B>>,
    callbacks: HashMap<DeviceId, oneshot::Sender<Box<dyn FnOnce() + Send>>>,
}

impl<B: DistributedBackend> DistributedSyncServer<B> {
    /// Create a new gradient sync server instance.
    pub(crate) fn new(devices: HashSet<DeviceId>, config: DistributedConfig) -> Self {
        Self {
            config,
            all_reduce_ops_queue: HashMap::default(),
            param_required_map: HashMap::default(),
            devices,
            devices_registered: HashSet::default(),
            collective_devices: HashSet::default(),
            syncing_devices: vec![],
            callbacks: HashMap::default(),
        }
    }

    /// Process message from client.
    pub(crate) fn process_message(&mut self, msg: DistributedSyncMessage<B>) {
        match msg {
            DistributedSyncMessage::RegisterSyncParameters(device, params) => {
                self.register_sync_params(device, params)
            }
            DistributedSyncMessage::TensorSync((tensor, params)) => {
                self.register_tensor(tensor, params)
            }
            DistributedSyncMessage::CollectiveSync((device, callback)) => {
                self.collective_sync(device, callback)
            }
        }
    }

    /// Called at the start of the backward process. Lets the device announce what parameters are nodes in the autodiff graph and how many times they are required.
    fn register_sync_params(&mut self, device: DeviceId, sharded_params: Vec<DistributedParams>) {
        assert!(self.devices.contains(&device), "device is not in the gradient sync group");
        assert!(
            self.devices_registered.insert(device),
            "device registered gradients twice in the same sync round"
        );
        sharded_params.iter().for_each(|params| {
            *self.param_required_map.entry(params.param_id).or_insert(0) += 1;
        });
        self.launch_ops();
        self.try_launch_sync();
    }

    /// Called on registration of a gradient. Calls the all_reduce operation for any parameter that is no longer required in the autodiff graph.
    fn register_tensor(&mut self, tensor: TensorRef<B>, sharded_params: DistributedParams) {
        let op_queue = self
            .all_reduce_ops_queue
            .entry(sharded_params.param_id)
            .or_insert(vec![]);
        op_queue.push(tensor.clone());
        self.launch_ops();
    }

    fn collective_sync(
        &mut self,
        device: Device<B>,
        callback: oneshot::Sender<Box<dyn FnOnce() + Send>>,
    ) {
        assert!(
            self.devices_registered.contains(&device.id()),
            "device must register this round's parameters before gradient sync"
        );
        assert!(
            self.callbacks.insert(device.id(), callback).is_none(),
            "device submitted gradient completion twice in the same sync round"
        );
        self.syncing_devices.push(device.clone());
        self.try_launch_sync();
    }

    fn try_launch_sync(&mut self) {
        if self.devices_registered.len() == self.devices.len()
            && self.param_required_map.is_empty()
            && self.all_reduce_ops_queue.is_empty()
            && self.callbacks.len() == self.devices.len()
        {
            // Retire the round before releasing any device into its next backward pass.
            self.devices_registered.clear();
            for d in self.syncing_devices.drain(..) {
                let callback = self.callbacks.remove(&d.id()).unwrap();
                let has_collective = self.collective_devices.remove(&d.id());
                let closure = Box::new(move || {
                    if has_collective {
                        B::sync_collective(&d);
                    }
                });
                callback.send(closure).expect("Can send callback");
            }
        }
    }

    fn launch_ops(&mut self) {
        if self.devices_registered.len() == self.devices.len() {
            for (param_id, num_tensors) in self.param_required_map.clone() {
                let queued_tensors = self.all_reduce_ops_queue.entry(param_id).or_insert(vec![]);

                if num_tensors == queued_tensors.len() {
                    // Safety: the queue owns the handles; device work is synchronized
                    // by the completion callback before gradients are published.
                    let device_ids = queued_tensors
                        .iter()
                        .map(|tensor| unsafe { B::comm_device(tensor) }.id())
                        .collect::<Vec<_>>();
                    self.collective_devices.extend(device_ids.iter().copied());
                    let reduced_tensors: Vec<B::FloatTensorPrimitive> = queued_tensors
                        .iter()
                        .map(|tensor|
                            // Safety: we can call `assume_resolved` on these tensors since we know `B::sync_collective` is called
                            // at the end of the backward pass.
                            unsafe {
                            B::all_reduce(
                                B::float_from_ref(tensor),
                                self.config.all_reduce_op,
                                device_ids.clone(),
                            )
                            .assume_resolved()
                        })
                        .collect();

                    for (tensor_ref, reduced_tensor) in queued_tensors.iter().zip(reduced_tensors) {
                        tensor_ref.replace(reduced_tensor);
                    }

                    self.all_reduce_ops_queue.remove(&param_id).unwrap();
                    self.param_required_map.remove(&param_id).unwrap();
                    self.try_launch_sync();
                }
            }
        }
    }
}
