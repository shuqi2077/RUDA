use crate::collections::HashMap;
use core::any::TypeId;
use ruda_tensor::{
    DeviceId, DeviceOps, TensorPrimitive,
    distributed::{DistributedBackend, DistributedParams, TensorRef},
    tensor::TensorContainer,
};

use crate::{NodeId, grads::GradID};

/// Trait for registering distributed gradients.
pub trait DistributedRegistration {
    /// Performs distributed registration operations on the tensor with the corresponding [`NodeId`].
    fn on_register(&mut self, node_id: &NodeId, container: &mut TensorContainer<GradID>);

    /// Wait on the registered backend and device, then publish synchronized gradients.
    /// The requested backend and device must match the registration while it is active.
    /// Repeated calls after completion must not submit another collective round.
    fn sync(
        &mut self,
        backend: TypeId,
        device: DeviceId,
        container: &mut TensorContainer<GradID>,
    );
}

/// Submits sync operations on gradient registrations.
pub struct DistributedGradientRegistration<B: DistributedBackend> {
    n_required_map: HashMap<NodeId, usize>,
    sharded_parameters_map: HashMap<NodeId, DistributedParams>,
    pending: HashMap<GradID, TensorRef<B>>,
    device: B::Device,
    completed: bool,
}

impl<B: DistributedBackend> DistributedGradientRegistration<B> {
    /// Create registration state for one backward pass.
    pub fn new(
        n_required_map: HashMap<NodeId, usize>,
        sharded_parameters_map: HashMap<NodeId, DistributedParams>,
        device: B::Device,
    ) -> Self {
        Self {
            n_required_map,
            sharded_parameters_map,
            pending: HashMap::default(),
            device,
            completed: false,
        }
    }
}

impl<B: DistributedBackend> DistributedRegistration for DistributedGradientRegistration<B> {
    fn on_register(&mut self, id: &NodeId, container: &mut TensorContainer<GradID>) {
        if self.completed {
            return;
        }
        if let Some(sharded_params) = self.sharded_parameters_map.get(id) {
            let n_required = self.n_required_map.get_mut(id).unwrap();
            *n_required -= 1;

            if *n_required == 0 {
                let gradient = container.get::<B>(&id.value).unwrap();
                let TensorPrimitive::Float(gradient) = gradient else {
                    panic!("quantized gradient synchronization is not supported");
                };
                let tensor_ref = TensorRef::new(gradient);
                self.pending.insert(id.value, tensor_ref.clone());
                B::submit_gradient_sync(tensor_ref, sharded_params.clone());
            }
        }
    }

    fn sync(
        &mut self,
        backend: TypeId,
        device: DeviceId,
        container: &mut TensorContainer<GradID>,
    ) {
        if self.completed {
            return;
        }
        assert_eq!(backend, TypeId::of::<B>(), "gradient sync backend does not match registration");
        assert_eq!(device, self.device.id(), "gradient sync device does not match registration");
        B::submit_sync_collective(&self.device);
        for (id, tensor) in self.pending.drain() {
            // Safety: the registered backend's completion callback has returned
            // after synchronizing the device that submitted these gradients.
            let gradient = unsafe { B::float_from_ref(&tensor) };
            container.register::<B>(id, TensorPrimitive::Float(gradient));
        }
        self.completed = true;
    }
}
