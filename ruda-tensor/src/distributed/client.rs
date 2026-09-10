use std::{collections::HashSet, sync::mpsc::Sender, thread::spawn};

use crate::tensor::Device;
use crate::{DeviceId, DeviceOps};

use crate::distributed::{
    DistributedBackend, DistributedConfig, DistributedParams, TensorRef,
    server::DistributedSyncServer,
};

pub(crate) enum ActionMessage<B: DistributedBackend> {
    Message(DistributedSyncMessage<B>),
    Close(),
}

pub(crate) enum DistributedSyncMessage<B: DistributedBackend> {
    RegisterSyncParameters(DeviceId, Vec<DistributedParams>),
    TensorSync((TensorRef<B>, DistributedParams)),
    #[allow(clippy::type_complexity)]
    CollectiveSync((Device<B>, oneshot::Sender<Box<dyn FnOnce() + Send>>)),
}

#[derive(Clone)]
pub struct DistributedSyncClient<B: DistributedBackend> {
    sender: Sender<ActionMessage<B>>,
}

impl<B: DistributedBackend> DistributedSyncClient<B> {
    pub(crate) fn new(devices: &[Device<B>], config: DistributedConfig) -> Self {
        let device_ids = devices.iter().map(DeviceOps::id).collect::<HashSet<_>>();
        assert!(!device_ids.is_empty(), "gradient sync requires at least one device");
        assert_eq!(device_ids.len(), devices.len(), "gradient sync devices must be distinct");
        let (tx, rx) = std::sync::mpsc::channel();

        let mut server = DistributedSyncServer::new(device_ids, config);
        spawn(move || {
            while let ActionMessage::Message(msg) =
                rx.recv().expect("Gradient sync server disconnected.")
            {
                server.process_message(msg)
            }
        });
        Self { sender: tx }
    }

    pub fn register_sync_parameters(&self, device: &Device<B>, sharded_params: Vec<DistributedParams>) {
        self.sender
            .send(ActionMessage::Message(
                DistributedSyncMessage::RegisterSyncParameters(device.id(), sharded_params),
            ))
            .unwrap();
    }

    pub fn submit_gradient_sync(&self, tensor: TensorRef<B>, params: DistributedParams) {
        self.sender
            .send(ActionMessage::Message(DistributedSyncMessage::TensorSync(
                (tensor, params),
            )))
            .unwrap();
    }

    pub fn submit_sync_collective(&self, device: Device<B>) {
        let (tx, rx) = oneshot::channel();

        self.sender
            .send(ActionMessage::Message(
                DistributedSyncMessage::CollectiveSync((device.clone(), tx)),
            ))
            .unwrap();

        let sync = rx.recv().expect("Can receive callback");

        sync();
    }

    pub(crate) fn close(&self) {
        self.sender.send(ActionMessage::Close()).unwrap();
    }
}
