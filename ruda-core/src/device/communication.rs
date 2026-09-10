use super::DeviceId;
use ahash::AHasher;
use alloc::vec::Vec;
use core::hash::{Hash, Hasher};

/// An ID unique to any unordered combination of devices.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct CommunicationId {
    /// The ID as a `String`.
    pub id: u64,
}

impl From<Vec<DeviceId>> for CommunicationId {
    fn from(mut value: Vec<DeviceId>) -> Self {
        // Make sure that device ids are sorted so that any combination of the same devices uses the same communicator.
        value.sort();
        let mut hasher = AHasher::default();
        value.hash(&mut hasher);
        CommunicationId {
            id: hasher.finish(),
        }
    }
}
