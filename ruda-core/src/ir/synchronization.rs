use core::fmt::Display;

use crate::ir::{OperationReflect, TypeHash};

/// All synchronization types.
#[cfg_attr(feature = "ir-serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = SyncOpCode)]
#[allow(missing_docs)]
pub enum Synchronization {
    // Synchronizize units in a ruda.
    SyncRuda,
    // Synchronize units within their plane
    SyncPlane,
    SyncStorage,
    /// Sync CTA proxy.
    /// Experimental, CUDA only, SM 9.0+ only
    SyncAsyncProxyShared,
}

impl Display for Synchronization {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Synchronization::SyncRuda => write!(f, "sync_ruda()"),
            Synchronization::SyncStorage => write!(f, "sync_storage()"),
            Synchronization::SyncAsyncProxyShared => write!(f, "sync_proxy_shared()"),
            Synchronization::SyncPlane => write!(f, "sync_plane()"),
        }
    }
}
