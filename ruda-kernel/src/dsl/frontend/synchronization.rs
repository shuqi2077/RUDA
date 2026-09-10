use crate::dsl::{
    ir::{Scope, Synchronization},
    unexpanded,
};

// Among all backends, the memory order guarantee of WebGPU is the weakest
// So Ruda's memory order cannot be stronger than that of WebGPU

/// # Coordinates the following among all invocations in the current ruda:
///
/// * Memory writes to variables in ruda address space(shared memory) complete,
///   e.g. writes that were initiated actually land in the ruda address space memory.
///
/// * Then all the invocations in the ruda wait for each other to arrive at the barrier, i.e. this step.
///
/// * Then all the invocations int the ruda begin executing after the barrier, and all writes to ruda address space made before the barrier are now visible to any invocation in this ruda.
pub fn sync_ruda() {}

pub mod sync_ruda {
    use super::*;

    pub fn expand(scope: &mut Scope) {
        scope.register(Synchronization::SyncRuda)
    }
}

/// Synchronizes units within their plane (e.g., warp or SIMD group).
///
/// Warning: not all targets support plane-level synchronization.
pub fn sync_plane() {
    unexpanded!()
}

pub mod sync_plane {
    use super::*;

    pub fn expand(scope: &mut Scope) {
        scope.register(Synchronization::SyncPlane);
    }
}

/// * `Sync_storage` is the same but change "ruda address space(shared memory)" to "storage address space(input args)". But the set of invocations that are collaborating is still only the invocations in the same ruda.
///
/// * There is no guarantee about using barriers alone to make the writes to storage buffer in one ruda become visible to invocations in a different ruda.
pub fn sync_storage() {}

pub mod sync_storage {
    use super::*;

    pub fn expand(scope: &mut Scope) {
        scope.register(Synchronization::SyncStorage)
    }
}

/// `sync_async_proxy_shared` is a synchronization fence for the experimental SM 9.0+ copy
/// functions, applying bidirectionally between the async proxy (i.e. TMA) and shared memory.
/// Should be used after initializing the barriers, and before the copy operation.
/// PTX: `fence.proxy.async.shared::cta`
/// Experimental and subject to change.
pub fn sync_async_proxy_shared() {
    unexpanded!()
}

pub mod sync_async_proxy_shared {
    use super::*;

    pub fn expand(scope: &mut Scope) {
        scope.register(Synchronization::SyncAsyncProxyShared)
    }
}
