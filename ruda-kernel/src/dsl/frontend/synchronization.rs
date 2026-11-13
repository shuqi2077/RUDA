use crate::dsl::{
    ir::{Scope, Synchronization},
    unexpanded,
};

/// Load a shared-memory value uniformly after synchronizing the workgroup.
pub fn workgroup_uniform_load<E: crate::dsl::frontend::RudaPrimitive>(_value: &E) -> E {
    unexpanded!()
}

pub mod workgroup_uniform_load {
    use super::*;
    use crate::dsl::frontend::{NativeExpand, RudaPrimitive};
    use ruda_core::ir::{Instruction, PointerOp};

    pub fn expand<E: RudaPrimitive>(scope: &mut Scope, value: NativeExpand<E>) -> NativeExpand<E> {
        let pointer = scope
            .reference_pointer(*value.expand)
            .expect("uniform load requires an address-preserving reference");
        let out = scope.create_local(
            pointer
                .ty
                .pointer_type()
                .expect("reference pointer")
                .pointee(),
        );
        let operation = PointerOp::UniformLoad { pointer };
        operation
            .validate(Some(*out))
            .expect("uniform load requires a matching shared-memory pointee");
        scope.register(Instruction::new(operation, *out));
        out.into()
    }
}

/// Atomically load a shared-memory value uniformly after synchronizing the workgroup.
pub fn workgroup_uniform_load_atomic<
    E: crate::dsl::frontend::RudaPrimitive<Scalar: crate::dsl::frontend::Numeric>,
>(
    _value: &crate::dsl::frontend::Atomic<E>,
) -> E {
    unexpanded!()
}

pub mod workgroup_uniform_load_atomic {
    use super::*;
    use crate::dsl::frontend::{Atomic, NativeExpand, Numeric, RudaPrimitive};
    use ruda_core::ir::{Instruction, PointerOp};

    pub fn expand<E: RudaPrimitive<Scalar: Numeric>>(
        scope: &mut Scope,
        value: NativeExpand<Atomic<E>>,
    ) -> NativeExpand<E> {
        let pointer = scope
            .reference_pointer(*value.expand)
            .expect("uniform atomic load requires an address-preserving reference");
        let pointee = pointer
            .ty
            .pointer_type()
            .expect("reference pointer")
            .pointee();
        let out = scope.create_local(
            ruda_core::ir::Type::scalar(pointee.elem_type())
                .with_vector_size(pointee.vector_size()),
        );
        let operation = PointerOp::UniformLoad { pointer };
        operation
            .validate(Some(*out))
            .expect("uniform atomic load requires a matching shared-memory atomic pointee");
        scope.register(Instruction::new(operation, *out));
        out.into()
    }
}

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

/// Synchronizes every unit in the ruda, including both shared and storage memory.
/// When `Features::device_memory_scope` is true, this also releases prior storage writes
/// to other rudas and acquires writes they published before their own synchronization.
/// Otherwise the memory-ordering guarantee is limited to the current ruda.
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

/// Publish/acquire global writes across peer devices. This is a memory fence,
/// not a CTA barrier; each publishing unit must call it before its ready flag.
/// Supported on native CUDA/HIP; shader-only targets reject system scope.
pub fn fence_system() {
    unexpanded!()
}
pub mod fence_system {
    use super::*;
    pub fn expand(scope: &mut Scope) {
        scope.register(Synchronization::FenceSystem);
    }
}
