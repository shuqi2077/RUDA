use rspirv::spirv::{MemorySemantics, Scope};
use ruda_core::ir::Synchronization;

use crate::spirv::{SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_sync(&mut self, sync: Synchronization) {
        match sync {
            Synchronization::SyncRuda => {
                // Adopting wgpu semantics
                let scope = self.const_u32(Scope::Workgroup as u32);
                let semantics =
                    MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY;
                let semantics = self.const_u32(semantics.bits());
                self.control_barrier(scope, scope, semantics).unwrap();
            }
            Synchronization::SyncPlane => {
                let scope = self.const_u32(Scope::Subgroup as u32);
                let semantics = MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::SUBGROUP_MEMORY;
                let semantics = self.const_u32(semantics.bits());
                self.control_barrier(scope, scope, semantics).unwrap();
            }
            Synchronization::SyncStorage => {
                let scope_exec = self.const_u32(Scope::Workgroup as u32);
                let scope_mem = self.const_u32(Scope::Device as u32);
                let semantics = MemorySemantics::ACQUIRE_RELEASE
                    | MemorySemantics::UNIFORM_MEMORY
                    | MemorySemantics::WORKGROUP_MEMORY
                    | MemorySemantics::MAKE_AVAILABLE
                    | MemorySemantics::MAKE_VISIBLE;
                let semantics = self.const_u32(semantics.bits());
                self.control_barrier(scope_exec, scope_mem, semantics)
                    .unwrap();
            }
            Synchronization::FenceSystem => {
                panic!("system-scope peer memory is not supported in SPIR-V shaders")
            }
            Synchronization::SyncAsyncProxyShared => {
                panic!("TMA proxy sync not supported in SPIR-V")
            }
        }
    }
}
