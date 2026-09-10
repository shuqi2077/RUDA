use super::{Result, emit::Emitter, invalid, types::Scalar};
use crate::optimizer::{Optimizer, SharedLiveness};
use ruda_core::{
    launch::RudaDim,
    ir::{Scope, Variable, VariableKind},
};

impl Emitter {
    pub fn shared_array(&mut self, variable: Variable) -> Result<String> {
        let VariableKind::SharedArray {
            id,
            length,
            unroll_factor,
            alignment,
        } = variable.kind
        else {
            return Err(invalid("expected shared array"));
        };
        let ty = Scalar::memory_element(variable.ty)?;
        if unroll_factor == 0 {
            return Err(invalid("shared array unroll factor must be nonzero"));
        }
        let element_bytes = ty.bytes().checked_mul(variable.ty.vector_size())
            .ok_or_else(|| invalid("shared vector element size overflow"))?;
        let natural_alignment = element_bytes.checked_next_power_of_two()
            .ok_or_else(|| invalid("shared vector alignment overflow"))?;
        let align = alignment.unwrap_or(natural_alignment);
        if !align.is_power_of_two() || align < ty.bytes() {
            return Err(invalid(
                "shared alignment must be a power of two and cover its element",
            ));
        }
        length
            .checked_mul(unroll_factor)
            .and_then(|length| length.checked_mul(element_bytes))
            .ok_or_else(|| invalid("shared array size overflow"))?;
        if let Some(previous) = self.shared_arrays.insert(id, variable) {
            if previous != variable {
                return Err(invalid("inconsistent shared array declaration"));
            }
        }
        Ok(format!("%shared_{id}"))
    }

    pub fn allocate_shared(&mut self, scope: Scope, ruda_dim: RudaDim) -> Result<usize> {
        if self.shared_arrays.is_empty() && self.unit_barriers.is_empty() {
            return Ok(0);
        }
        let mut optimizer = Optimizer::shared_only(scope, ruda_dim);
        let liveness = optimizer.analysis::<SharedLiveness>();
        let mut allocations = liveness.allocations.values().collect::<Vec<_>>();
        allocations.sort_by_key(|allocation| allocation.smem.id());
        let mut bytes = 0;
        let mut alignment = 1;
        for allocation in allocations {
            let id = allocation.smem.id();
            if !self.shared_arrays.contains_key(&id) {
                return Err(invalid("shared allocation has no PTX declaration"));
            }
            bytes = bytes.max(
                allocation
                    .offset
                    .checked_add(allocation.smem.size())
                    .ok_or_else(|| invalid("shared allocation size overflow"))?,
            );
            alignment = alignment.max(allocation.smem.align());
            self.declarations += &format!("    .reg .u64 %shared_{id};\n");
            self.prologue += &format!(
                "    mov.u64 %shared_{id}, dynamic_shared_mem;\n    add.u64 %shared_{id}, %shared_{id}, {};\n",
                allocation.offset
            );
        }
        if self.shared_arrays.len() != liveness.allocations.len() {
            return Err(invalid("shared allocation missing from liveness analysis"));
        }
        if !self.unit_barriers.is_empty() {
            alignment = alignment.max(8);
            let unit = self.value(Variable::builtin(
                ruda_core::ir::Builtin::UnitPos,
                ruda_core::ir::StorageType::Scalar(ruda_core::ir::ElemType::UInt(ruda_core::ir::UIntKind::U64)),
            ))?;
            let per_barrier = (ruda_dim.x as usize)
                .checked_mul(ruda_dim.y as usize).and_then(|value| value.checked_mul(ruda_dim.z as usize))
                .and_then(|value| value.checked_mul(8))
                .ok_or_else(|| invalid("per-unit barrier allocation overflow"))?;
            for (_, pointer) in &self.unit_barriers {
                bytes = bytes.checked_add(7).map(|value| value & !7)
                    .ok_or_else(|| invalid("barrier alignment overflow"))?;
                self.prologue += &format!(
                    "    mov.u64 {pointer}, dynamic_shared_mem;\n    add.u64 {pointer}, {pointer}, {bytes};\n    mad.lo.u64 {pointer}, {unit}, 8, {pointer};\n"
                );
                bytes = bytes.checked_add(per_barrier)
                    .ok_or_else(|| invalid("barrier allocation overflow"))?;
            }
        }
        self.shared_declaration =
            format!(".extern .shared .align {alignment} .b8 dynamic_shared_mem[];\n");
        Ok(bytes)
    }
}
