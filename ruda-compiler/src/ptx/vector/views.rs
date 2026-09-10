use super::*;

impl Emitter {
    pub(super) fn unrolled_memory_view(&self, array: Variable, unroll: usize) -> Result<Variable> {
        match array.kind {
            VariableKind::GlobalInputArray(id) | VariableKind::GlobalOutputArray(id) => {
                let declared = self.buffers.get(id as usize)
                    .ok_or_else(|| invalid("unrolled buffer argument index out of range"))?;
                if declared.ty == array.ty { return Ok(array); }
                if unroll <= 1 || array.storage_type() != declared.ty.storage_type()
                    || array.ty.vector_size().checked_mul(unroll) != Some(declared.ty.vector_size())
                {
                    return Err(invalid("unrolled buffer view does not match argument storage"));
                }
                Ok(Variable::new(array.kind, declared.ty))
            }
            VariableKind::ConstantArray { id, length, unroll_factor } => {
                let (declared, _) = self.constant_arrays.get(&id)
                    .ok_or_else(|| invalid("unrolled constant array initializer missing"))?;
                if *declared == array { return Ok(array); }
                let VariableKind::ConstantArray { length: declared_length, unroll_factor: declared_unroll, .. } = declared.kind else {
                    return Err(invalid("unrolled constant array declaration kind mismatch"));
                };
                if unroll <= 1 || unroll_factor != unroll || declared_unroll != 1
                    || length != declared_length || array.storage_type() != declared.storage_type()
                    || array.ty.vector_size().checked_mul(unroll) != Some(declared.ty.vector_size())
                {
                    return Err(invalid("unrolled constant view does not match initialized storage"));
                }
                Ok(*declared)
            }
            _ => Ok(array),
        }
    }
}
