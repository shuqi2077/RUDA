use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::{ir::{Type, Variable, VariableKind}, kernel::Visibility};

impl Emitter {
    pub fn memory_base(&mut self, array: Variable, store: bool) -> Result<(String, &'static str)> {
        match array.kind {
            VariableKind::SharedArray { .. } => Ok((self.shared_array(array)?, "shared")),
            VariableKind::LocalArray { .. } => Ok((self.local_array(array)?, "local")),
            VariableKind::ConstantArray { id, .. } => {
                if store { return Err(invalid("store to constant array")); }
                let (declared, _) = self.constant_arrays.get(&id)
                    .ok_or_else(|| invalid("constant array initializer missing"))?;
                if *declared != array { return Err(invalid("constant array declaration mismatch")); }
                Ok((format!("%constant_{id}"), "const"))
            }
            _ => {
                let buffer = self.buffer(array)?;
                if store && buffer.visibility != Visibility::ReadWrite { return Err(invalid("store to read-only buffer")); }
                Ok((format!("%buffer_{}", buffer.id), "global"))
            }
        }
    }

    fn local_array(&mut self, array: Variable) -> Result<String> {
        let VariableKind::LocalArray { id, length, unroll_factor } = array.kind else {
            return Err(invalid("expected thread-local array"));
        };
        let ty = Scalar::of(array.ty.with_vector_size(1))?;
        self.half_target(ty)?;
        let stride = array.ty.vector_size().checked_mul(ty.bytes())
            .ok_or_else(|| invalid("local array element size overflow"))?;
        let bytes = length.checked_mul(unroll_factor).and_then(|count| count.checked_mul(stride))
            .filter(|bytes| *bytes > 0).ok_or_else(|| invalid("empty or overflowing local array"))?;
        let alignment = stride.checked_next_power_of_two().ok_or_else(|| invalid("local array alignment overflow"))?;
        if let Some(previous) = self.local_arrays.get(&id) {
            if *previous != array { return Err(invalid("inconsistent local array declaration")); }
            return Ok(format!("%local_{id}"));
        }
        self.local_arrays.insert(id, array);
        self.declarations += &format!("    .local .align {alignment} .b8 local_{id}[{bytes}];\n    .reg .u64 %local_{id};\n");
        self.prologue += &format!("    mov.u64 %local_{id}, local_{id};\n");
        Ok(format!("%local_{id}"))
    }

    pub fn constant_array(&mut self, array: Variable, values: Vec<Variable>) -> Result<()> {
        let VariableKind::ConstantArray { id, length, unroll_factor } = array.kind else {
            return Err(invalid("constant initializer must name a constant array"));
        };
        if length == 0 || values.len() != length || unroll_factor != 1 {
            return Err(unsupported("empty, unrolled or inconsistent constant array"));
        }
        let ty = Scalar::of(array.ty.with_vector_size(1))?;
        self.half_target(ty)?;
        if let Some(previous) = self.constant_arrays.get(&id) {
            if previous != &(array, values) { return Err(invalid("inconsistent constant array initializer")); }
            return Ok(());
        }
        let width = array.ty.vector_size();
        let count = length.checked_mul(width).ok_or_else(|| invalid("constant array length overflow"))?;
        let stride = width.checked_mul(ty.bytes()).ok_or_else(|| invalid("constant array element size overflow"))?;
        let alignment = stride.checked_next_power_of_two().ok_or_else(|| invalid("constant array alignment overflow"))?;
        count.checked_mul(ty.bytes()).ok_or_else(|| invalid("constant array byte size overflow"))?;
        let mut constants = Vec::with_capacity(count);
        for value in &values {
            let VariableKind::Constant(value) = value.kind else { return Err(invalid("constant array requires literal elements")); };
            let value = value.cast_to(Type::scalar(array.elem_type()));
            let literal = ty.constant(value)?;
            constants.extend(std::iter::repeat_n(literal, width));
        }
        self.constant_arrays.insert(id, (array, values));
        self.module_declarations += &format!(".const .align {alignment} .{} constant_{id}[{count}] = {{{}}};\n", ty.memory_suffix(), constants.join(", "));
        self.declarations += &format!("    .reg .u64 %constant_{id};\n");
        self.prologue += &format!("    mov.u64 %constant_{id}, constant_{id};\n");
        Ok(())
    }
}
