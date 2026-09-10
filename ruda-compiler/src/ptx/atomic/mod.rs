use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{AtomicOp, Instruction, Operation, Operator, StorageType, Type, Variable, VariableKind};

mod pointer;
mod reduction;
mod integer;

fn is_atomic(variable: Variable) -> bool {
    matches!(variable.ty, Type::Scalar(StorageType::Atomic(_)) | Type::Vector(StorageType::Atomic(_), _))
}

fn element(variable: Variable) -> Result<Scalar> {
    let Type::Scalar(StorageType::Atomic(elem)) = variable.ty else {
        return Err(unsupported("atomic pointer must have a scalar atomic element"));
    };
    let ty = Scalar::of(Type::scalar(elem))?;
    if matches!(ty, Scalar::I8 | Scalar::U8) { return Err(unsupported("8-bit integer atomics")); }
    if ty == Scalar::Pred { return Err(invalid("predicate atomic element")); }
    Ok(ty)
}

impl Emitter {
    pub fn atomic_instruction(&mut self, instruction: &Instruction) -> Result<bool> {
        let output = || instruction.out.ok_or_else(|| invalid("atomic instruction requires output"));
        match &instruction.operation {
            Operation::Atomic(op) => self.atomic(op, output()?)?,
            Operation::Operator(Operator::Index(op) | Operator::UncheckedIndex(op)) if is_atomic(op.list) => {
                if op.vector_size > 1 || op.unroll_factor != 1 {
                    return Err(unsupported("vectorized atomic pointer indexing"));
                }
                let checked = matches!(&instruction.operation, Operation::Operator(Operator::Index(_)));
                self.atomic_index(op.list, op.index, output()?, checked)?;
            }
            Operation::Copy(input) if is_atomic(*input) => {
                let out = output()?;
                if out.ty != input.ty { return Err(invalid("atomic pointer copy type mismatch")); }
                let source = self.atomic_pointer(*input)?;
                let destination = self.atomic_pointer_destination(out)?;
                self.line(format!("mov.u64 {destination}, {source};"));
            }
            Operation::Operator(Operator::Select(op)) if instruction.out.is_some_and(is_atomic) => {
                let out = output()?;
                if out.ty != op.then.ty || out.ty != op.or_else.ty || Scalar::of(op.cond.ty)? != Scalar::Pred {
                    return Err(invalid("atomic pointer select operand types"));
                }
                let condition = self.value(op.cond)?;
                let yes = self.atomic_pointer(op.then)?;
                let no = self.atomic_pointer(op.or_else)?;
                let destination = self.atomic_pointer_destination(out)?;
                self.line(format!("selp.u64 {destination}, {yes}, {no}, {condition};"));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn atomic(&mut self, operation: &AtomicOp, out: Variable) -> Result<()> {
        match operation {
            AtomicOp::Load(op) => {
                let ty = element(op.input)?;
                if Scalar::of(out.ty)? != ty { return Err(invalid("atomic load element type mismatch")); }
                let pointer = self.atomic_pointer(op.input)?;
                let destination = self.destination(out)?;
                if ty.narrow() {
                    return self.atomic_integer_compare_exchange(ty, &pointer, "0", "0", &destination);
                }
                let zero = self.reg(ty);
                self.line(format!("mov.{} {zero}, 0;", ty.bits()));
                self.atomic_reduce(ty, "add", &pointer, &zero, &destination)
            }
            AtomicOp::Store(op) => {
                let ty = element(out)?;
                if Scalar::of(op.input.ty)? != ty { return Err(invalid("atomic store element type mismatch")); }
                let pointer = self.atomic_pointer(out)?;
                let value = self.value(op.input)?;
                let previous = self.reg(ty);
                self.atomic_reduce(ty, "exch", &pointer, &value, &previous)
            }
            AtomicOp::CompareAndSwap(op) => {
                let ty = element(op.input)?;
                if [op.cmp.ty, op.val.ty, out.ty].iter().any(|value| Scalar::of(*value).ok() != Some(ty)) {
                    return Err(invalid("atomic compare-and-swap element type mismatch"));
                }
                self.atomic_target(ty, "cas")?;
                let pointer = self.atomic_pointer(op.input)?;
                let compare = self.value(op.cmp)?;
                let value = self.value(op.val)?;
                let destination = self.destination(out)?;
                if ty.narrow() {
                    return self.atomic_integer_compare_exchange(ty, &pointer, &compare, &value, &destination);
                }
                self.line(format!("atom.cas.{} {destination}, [{pointer}], {compare}, {value};", ty.bits()));
                Ok(())
            }
            AtomicOp::Swap(op) | AtomicOp::Add(op) | AtomicOp::Sub(op) |
            AtomicOp::Min(op) | AtomicOp::Max(op) | AtomicOp::And(op) |
            AtomicOp::Or(op) | AtomicOp::Xor(op) => {
                let ty = element(op.lhs)?;
                if Scalar::of(op.rhs.ty)? != ty || Scalar::of(out.ty)? != ty {
                    return Err(invalid("atomic reduction element type mismatch"));
                }
                let opcode = match operation {
                    AtomicOp::Swap(_) => "exch",
                    AtomicOp::Add(_) | AtomicOp::Sub(_) => "add",
                    AtomicOp::Min(_) => "min",
                    AtomicOp::Max(_) => "max",
                    AtomicOp::And(_) => "and",
                    AtomicOp::Or(_) => "or",
                    AtomicOp::Xor(_) => "xor",
                    _ => unreachable!(),
                };
                let pointer = self.atomic_pointer(op.lhs)?;
                let mut value = self.value(op.rhs)?;
                if matches!(operation, AtomicOp::Sub(_)) {
                    let negated = self.reg(ty);
                    if ty.integer() {
                        let suffix = if ty.bytes() == 8 { "u64" } else { "u32" };
                        self.line(format!("sub.{suffix} {negated}, 0, {value};"));
                    } else {
                        let mask = match ty.bytes() {
                            2 => "0x8000",
                            4 => "0x80000000",
                            _ => "0x8000000000000000",
                        };
                        self.line(format!("xor.{} {negated}, {value}, {mask};", ty.bits()));
                    }
                    value = negated;
                }
                let destination = self.destination(out)?;
                self.atomic_reduce(ty, opcode, &pointer, &value, &destination)
            }
        }
    }
}
