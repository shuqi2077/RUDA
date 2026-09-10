use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{Arithmetic, Instruction, Operation, OperationReflect, Operator, Plane, Variable, VariableKind};

mod memory;
mod views;
mod indexing;
mod reduction;

fn scalar(variable: Variable) -> Variable {
    Variable::new(variable.kind, variable.ty.with_vector_size(1))
}

impl Emitter {
    pub fn vector_values(&mut self, variable: Variable) -> Result<Vec<String>> {
        let width = variable.ty.vector_size();
        if width == 1 { return Ok(vec![self.value(variable)?]); }
        if width == 0 { return Err(unsupported("semantic vector value")); }
        if let VariableKind::Constant(_) = variable.kind {
            return Ok(vec![self.value(scalar(variable))?; width]);
        }
        if !matches!(variable.kind, VariableKind::LocalMut { .. } | VariableKind::LocalConst { .. } | VariableKind::Versioned { .. }) {
            return Err(unsupported("vector value must be a local register or constant"));
        }
        if let Some(registers) = self.vector_registers.get(&variable) {
            return Ok(registers.clone());
        }
        let ty = Scalar::of(scalar(variable).ty)?;
        self.half_target(ty)?;
        let registers = (0..width).map(|_| self.reg(ty)).collect::<Vec<_>>();
        self.vector_registers.insert(variable, registers.clone());
        Ok(registers)
    }

    pub fn vector_destination(&mut self, variable: Variable) -> Result<Vec<String>> {
        if !matches!(variable.kind, VariableKind::LocalMut { .. } | VariableKind::LocalConst { .. } | VariableKind::Versioned { .. }) {
            return Err(invalid("invalid vector destination"));
        }
        self.vector_values(variable)
    }

    pub fn vector_instruction(&mut self, instruction: &Instruction) -> Result<bool> {
        let output = instruction.out;
        match &instruction.operation {
            Operation::Arithmetic(Arithmetic::Normalize(op)) => {
                let out = output.ok_or_else(|| invalid("vector normalize requires output"))?;
                self.vector_normalize(out, op.input)?;
                return Ok(true);
            }
            Operation::Arithmetic(Arithmetic::Magnitude(op)) => {
                let out = output.ok_or_else(|| invalid("vector magnitude requires output"))?;
                self.vector_magnitude(out, op.input)?;
                return Ok(true);
            }
            Operation::Arithmetic(Arithmetic::Dot(op)) => {
                let out = output.ok_or_else(|| invalid("dot product requires output"))?;
                self.vector_reduce(out, op.lhs, Some(op.rhs))?;
                return Ok(true);
            }
            Operation::Arithmetic(Arithmetic::VectorSum(op)) => {
                let out = output.ok_or_else(|| invalid("vector sum requires output"))?;
                self.vector_reduce(out, op.input, None)?;
                return Ok(true);
            }
            Operation::Operator(Operator::InitVector(op)) => {
                let out = output.ok_or_else(|| invalid("vector constructor requires output"))?;
                let destinations = self.vector_destination(out)?;
                if op.inputs.len() != destinations.len() {
                    return Err(invalid("vector constructor element count mismatch"));
                }
                let mut sources = Vec::with_capacity(op.inputs.len());
                for input in &op.inputs {
                    if input.ty != scalar(out).ty { return Err(invalid("vector constructor element type mismatch")); }
                    sources.push(self.value(*input)?);
                }
                let ty = Scalar::of(scalar(out).ty)?;
                for (destination, source) in destinations.iter().zip(sources) {
                    self.line(format!("mov.{} {destination}, {source};", ty.storage()));
                }
                return Ok(true);
            }
            Operation::Operator(Operator::Index(op) | Operator::UncheckedIndex(op)) => {
                let out = output.ok_or_else(|| invalid("index requires output"))?;
                if !op.list.is_array() {
                    self.vector_index(op.list, op.index, out, false)?;
                    return Ok(true);
                }
                if op.list.ty.vector_size() > 1 || out.ty.vector_size() > 1 || op.vector_size > 1 || op.unroll_factor > 1 {
                    let checked = matches!(&instruction.operation, Operation::Operator(Operator::Index(_)));
                    self.vector_memory(op.list, op.index, out, op.vector_size, op.unroll_factor, false, checked)?;
                    return Ok(true);
                }
            }
            Operation::Operator(Operator::IndexAssign(op) | Operator::UncheckedIndexAssign(op)) => {
                let out = output.ok_or_else(|| invalid("index assignment requires output"))?;
                if !out.is_array() {
                    self.vector_index(out, op.index, op.value, true)?;
                    return Ok(true);
                }
                if out.ty.vector_size() > 1 || op.value.ty.vector_size() > 1 || op.vector_size > 1 || op.unroll_factor > 1 {
                    let checked = matches!(&instruction.operation, Operation::Operator(Operator::IndexAssign(_)));
                    self.vector_memory(out, op.index, op.value, op.vector_size, op.unroll_factor, true, checked)?;
                    return Ok(true);
                }
            }
            Operation::Plane(Plane::Ballot(op)) if output.is_some_and(|out| out.ty.vector_size() > 1) => {
                let out = output.unwrap();
                if out.ty.vector_size() != 4 || Scalar::of(scalar(out).ty)? != Scalar::U32 || Scalar::of(op.input.ty)? != Scalar::Pred {
                    return Err(invalid("plane ballot requires a predicate input and four U32 outputs"));
                }
                let input = self.value(op.input)?;
                let registers = self.vector_destination(out)?;
                self.line(format!("vote.sync.ballot.b32 {}, {input}, 0xffffffff;", registers[0]));
                for register in &registers[1..] { self.line(format!("mov.u32 {register}, 0;")); }
                return Ok(true);
            }
            _ => {}
        }
        let Some(out) = output else { return Ok(false); };
        let Some(args) = instruction.operation.args() else { return Ok(false); };
        if out.ty.vector_size() <= 1 && args.iter().all(|arg| arg.ty.vector_size() <= 1) { return Ok(false); }
        if !matches!(&instruction.operation, Operation::Arithmetic(_) | Operation::Comparison(_) | Operation::Bitwise(_) | Operation::Copy(_) | Operation::Plane(_) |
            Operation::Operator(Operator::Cast(_) | Operator::Reinterpret(_) | Operator::Select(_) | Operator::And(_) | Operator::Or(_) | Operator::Not(_))) {
            return Ok(false);
        }
        let width = out.ty.vector_size();
        if width <= 1 || args.iter().any(|arg| arg.ty.vector_size() != 1 && arg.ty.vector_size() != width) {
            return Err(invalid("elementwise vector width mismatch"));
        }
        let destinations = self.vector_destination(out)?;
        let operands = args.iter().map(|arg| self.vector_values(*arg)).collect::<Result<Vec<_>>>()?;
        for lane in 0..width {
            let mut restore = Vec::new();
            let scalar_out = scalar(out);
            restore.push((scalar_out, self.registers.insert(scalar_out, destinations[lane].clone())));
            let scalar_args = args.iter().zip(&operands).map(|(arg, values)| {
                let variable = scalar(*arg);
                if arg.ty.vector_size() > 1 && !matches!(arg.kind, VariableKind::Constant(_)) {
                    restore.push((variable, self.registers.insert(variable, values[lane].clone())));
                }
                variable
            }).collect::<Vec<_>>();
            let operation = Operation::from_code_and_args(instruction.operation.op_code(), &scalar_args)
                .ok_or_else(|| unsupported("elementwise vector operation reflection"))?;
            let mut scalar_instruction = instruction.clone();
            scalar_instruction.out = Some(scalar_out);
            scalar_instruction.operation = operation;
            let result = self.instruction(scalar_instruction);
            for (variable, previous) in restore.into_iter().rev() {
                if let Some(previous) = previous { self.registers.insert(variable, previous); }
                else { self.registers.remove(&variable); }
            }
            result?;
        }
        Ok(true)
    }
}
