use super::super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{
    Arithmetic, ConstantValue, ElemType, FloatKind, Instruction, Operation, OperationReflect,
    Type, Variable, VariableKind,
};

impl Emitter {
    pub(super) fn fp8_promoted_arithmetic(
        &mut self,
        operation: Arithmetic,
        out: Variable,
        ty: Scalar,
    ) -> Result<()> {
        let operation = Operation::Arithmetic(operation);
        let args = operation.args().ok_or_else(|| unsupported("FP8 arithmetic reflection"))?;
        let f32_type = Type::scalar(ElemType::Float(FloatKind::F32));
        let mut converted = Vec::with_capacity(args.len());
        let mut bindings = Vec::new();
        for input in args {
            let input_type = Scalar::of(input.ty)?;
            if !input_type.fp8() {
                if input_type.float() {
                    return Err(invalid("FP8 arithmetic floating operand types differ"));
                }
                converted.push(input);
                continue;
            }
            if input_type != ty {
                return Err(invalid("FP8 arithmetic operand formats differ"));
            }
            if let VariableKind::Constant(ConstantValue::Float(value)) = input.kind {
                let value = match ty {
                    Scalar::E4M3 => ruda_core::e4m3::from_f64(value).to_f32(),
                    Scalar::E5M2 => ruda_core::e5m2::from_f64(value).to_f32(),
                    _ => return Err(invalid("FP8 arithmetic output type")),
                };
                converted.push(Variable::new(
                    VariableKind::Constant(ConstantValue::Float(value as f64)),
                    f32_type,
                ));
            } else {
                let source = self.value(input)?;
                let source = self.fp8_to_f32(ty, &source)?;
                let variable = Variable::new(input.kind, f32_type);
                bindings.push((variable, source));
                converted.push(variable);
            }
        }
        let promoted = Operation::from_code_and_args(operation.op_code(), &converted)
            .ok_or_else(|| unsupported("FP8 arithmetic promotion"))?;
        let destination = self.destination(out)?;
        let promoted_out = Variable::new(out.kind, f32_type);
        let promoted_register = bindings.iter().rev()
            .find(|(variable, _)| *variable == promoted_out)
            .map(|(_, register)| register.clone())
            .unwrap_or_else(|| self.reg(Scalar::F32));
        bindings.push((promoted_out, promoted_register.clone()));
        let restore = bindings.into_iter().map(|(variable, register)| {
            (variable, self.registers.insert(variable, register))
        }).collect::<Vec<_>>();

        let result = self.instruction(Instruction::new(promoted, promoted_out));
        for (variable, previous) in restore.into_iter().rev() {
            if let Some(previous) = previous {
                self.registers.insert(variable, previous);
            } else {
                self.registers.remove(&variable);
            }
        }
        result?;
        self.f32_to_fp8(ty, &destination, &promoted_register)
    }
}
