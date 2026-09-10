use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::{Arithmetic, FastMath, Instruction, Operation};

impl Emitter {
    pub(super) fn fast_math_instruction(&mut self, instruction: &Instruction) -> Result<bool> {
        let Operation::Arithmetic(operation) = &instruction.operation else { return Ok(false); };
        let Some(out) = instruction.out else { return Ok(false); };
        if out.ty.vector_size() != 1 || Scalar::of(out.ty)? != Scalar::F32 {
            return Ok(false);
        }
        let flags = instruction.modes.fp_math_mode;
        match operation {
            Arithmetic::Div(op) if flags.is_superset(
                FastMath::AllowReciprocal | FastMath::ReducedPrecision | FastMath::UnsignedZero | FastMath::NotInf,
            ) => {
                if op.lhs.ty != out.ty || op.rhs.ty != out.ty {
                    return Err(invalid("fast division operand types differ"));
                }
                let lhs = self.value(op.lhs)?;
                let rhs = self.value(op.rhs)?;
                let dst = self.destination(out)?;
                self.line(format!("div.approx.f32 {dst}, {lhs}, {rhs};"));
            }
            Arithmetic::Recip(op) if flags.is_superset(
                FastMath::AllowReciprocal | FastMath::ReducedPrecision | FastMath::UnsignedZero | FastMath::NotInf,
            ) => {
                if op.input.ty != out.ty {
                    return Err(invalid("fast reciprocal operand types differ"));
                }
                let input = self.value(op.input)?;
                let dst = self.destination(out)?;
                self.line(format!("rcp.approx.f32 {dst}, {input};"));
            }
            Arithmetic::Exp(op) if flags.is_superset(
                FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
            ) => {
                if op.input.ty != out.ty {
                    return Err(invalid("fast exponential operand types differ"));
                }
                let input = self.value(op.input)?;
                let scaled = self.reg(Scalar::F32);
                let dst = self.destination(out)?;
                self.line(format!("mul.rn.f32 {scaled}, {input}, 0f{:08x};", core::f32::consts::LOG2_E.to_bits()));
                self.line(format!("ex2.approx.f32 {dst}, {scaled};"));
            }
            Arithmetic::Log(op) if flags.is_superset(
                FastMath::ReducedPrecision | FastMath::NotNaN | FastMath::NotInf,
            ) => {
                if op.input.ty != out.ty {
                    return Err(invalid("fast logarithm operand types differ"));
                }
                let input = self.value(op.input)?;
                let log2 = self.reg(Scalar::F32);
                let dst = self.destination(out)?;
                self.line(format!("lg2.approx.f32 {log2}, {input};"));
                self.line(format!("mul.rn.f32 {dst}, {log2}, 0f{:08x};", core::f32::consts::LN_2.to_bits()));
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}
