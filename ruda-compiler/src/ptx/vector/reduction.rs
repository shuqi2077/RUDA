use super::{Emitter, Result, Scalar, invalid, scalar};
use ruda_core::ir::{ConstantValue, Variable};

impl Emitter {
    pub fn vector_magnitude(&mut self, out: Variable, input: Variable) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if !ty.float() || input.ty.vector_size() == 0 || scalar(input).ty != out.ty {
            return Err(invalid("vector magnitude requires a matching floating scalar output"));
        }
        let values = self.vector_values(input)?;
        let sum = self.vector_square_sum(ty, &values)?;
        let magnitude = self.square_root(ty, &sum, false)?;
        let destination = self.destination(out)?;
        self.line(format!("mov.{} {destination}, {magnitude};", ty.storage()));
        Ok(())
    }

    pub fn vector_normalize(&mut self, out: Variable, input: Variable) -> Result<()> {
        let ty = Scalar::of(scalar(out).ty)?;
        if !ty.float() || input.ty.vector_size() == 0 || input.ty != out.ty {
            return Err(invalid("vector normalize requires matching floating operands"));
        }
        let values = self.vector_values(input)?;
        let sum = self.vector_square_sum(ty, &values)?;
        let inverse = self.square_root(ty, &sum, true)?;
        let destinations = self.vector_destination(out)?;
        for (destination, value) in destinations.iter().zip(values.iter()) {
            let normalized = self.numeric_binary(ty, "mul", value, &inverse)?;
            self.line(format!("mov.{} {destination}, {normalized};", ty.storage()));
        }
        Ok(())
    }

    fn vector_square_sum(&mut self, ty: Scalar, values: &[String]) -> Result<String> {
        let mut sum = self.reg(ty);
        let zero = ty.constant(ConstantValue::Float(0.0))?;
        self.line(format!("mov.{} {sum}, {zero};", ty.storage()));
        for value in values {
            let squared = self.numeric_binary(ty, "mul", value, value)?;
            sum = self.numeric_binary(ty, "add", &sum, &squared)?;
        }
        Ok(sum)
    }

    pub fn vector_reduce(
        &mut self,
        out: Variable,
        input: Variable,
        rhs: Option<Variable>,
    ) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if ty == Scalar::Pred || input.ty.vector_size() == 0 || scalar(input).ty != out.ty {
            return Err(invalid("vector reduction requires a matching numeric scalar output"));
        }
        if rhs.is_some_and(|rhs| rhs.ty != input.ty) {
            return Err(invalid("dot product operand types or vector widths differ"));
        }
        let lhs = self.vector_values(input)?;
        let rhs = rhs.map(|rhs| self.vector_values(rhs)).transpose()?;
        let mut result: Option<String> = None;
        for (lane, value) in lhs.iter().enumerate() {
            let term = match &rhs {
                Some(rhs) => self.numeric_binary(ty, "mul", value, &rhs[lane])?,
                None => value.clone(),
            };
            result = Some(match result {
                Some(accumulator) => self.numeric_binary(ty, "add", &accumulator, &term)?,
                None => term,
            });
        }
        let result = result.ok_or_else(|| invalid("empty vector reduction"))?;
        let destination = self.destination(out)?;
        self.line(format!("mov.{} {destination}, {result};", ty.storage()));
        Ok(())
    }
}
