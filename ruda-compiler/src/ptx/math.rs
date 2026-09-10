use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::{ConstantValue, Variable};

impl Emitter {
    pub fn reciprocal(&mut self, out: Variable, input: Variable) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if input.ty != out.ty || ty == Scalar::Pred {
            return Err(invalid("reciprocal requires matching numeric operands"));
        }
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        let one = if ty.float() {
            ty.constant(ConstantValue::Float(1.0))?
        } else {
            "1".to_owned()
        };
        if ty.half() {
            let numerator = self.reg(ty);
            self.line(format!("mov.b16 {numerator}, {one};"));
            return self.half_binary("div", ty, &destination, &numerator, &source);
        }
        let modifier = if ty.float() { ".rn" } else { "" };
        self.line(format!("div{modifier}.{} {destination}, {one}, {source};", ty.suffix()));
        Ok(())
    }

    pub fn square_root(&mut self, ty: Scalar, source: &str, inverse: bool) -> Result<String> {
        if ty.fp8() {
            let source = self.fp8_to_f32(ty, source)?;
            let result = self.square_root(Scalar::F32, &source, inverse)?;
            let destination = self.reg(ty);
            self.f32_to_fp8(ty, &destination, &result)?;
            return Ok(destination);
        }
        if !ty.float() {
            return Err(invalid("square root requires a floating-point operand"));
        }
        if inverse && !ty.half() {
            return Ok(self.inverse_root_register(ty, source));
        }
        self.half_target(ty)?;
        let result = self.reg(ty);
        if ty.half() {
            let source = self.half_to_f32(ty, source)?;
            let intermediate = self.reg(Scalar::F32);
            let operation = if inverse { "rsqrt" } else { "sqrt" };
            let flush = if ty == Scalar::F16 { ".ftz" } else { "" };
            self.line(format!("{operation}.approx{flush}.f32 {intermediate}, {source};"));
            self.line(format!("cvt.rn.{}.f32 {result}, {intermediate};", ty.suffix()));
        } else {
            self.line(format!("sqrt.rn.{} {result}, {source};", ty.suffix()));
        }
        Ok(result)
    }

    fn inverse_root_register(&mut self, ty: Scalar, source: &str) -> String {
        let input = if ty == Scalar::F32 {
            let input = self.reg(Scalar::F64);
            self.line(format!("cvt.f64.f32 {input}, {source};"));
            input
        } else {
            source.to_owned()
        };
        let root = self.reg(Scalar::F64);
        let inverse = self.reg(Scalar::F64);
        self.line(format!("sqrt.rn.f64 {root}, {input};"));
        self.line(format!("div.rn.f64 {inverse}, 0d3ff0000000000000, {root};"));

        if ty == Scalar::F64 {
            let predicate = self.reg(Scalar::Pred);
            let end = self.label();
            self.line(format!("setp.leu.f64 {predicate}, {input}, 0d0000000000000000;"));
            self.line(format!("@{predicate} bra {end};"));
            self.line(format!("setp.eq.f64 {predicate}, {input}, 0d7ff0000000000000;"));
            self.line(format!("@{predicate} bra {end};"));

            let negative_input = self.reg(Scalar::F64);
            let negative_inverse = self.reg(Scalar::F64);
            let product = self.reg(Scalar::F64);
            let product_error = self.reg(Scalar::F64);
            let residual = self.reg(Scalar::F64);
            let half_inverse = self.reg(Scalar::F64);
            self.line(format!("neg.f64 {negative_input}, {input};"));
            self.line(format!("neg.f64 {negative_inverse}, {inverse};"));
            // Form 1 - x*y*y without squaring y, which can overflow for tiny x.
            self.line(format!("mul.rn.f64 {product}, {input}, {inverse};"));
            self.line(format!("fma.rn.f64 {product_error}, {negative_input}, {inverse}, {product};"));
            self.line(format!("fma.rn.f64 {residual}, {negative_inverse}, {product}, 0d3ff0000000000000;"));
            self.line(format!("fma.rn.f64 {residual}, {inverse}, {product_error}, {residual};"));
            self.line(format!("mul.rn.f64 {half_inverse}, {inverse}, 0d3fe0000000000000;"));
            self.line(format!("fma.rn.f64 {inverse}, {half_inverse}, {residual}, {inverse};"));
            self.line(format!("{end}:"));
            inverse
        } else {
            let result = self.reg(Scalar::F32);
            self.line(format!("cvt.rn.f32.f64 {result}, {inverse};"));
            result
        }
    }

    pub fn emit_square_root(&mut self, out: Variable, input: Variable, inverse: bool) -> Result<()> {
        if out.ty != input.ty {
            return Err(invalid("square root operand types differ"));
        }
        let ty = Scalar::of(out.ty)?;
        let source = self.value(input)?;
        let result = self.square_root(ty, &source, inverse)?;
        let destination = self.destination(out)?;
        self.line(format!("mov.{} {destination}, {result};", ty.storage()));
        Ok(())
    }
}
