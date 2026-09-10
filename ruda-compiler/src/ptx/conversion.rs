use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{ConstantValue, Variable};

impl Emitter {
    pub fn tf32_target(&self) -> Result<()> {
        if self.target.sm < 80 || self.target.version < (7, 0) {
            return Err(unsupported("TF32 conversion/MMA requires SM >= 80 and PTX >= 7.0"));
        }
        Ok(())
    }

    pub fn tf32_cast(&mut self, out: Variable, input: Variable) -> Result<()> {
        self.tf32_target()?;
        if !Scalar::is_tf32(out.ty) || out.ty.vector_size() != 1 {
            return Err(invalid("expected scalar TF32 conversion output"));
        }
        let from = Scalar::of(input.ty)?;
        let mut source = self.value(input)?;
        if from.half() { source = self.half_to_f32(from, &source)?; }
        else if from == Scalar::Pred {
            let converted = self.reg(Scalar::F32);
            self.line(format!("selp.f32 {converted}, 0f3F800000, 0f00000000, {source};"));
            source = converted;
        } else if from != Scalar::F32 {
            let converted = self.reg(Scalar::F32);
            self.line(format!("cvt.rn.f32.{} {converted}, {source};", from.suffix()));
            source = converted;
        }
        let bits = self.reg(Scalar::U32);
        self.line(format!("cvt.rna.tf32.f32 {bits}, {source};"));
        let destination = self.destination(out)?;
        self.line(format!("mov.b32 {destination}, {bits};"));
        Ok(())
    }

    pub fn reinterpret(&mut self, out: Variable, input: Variable) -> Result<()> {
        let to = Scalar::of(out.ty)?;
        let from = Scalar::of(input.ty)?;
        if to == Scalar::Pred || from == Scalar::Pred || to.bytes() != from.bytes() {
            return Err(invalid("reinterpret requires equal-width non-predicate storage"));
        }
        self.half_target(to)?;
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        if to.narrow() && from.half() {
            self.half_target(from)?;
            self.line(format!("cvt.u32.u16 {destination}, {source};"));
            return Ok(());
        }
        if to.half() && from.narrow() {
            self.line(format!("cvt.u16.u32 {destination}, {source};"));
            return Ok(());
        }
        self.line(format!("mov.{} {destination}, {source};", to.bits()));
        Ok(())
    }

    pub fn predicate_cast(&mut self, out: Variable, input: Variable, to: Scalar, from: Scalar) -> Result<()> {
        let mut source = self.value(input)?;
        let destination = self.destination(out)?;
        if from == Scalar::Pred {
            self.half_target(to)?;
            let one = if to.float() { to.constant(ConstantValue::Float(1.0))? } else { "1".into() };
            let zero = if to.float() { to.constant(ConstantValue::Float(0.0))? } else { "0".into() };
            self.line(format!("selp.{} {destination}, {one}, {zero}, {source};", to.storage()));
        } else {
            let compared = if from.half() {
                source = self.half_to_f32(from, &source)?;
                Scalar::F32
            } else {
                from
            };
            let comparison = if compared.float() { "neu" } else { "ne" };
            let zero = if compared.float() { compared.constant(ConstantValue::Float(0.0))? } else { "0".into() };
            self.line(format!("setp.{comparison}.{} {destination}, {source}, {zero};", compared.suffix()));
        }
        Ok(())
    }

    pub fn integer_to_half(
        &mut self,
        out: Variable,
        input: Variable,
        to: Scalar,
        from: Scalar,
    ) -> Result<()> {
        if !to.half() || !from.integer() {
            return Err(invalid("integer-to-half conversion requires integer and half endpoints"));
        }
        self.half_target(to)?;
        let source = self.value(input)?;
        let destination = self.destination(out)?;
        let rounded = self.reg(Scalar::F32);
        if from.narrow() {
            self.line(format!("cvt.rn.f32.{} {rounded}, {source};", from.suffix()));
            self.line(format!("cvt.rn.{}.f32 {destination}, {rounded};", to.suffix()));
            return Ok(());
        }
        let bits = self.reg(Scalar::U32);
        let recovered = self.reg(from);
        let inexact = self.reg(Scalar::Pred);

        // RZ keeps the intermediate inside the source integer range. Its exact
        // integer round trip detects discarded bits even at I64/U64 endpoints.
        // Round-to-odd at F32 precision then permits one RN half conversion.
        self.line(format!("cvt.rz.f32.{} {rounded}, {source};", from.suffix()));
        self.line(format!("cvt.rzi.{}.f32 {recovered}, {rounded};", from.suffix()));
        self.line(format!("setp.ne.{} {inexact}, {recovered}, {source};", from.suffix()));
        self.line(format!("mov.b32 {bits}, {rounded};"));
        self.line(format!("@{inexact} or.b32 {bits}, {bits}, 1;"));
        self.line(format!("mov.b32 {rounded}, {bits};"));
        self.line(format!("cvt.rn.{}.f32 {destination}, {rounded};", to.suffix()));
        Ok(())
    }

    pub fn to_float(&mut self, out: Variable, input: Variable, to: Scalar, from: Scalar) -> Result<()> {
        let mut source = self.value(input)?;
        let destination = self.destination(out)?;
        let from = if from.half() {
            source = self.half_to_f32(from, &source)?;
            Scalar::F32
        } else {
            from
        };
        if from == to {
            self.line(format!("mov.{} {destination}, {source};", to.suffix()));
        } else if from == Scalar::F32 && to == Scalar::F64 {
            self.line(format!("cvt.f64.f32 {destination}, {source};"));
        } else {
            self.line(format!("cvt.rn.{}.{} {destination}, {source};", to.suffix(), from.suffix()));
        }
        Ok(())
    }
}
