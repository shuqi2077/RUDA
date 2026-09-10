use super::*;
use ruda_core::ir::{Builtin, ElemType, UIntKind};

impl Emitter {
    pub(super) fn async_copy_slice(
        &mut self, barrier: Variable, source: Variable, destination: Variable,
        source_length: Variable, offset_source: Variable, offset_out: Variable,
        cooperative: bool,
    ) -> Result<()> {
        if cooperative && level(barrier)? != BarrierLevel::Ruda {
            return Err(invalid("cooperative copy requires a ruda-scoped barrier"));
        }
        let scalar = Scalar::memory_element(source.ty)?;
        if scalar != Scalar::memory_element(destination.ty)? {
            return Err(invalid("async slice copy element types differ"));
        }
        let (source_base, source_space) = self.memory_base(source, false)?;
        let (destination_base, destination_space) = self.memory_base(destination, true)?;
        let source_stride = source.ty.vector_size().checked_mul(scalar.bytes())
            .ok_or_else(|| invalid("async slice source stride overflow"))?;
        let destination_stride = destination.ty.vector_size().checked_mul(scalar.bytes())
            .ok_or_else(|| invalid("async slice destination stride overflow"))?;
        let source = self.async_address(&source_base, offset_source, source_stride)?;
        let destination = self.async_address(&destination_base, offset_out, destination_stride)?;
        let length_type = Scalar::of(source_length.ty)?;
        if !matches!(length_type, Scalar::U32 | Scalar::U64) {
            return Err(invalid("async slice length must be unsigned"));
        }
        let length = self.value(source_length)?;
        let bytes = self.reg(Scalar::U64);
        let multiply = if length_type == Scalar::U32 { "mul.wide.u32" } else { "mul.lo.u64" };
        self.line(format!("{multiply} {bytes}, {length}, {source_stride};"));
        let (rank, group_size) = if cooperative {
            let ty = StorageType::Scalar(ElemType::UInt(UIntKind::U64));
            (self.value(Variable::builtin(Builtin::UnitPos, ty))?,
                self.value(Variable::builtin(Builtin::RudaDim, ty))?)
        } else { ("0".into(), "1".into()) };
        let barrier = self.barrier_address(barrier)?;
        let done = self.label();
        if source_space == "global" && destination_space == "shared" {
            let alignment = self.reg(Scalar::U64);
            let remainder = self.reg(Scalar::U64);
            let aligned = self.reg(Scalar::Pred);
            self.line(format!("or.b64 {alignment}, {source}, {destination};"));
            self.line(format!("or.b64 {alignment}, {alignment}, {bytes};"));
            for width in [16, 8, 4] {
                let next = self.label();
                self.line(format!("and.b64 {remainder}, {alignment}, {};", width - 1));
                self.line(format!("setp.eq.u64 {aligned}, {remainder}, 0;"));
                self.line(format!("@!{aligned} bra {next};"));
                self.slice_copy_loop(&source, source_space, &destination, destination_space,
                    &bytes, &rank, &group_size, width);
                self.line(format!("cp.async.mbarrier.arrive.shared.b64 [{barrier}];"));
                self.line(format!("bra {done};"));
                self.line(format!("{next}:"));
            }
        }
        self.slice_copy_loop(&source, source_space, &destination, destination_space,
            &bytes, &rank, &group_size, 1);
        self.line(format!("{done}:"));
        Ok(())
    }

    fn slice_copy_loop(
        &mut self, source: &str, source_space: &str, destination: &str,
        destination_space: &str, bytes: &str, rank: &str, group_size: &str, width: u32,
    ) {
        let offset = self.reg(Scalar::U64);
        let stride = self.reg(Scalar::U64);
        let source_address = self.reg(Scalar::U64);
        let destination_address = self.reg(Scalar::U64);
        let finished = self.reg(Scalar::Pred);
        let again = self.label();
        let done = self.label();
        self.line(format!("mov.u64 {offset}, {rank};"));
        self.line(format!("mul.lo.u64 {offset}, {offset}, {width};"));
        self.line(format!("mov.u64 {stride}, {group_size};"));
        self.line(format!("mul.lo.u64 {stride}, {stride}, {width};"));
        self.line(format!("{again}:"));
        self.line(format!("setp.ge.u64 {finished}, {offset}, {bytes};"));
        self.line(format!("@{finished} bra {done};"));
        self.line(format!("add.u64 {source_address}, {source}, {offset};"));
        self.line(format!("add.u64 {destination_address}, {destination}, {offset};"));
        if width == 1 {
            let byte = self.reg(Scalar::U32);
            self.line(format!("ld.{source_space}.u8 {byte}, [{source_address}];"));
            self.line(format!("st.{destination_space}.u8 [{destination_address}], {byte};"));
        } else {
            let shared = self.reg(Scalar::U32);
            self.line(format!("cvt.u32.u64 {shared}, {destination_address};"));
            let cache = if width == 16 { "cg" } else { "ca" };
            self.line(format!("cp.async.{cache}.shared.global [{shared}], [{source_address}], {width};"));
        }
        self.line(format!("add.u64 {offset}, {offset}, {stride};"));
        self.line(format!("bra {again};"));
        self.line(format!("{done}:"));
    }
}
