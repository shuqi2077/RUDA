use super::*;

impl Emitter {
    pub(super) fn transaction_barrier(&mut self, barrier: Variable) -> Result<String> {
        if self.target.sm < 90 || self.target.version < (8, 0) {
            return Err(unsupported("barrier transactions require SM >= 90 and PTX >= 8.0"));
        }
        if level(barrier)? != BarrierLevel::Ruda {
            return Err(unsupported("transaction operations require a ruda-scoped shared barrier"));
        }
        self.barrier_address(barrier)
    }

    pub(super) fn barrier_expect_tx(&mut self, barrier: Variable, count: Variable) -> Result<()> {
        let address = self.transaction_barrier(barrier)?;
        let count = self.barrier_count(count)?;
        self.line(format!("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [{address}], {count};"));
        Ok(())
    }

    pub(super) fn barrier_arrive_tx(
        &mut self, barrier: Variable, arrivals: Variable, transactions: Variable, output: Variable,
    ) -> Result<()> {
        let address = self.transaction_barrier(barrier)?;
        let arrivals = self.barrier_count(arrivals)?;
        let transactions = self.barrier_count(transactions)?;
        let token = self.barrier_token_destination(output)?;
        let single = self.reg(Scalar::Pred);
        let multiple = self.label();
        let done = self.label();
        self.line(format!("setp.eq.u32 {single}, {arrivals}, 1;"));
        self.line(format!("@!{single} bra {multiple};"));
        self.line(format!("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 {token}, [{address}], {transactions};"));
        self.line(format!("bra {done};"));
        self.line(format!("{multiple}:"));
        self.line(format!("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [{address}], {transactions};"));
        self.line(format!("mbarrier.arrive.release.cta.shared::cta.b64 {token}, [{address}], {arrivals};"));
        self.line(format!("{done}:"));
        Ok(())
    }

    pub(super) fn async_copy_tx(
        &mut self, barrier: Variable, source: Variable, destination: Variable,
        source_length: Variable, offset_source: Variable, offset_out: Variable,
    ) -> Result<()> {
        let barrier = self.transaction_barrier(barrier)?;
        let scalar = Scalar::memory_element(source.ty)?;
        if scalar != Scalar::memory_element(destination.ty)? {
            return Err(invalid("transaction copy element types differ"));
        }
        let (source_base, source_space) = self.memory_base(source, false)?;
        let (destination_base, destination_space) = self.memory_base(destination, true)?;
        if source_space != "global" || destination_space != "shared" {
            return Err(unsupported("transaction copy requires global source and shared destination"));
        }
        let source_stride = source.ty.vector_size().checked_mul(scalar.bytes())
            .ok_or_else(|| invalid("transaction copy source stride overflow"))?;
        let destination_stride = destination.ty.vector_size().checked_mul(scalar.bytes())
            .ok_or_else(|| invalid("transaction copy destination stride overflow"))?;
        let source_address = self.async_address(&source_base, offset_source, source_stride)?;
        let destination_address = self.async_address(&destination_base, offset_out, destination_stride)?;
        let shared = self.reg(Scalar::U32);
        self.line(format!("cvt.u32.u64 {shared}, {destination_address};"));
        let length_type = Scalar::of(source_length.ty)?;
        if !matches!(length_type, Scalar::U32 | Scalar::U64) {
            return Err(invalid("transaction copy source length must be unsigned"));
        }
        let length = self.value(source_length)?;
        let bytes = self.reg(length_type);
        self.line(format!("mul.lo.{} {bytes}, {length}, {source_stride};", length_type.suffix()));
        let bytes = if length_type == Scalar::U64 {
            let size = self.reg(Scalar::U32);
            self.line(format!("cvt.u32.u64 {size}, {bytes};"));
            size
        } else { bytes };
        let shared_scope = if self.target.version >= (8, 6) { "cta" } else { "cluster" };
        self.line(format!("cp.async.bulk.shared::{shared_scope}.global.mbarrier::complete_tx::bytes [{shared}], [{source_address}], {bytes}, [{barrier}];"));
        Ok(())
    }
}
