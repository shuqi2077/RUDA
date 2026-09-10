use super::{Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::ir::{BarrierLevel, BarrierOps, OpaqueType, SemanticType, StorageType, Type, Variable, VariableKind};

mod copy;
mod slice;
mod tma;
mod transaction;

fn level(barrier: Variable) -> Result<BarrierLevel> {
    match barrier.ty {
        Type::Scalar(StorageType::Opaque(OpaqueType::Barrier(level))) => Ok(level),
        _ => Err(invalid("expected a scalar barrier object")),
    }
}

impl Emitter {
    pub fn barrier(&mut self, operation: BarrierOps, output: Option<Variable>) -> Result<()> {
        if self.target.sm < 80 || self.target.version < (7, 0) {
            return Err(unsupported("hardware async barriers require SM >= 80 and PTX >= 7.0"));
        }
        match operation {
            BarrierOps::Declare { barrier } => { self.barrier_address(barrier)?; }
            BarrierOps::Init { barrier, is_elected, arrival_count } => {
                let address = self.barrier_address(barrier)?;
                let count = self.barrier_count(arrival_count)?;
                if level(barrier)? == BarrierLevel::Ruda {
                    if Scalar::of(is_elected.ty)? != Scalar::Pred {
                        return Err(invalid("barrier election must be a predicate"));
                    }
                    let elected = self.value(is_elected)?;
                    self.line(format!("@{elected} mbarrier.init.shared.b64 [{address}], {count};"));
                    self.line("bar.sync 0;");
                } else {
                    self.line(format!("mbarrier.init.shared.b64 [{address}], {count};"));
                }
            }
            BarrierOps::InitManual { barrier, arrival_count } => {
                let address = self.barrier_address(barrier)?;
                let count = self.barrier_count(arrival_count)?;
                self.line(format!("mbarrier.init.shared.b64 [{address}], {count};"));
            }
            BarrierOps::Arrive { barrier } => {
                let address = self.barrier_address(barrier)?;
                let token = self.barrier_token_destination(output.ok_or_else(|| invalid("barrier arrival requires token output"))?)?;
                self.line(format!("mbarrier.arrive.shared.b64 {token}, [{address}];"));
            }
            BarrierOps::Wait { barrier, token } => {
                let address = self.barrier_address(barrier)?;
                let token = self.barrier_token_value(token)?;
                self.wait_barrier(&address, &token, false);
            }
            BarrierOps::WaitParity { barrier, phase } => {
                if self.target.version < (7, 1) {
                    return Err(unsupported("barrier parity wait requires PTX >= 7.1"));
                }
                let address = self.barrier_address(barrier)?;
                let phase = match Scalar::of(phase.ty)? {
                    Scalar::Pred => {
                        let predicate = self.value(phase)?;
                        let phase = self.reg(Scalar::U32);
                        self.line(format!("selp.u32 {phase}, 1, 0, {predicate};"));
                        phase
                    }
                    Scalar::U32 => self.value(phase)?,
                    _ => return Err(invalid("barrier phase must be bool or U32")),
                };
                self.wait_barrier(&address, &phase, true);
            }
            BarrierOps::ArriveAndWait { barrier } => {
                let address = self.barrier_address(barrier)?;
                let token = self.reg(Scalar::U64);
                self.line(format!("mbarrier.arrive.shared.b64 {token}, [{address}];"));
                self.wait_barrier(&address, &token, false);
            }
            BarrierOps::CommitCopyAsync { barrier } => {
                let address = self.barrier_address(barrier)?;
                self.line(format!("cp.async.mbarrier.arrive.shared.b64 [{address}];"));
            }
            BarrierOps::CopyAsync { source, source_length, offset_source, offset_out, copy_length, checked } => {
                self.async_copy(source, output.ok_or_else(|| invalid("async copy destination missing"))?,
                    source_length, offset_source, offset_out, copy_length, checked)?;
            }
            BarrierOps::MemCopyAsync { barrier, source, source_length, offset_source, offset_out } => {
                self.async_copy_slice(barrier, source,
                    output.ok_or_else(|| invalid("async slice copy destination missing"))?,
                    source_length, offset_source, offset_out, false)?;
            }
            BarrierOps::MemCopyAsyncCooperative { barrier, source, source_length, offset_source, offset_out } => {
                self.async_copy_slice(barrier, source,
                    output.ok_or_else(|| invalid("cooperative copy destination missing"))?,
                    source_length, offset_source, offset_out, true)?;
            }
            BarrierOps::ExpectTx { barrier, transaction_count_update } => {
                self.barrier_expect_tx(barrier, transaction_count_update)?;
            }
            BarrierOps::ArriveTx { barrier, arrive_count_update, transaction_count_update } => {
                self.barrier_arrive_tx(barrier, arrive_count_update, transaction_count_update,
                    output.ok_or_else(|| invalid("transaction arrival requires token output"))?)?;
            }
            BarrierOps::MemCopyAsyncTx { barrier, source, source_length, offset_source, offset_out } => {
                self.async_copy_tx(barrier, source,
                    output.ok_or_else(|| invalid("transaction copy destination missing"))?,
                    source_length, offset_source, offset_out)?;
            }
            BarrierOps::TmaLoad { barrier, tensor_map, indices, offset_out } => {
                self.tma_load(barrier, tensor_map,
                    output.ok_or_else(|| invalid("TMA load destination missing"))?,
                    &indices, None, offset_out)?;
            }
            BarrierOps::TmaLoadIm2col { barrier, tensor_map, indices, offsets, offset_out } => {
                self.tma_load(barrier, tensor_map,
                    output.ok_or_else(|| invalid("TMA im2col destination missing"))?,
                    &indices, Some(&offsets), offset_out)?;
            }
        }
        Ok(())
    }

    fn barrier_address(&mut self, barrier: Variable) -> Result<String> {
        let base = match level(barrier)? {
            BarrierLevel::Ruda => {
                let VariableKind::Shared { id } = barrier.kind else {
                    return Err(invalid("ruda barrier must use shared storage"));
                };
                if let Some(previous) = self.shared_arrays.insert(id, barrier) {
                    if previous != barrier { return Err(invalid("inconsistent shared barrier declaration")); }
                }
                format!("%shared_{id}")
            }
            BarrierLevel::Unit => {
                if let Some((_, pointer)) = self.unit_barriers.iter().find(|(variable, _)| *variable == barrier) {
                    pointer.clone()
                } else {
                    let pointer = self.reg(Scalar::U64);
                    self.unit_barriers.push((barrier, pointer.clone()));
                    pointer
                }
            }
        };
        let address = self.reg(Scalar::U32);
        self.line(format!("cvt.u32.u64 {address}, {base};"));
        Ok(address)
    }

    fn barrier_count(&mut self, count: Variable) -> Result<String> {
        if !matches!(Scalar::of(count.ty)?, Scalar::U32 | Scalar::I32) {
            return Err(invalid("barrier arrival count must be a 32-bit integer"));
        }
        self.value(count)
    }

    pub fn barrier_token_destination(&mut self, token: Variable) -> Result<String> {
        if token.ty != Type::Semantic(SemanticType::BarrierToken)
            || !matches!(token.kind, VariableKind::BarrierToken { .. } | VariableKind::LocalMut { .. }
                | VariableKind::LocalConst { .. } | VariableKind::Versioned { .. })
        {
            return Err(invalid("invalid barrier token destination"));
        }
        if let Some(register) = self.registers.get(&token) { return Ok(register.clone()); }
        let register = self.reg(Scalar::U64);
        self.registers.insert(token, register.clone());
        Ok(register)
    }

    pub fn barrier_token_value(&self, token: Variable) -> Result<String> {
        if token.ty != Type::Semantic(SemanticType::BarrierToken) {
            return Err(invalid("expected barrier arrival token"));
        }
        self.registers.get(&token).cloned().ok_or_else(|| invalid("barrier token has no arrival"))
    }

    fn wait_barrier(&mut self, address: &str, state: &str, parity: bool) {
        let ready = self.reg(Scalar::Pred);
        let again = self.label();
        let parity = if parity { ".parity" } else { "" };
        self.line(format!("{again}:"));
        self.line(format!("mbarrier.test_wait{parity}.shared.b64 {ready}, [{address}], {state};"));
        self.line(format!("@!{ready} bra {again};"));
    }
}
