use super::*;
use ruda_core::ir::TmaOps;

impl Emitter {
    pub fn tma(&mut self, operation: TmaOps, output: Option<Variable>) -> Result<()> {
        if self.target.sm < 90 || self.target.version < (8, 0) {
            return Err(unsupported("TMA requires SM >= 90 and PTX >= 8.0"));
        }
        match operation {
            TmaOps::TmaStore { source, coordinates, offset_source } => {
                let map = output.ok_or_else(|| invalid("TMA store tensor map missing"))?;
                let descriptor = self.tensor_map_address(map, source)?;
                let rank = coordinates.len();
                let coordinates = self.tma_coordinates(&coordinates)?;
                let source = self.tma_shared_address(source, offset_source, false)?;
                self.line(format!("cp.async.bulk.tensor.{rank}d.global.shared::cta.bulk_group [{descriptor}, {{{coordinates}}}], [{source}];"));
            }
            TmaOps::CommitGroup => self.line("cp.async.bulk.commit_group;"),
            TmaOps::WaitGroup { max_pending } => {
                self.line(format!("cp.async.bulk.wait_group {max_pending};"));
            }
            TmaOps::WaitGroupRead { max_pending } => {
                self.line(format!("cp.async.bulk.wait_group.read {max_pending};"));
            }
        }
        Ok(())
    }

    pub(super) fn tma_load(
        &mut self, barrier: Variable, map: Variable, destination: Variable,
        coordinates: &[Variable], offsets: Option<&[Variable]>, offset_out: Variable,
    ) -> Result<()> {
        let barrier = self.transaction_barrier(barrier)?;
        let descriptor = self.tensor_map_address(map, destination)?;
        let rank = coordinates.len();
        let coordinates = self.tma_coordinates(coordinates)?;
        let destination = self.tma_shared_address(destination, offset_out, true)?;
        let (mode, offsets) = if let Some(offsets) = offsets {
            if !(3..=5).contains(&rank) || offsets.len() != rank - 2 {
                return Err(invalid("TMA im2col requires rank 3..=5 and rank-2 spatial offsets"));
            }
            let mut registers = Vec::with_capacity(offsets.len());
            for offset in offsets.iter().rev() {
                if Scalar::of(offset.ty)? != Scalar::U16 {
                    return Err(invalid("TMA im2col offsets must be U16"));
                }
                let value = self.value(*offset)?;
                let narrow = self.reg_b16();
                self.line(format!("cvt.u16.u32 {narrow}, {value};"));
                registers.push(narrow);
            }
            (".im2col", format!(", {{{}}}", registers.join(", ")))
        } else { ("", String::new()) };
        let scope = if self.target.version >= (8, 6) { "cta" } else { "cluster" };
        self.line(format!("cp.async.bulk.tensor.{rank}d.shared::{scope}.global{mode}.mbarrier::complete_tx::bytes [{destination}], [{descriptor}, {{{coordinates}}}], [{barrier}]{offsets};"));
        Ok(())
    }

    fn tensor_map_address(&self, map: Variable, shared: Variable) -> Result<String> {
        let id = match map.kind {
            VariableKind::TensorMapInput(id) | VariableKind::TensorMapOutput(id) => id,
            _ => return Err(invalid("TMA requires a tensor map argument")),
        };
        let binding = self.tensor_maps.iter().find(|binding| binding.id == id)
            .ok_or_else(|| invalid("TMA tensor map argument not declared"))?;
        if binding.ty != map.ty || Scalar::memory_element(map.ty)? != Scalar::memory_element(shared.ty)? {
            return Err(invalid("TMA tensor map and shared element types differ"));
        }
        Ok(format!("%tensor_map_{id}"))
    }

    fn tma_coordinates(&mut self, coordinates: &[Variable]) -> Result<String> {
        if !(1..=5).contains(&coordinates.len()) {
            return Err(invalid("TMA tensor rank must be 1..=5"));
        }
        let mut registers = Vec::with_capacity(coordinates.len());
        for coordinate in coordinates.iter().rev() {
            if Scalar::of(coordinate.ty)? != Scalar::I32 {
                return Err(invalid("TMA coordinates must be I32"));
            }
            let value = self.value(*coordinate)?;
            let register = self.reg(Scalar::I32);
            self.line(format!("mov.b32 {register}, {value};"));
            registers.push(register);
        }
        Ok(registers.join(", "))
    }

    fn tma_shared_address(&mut self, variable: Variable, offset: Variable, store: bool) -> Result<String> {
        let (base, space) = self.memory_base(variable, store)?;
        if space != "shared" {
            return Err(invalid("TMA tile must use shared memory"));
        }
        let stride = variable.ty.vector_size().checked_mul(Scalar::memory_element(variable.ty)?.bytes())
            .ok_or_else(|| invalid("TMA shared element stride overflow"))?;
        let address = self.async_address(&base, offset, stride)?;
        let shared = self.reg(Scalar::U32);
        self.line(format!("cvt.u32.u64 {shared}, {address};"));
        Ok(shared)
    }
}
