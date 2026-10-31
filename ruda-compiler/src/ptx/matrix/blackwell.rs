use super::*;
use ruda_core::ir::Tcgen05Op;

fn supports_tcgen05(target: &crate::ptx::PtxTarget) -> bool {
    match target.sm {
        100 => target.version >= (8, 6),
        103 => target.version >= (8, 8),
        _ => false,
    }
}

#[cfg(test)]
mod target_tests {
    use super::*;
    use crate::ptx::PtxTarget;

    #[test]
    fn blackwell_target_versions() {
        for (sm, version, expected) in [
            (100, (8, 5), false), (100, (8, 6), true),
            (103, (8, 7), false), (103, (8, 8), true),
            (90, (8, 8), false), (120, (8, 8), false),
        ] {
            assert_eq!(supports_tcgen05(&PtxTarget { sm, version }), expected);
        }
    }
}

impl Emitter {
    pub(super) fn tcgen05(&mut self, out: Variable, kind: Tcgen05Op, args: Vec<Variable>) -> Result<()> {
        if !supports_tcgen05(&self.target) {
            return Err(unsupported("tcgen05 requires SM100a/PTX >= 8.6 or SM103a/PTX >= 8.8"));
        }
        self.architecture_specific = true;
        let arity = match kind {
            Tcgen05Op::Allocate { .. } | Tcgen05Op::Descriptor | Tcgen05Op::DescriptorStrided { .. } => 2,
            Tcgen05Op::Deallocate { .. } | Tcgen05Op::Commit | Tcgen05Op::Load { .. } => 1,
            Tcgen05Op::Mma { .. } | Tcgen05Op::MmaFp8 { .. } | Tcgen05Op::MmaKind { .. } => 4,
            Tcgen05Op::Relinquish | Tcgen05Op::Fence { .. } => 0,
        };
        if args.len() != arity { return Err(invalid("tcgen05 operand count mismatch")); }
        match kind {
            Tcgen05Op::Allocate { columns } => {
                if !(32..=512).contains(&columns) || !columns.is_power_of_two() {
                    return Err(invalid("TMEM allocation requires 32..512 power-of-two columns"));
                }
                let (address, space) = self.matrix_address(args[0], args[1], Some(1), true)?;
                if space != "shared" { return Err(invalid("TMEM allocation destination must be shared")); }
                let shared = self.reg(Scalar::U32);
                self.line(format!("cvt.u32.u64 {shared}, {address};"));
                self.line(format!("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [{shared}], {columns};"));
            }
            Tcgen05Op::Deallocate { columns } => {
                if !(32..=512).contains(&columns) || !columns.is_power_of_two() {
                    return Err(invalid("invalid TMEM deallocation size"));
                }
                let address = self.value(args[0])?;
                self.line(format!("tcgen05.dealloc.cta_group::1.sync.aligned.b32 {address}, {columns};"));
            }
            Tcgen05Op::Relinquish => self.line("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"),
            Tcgen05Op::DescriptorStrided { lbo, sbo } => {
                if !matches!(Scalar::memory_element(args[0].ty)?.bytes(), 1 | 2 | 4) {
                    return Err(invalid("SM100 descriptor requires 8/16/32-bit shared storage"));
                }
                if lbo % 16 != 0 || sbo % 16 != 0 || lbo >> 4 >= 1 << 14 || sbo >> 4 >= 1 << 14 {
                    return Err(invalid("SM100 descriptor offsets must be 16-byte multiples below 256 KiB"));
                }
                let (address, space) = self.matrix_address(args[0], args[1], Some(1), false)?;
                if space != "shared" { return Err(invalid("SM100 descriptor requires shared storage")); }
                let dst = self.destination(out)?;
                self.line(format!("shr.u64 {dst}, {address}, 4;"));
                self.line(format!("and.b64 {dst}, {dst}, 16383;"));
                let bits = (u64::from(lbo >> 4) << 16) | (u64::from(sbo >> 4) << 32) | (1u64 << 46) | (2u64 << 61);
                self.line(format!("or.b64 {dst}, {dst}, {bits};"));
                return Ok(());
            }
            Tcgen05Op::MmaKind { n, kind, a_mn, b_mn } => {
                if n < 8 || n > 256 || n % 8 != 0 || kind > 4 { return Err(invalid("invalid SM100 MMA shape/kind")); }
                if kind >= 3 && self.target.version < (8, 7) { return Err(unsupported("SM100 FP8 MMA requires PTX >= 8.7")); }
                let address = self.value(args[0])?;
                let a = self.value(args[1])?;
                let b = self.value(args[2])?;
                let accumulate = self.value(args[3])?;
                let destination = self.reg_b32();
                self.line(format!("mov.b32 {destination}, {address};"));
                let (format, family) = match kind {
                    0 => (0u32, "f16"), 1 => (1, "f16"), 2 => (2, "tf32"), 3 => (0, "f8f6f4"), _ => (1, "f8f6f4"),
                };
                let descriptor = (1u32 << 4) | (format << 7) | (format << 10) | ((a_mn as u32) << 15) | ((b_mn as u32) << 16)
                    | ((n / 8) << 17) | (8u32 << 24);
                let idesc = self.reg(Scalar::U32);
                let mask = self.reg(Scalar::U32);
                self.line(format!("mov.u32 {idesc}, {descriptor};"));
                self.line(format!("mov.u32 {mask}, 0;"));
                self.line(format!("tcgen05.mma.cta_group::1.kind::{family} [{destination}], {a}, {b}, {idesc}, {{{mask}, {mask}, {mask}, {mask}}}, {accumulate};"));
            }
            Tcgen05Op::Descriptor => {
                if !matches!(Scalar::memory_element(args[0].ty)?.bytes(), 1 | 2 | 4) {
                    return Err(invalid("SM100 descriptor requires 8/16/32-bit shared storage"));
                }
                let (address, space) = self.matrix_address(args[0], args[1], Some(1), false)?;
                if space != "shared" { return Err(invalid("SM100 descriptor requires shared storage")); }
                let dst = self.destination(out)?;
                self.line(format!("shr.u64 {dst}, {address}, 4;"));
                self.line(format!("and.b64 {dst}, {dst}, 16383;"));
                // K-major, 128-byte swizzle, 16-byte leading / 1024-byte stride.
                let bits = (1u64 << 16) | (64u64 << 32) | (1u64 << 46) | (2u64 << 61);
                self.line(format!("or.b64 {dst}, {dst}, {bits};"));
                return Ok(());
            }
            Tcgen05Op::Mma { n, bf16 } => {
                if n < 8 || n > 256 || n % 8 != 0 { return Err(invalid("invalid SM100 MMA N")); }
                let address = self.value(args[0])?;
                let a = self.value(args[1])?;
                let b = self.value(args[2])?;
                let accumulate = self.value(args[3])?;
                let destination = self.reg_b32();
                self.line(format!("mov.b32 {destination}, {address};"));
                let descriptor = (1u32 << 4) | ((bf16 as u32) << 7) | ((bf16 as u32) << 10)
                    | ((n / 8) << 17) | (8u32 << 24);
                let idesc = self.reg(Scalar::U32);
                let mask = self.reg(Scalar::U32);
                self.line(format!("mov.u32 {idesc}, {descriptor};"));
                self.line(format!("mov.u32 {mask}, 0;"));
                self.line(format!("tcgen05.mma.cta_group::1.kind::f16 [{destination}], {a}, {b}, {idesc}, {{{mask}, {mask}, {mask}, {mask}}}, {accumulate};"));
            }
            Tcgen05Op::MmaFp8 { n, e5m2_a, e5m2_b } => {
                if self.target.version < (8, 7) { return Err(unsupported("SM100 FP8 MMA requires PTX >= 8.7")); }
                if n < 8 || n > 256 || n % 8 != 0 { return Err(invalid("invalid SM100 FP8 MMA N")); }
                let address = self.value(args[0])?;
                let a = self.value(args[1])?;
                let b = self.value(args[2])?;
                let accumulate = self.value(args[3])?;
                let destination = self.reg_b32();
                self.line(format!("mov.b32 {destination}, {address};"));
                let descriptor = (1u32 << 4) | ((e5m2_a as u32) << 7) | ((e5m2_b as u32) << 10)
                    | ((n / 8) << 17) | (8u32 << 24);
                let idesc = self.reg(Scalar::U32);
                let mask = self.reg(Scalar::U32);
                self.line(format!("mov.u32 {idesc}, {descriptor};"));
                self.line(format!("mov.u32 {mask}, 0;"));
                self.line(format!("tcgen05.mma.cta_group::1.kind::f8f6f4 [{destination}], {a}, {b}, {idesc}, {{{mask}, {mask}, {mask}, {mask}}}, {accumulate};"));
            }
            Tcgen05Op::Commit => {
                let barrier = self.transaction_barrier(args[0])?;
                self.line(format!("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [{barrier}];"));
            }
            Tcgen05Op::Load { columns } => {
                if columns != 32 { return Err(unsupported("SM100 TMEM loads currently use x32")); }
                let address = self.value(args[0])?;
                let registers = (0..columns).map(|_| self.reg(Scalar::F32)).collect::<Vec<_>>();
                self.line(format!("tcgen05.ld.sync.aligned.32x32b.x32.b32 {{{}}}, [{address}];", registers.join(", ")));
                self.line("tcgen05.wait::ld.sync.aligned;");
                return self.matrix_write_registers(out, &registers);
            }
            Tcgen05Op::Fence { before } => {
                self.line(if before { "tcgen05.fence::before_thread_sync;" } else { "tcgen05.fence::after_thread_sync;" });
            }
        }
        let dst = self.destination(out)?;
        self.line(format!("mov.u32 {dst}, 0;"));
        Ok(())
    }
}
