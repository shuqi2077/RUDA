use super::*;
use ruda_core::ir::ConstantValue;

impl Emitter {
    pub(super) fn plane_reduce_active(
        &mut self,
        ty: Scalar,
        source: &str,
        destination: &str,
        mask: &str,
        operation: &str,
        scan: bool,
        exclusive: bool,
    ) -> Result<()> {
        // Test once per collective, not once per shuffle. In a low prefix,
        // logical ranks are already physical lane IDs.
        let bits = self.reg(Scalar::U32);
        let count = self.reg(Scalar::U32);
        let sparse = self.reg(Scalar::Pred);
        let holes = self.label();
        let done = self.label();
        self.line(format!("popc.b32 {count}, {mask};"));
        self.line(format!("add.u32 {bits}, {mask}, 1;"));
        self.line(format!("and.b32 {bits}, {bits}, {mask};"));
        self.line(format!("setp.ne.u32 {sparse}, {bits}, 0;"));
        self.line(format!("@{sparse} bra {holes};"));
        self.plane_reduce_ranks(ty, source, destination, mask, &count, operation, scan, exclusive, true)?;
        self.line(format!("bra {done};"));
        self.line(format!("{holes}:"));
        self.plane_reduce_ranks(ty, source, destination, mask, &count, operation, scan, exclusive, false)?;
        self.line(format!("{done}:"));
        Ok(())
    }

    pub(super) fn plane_reduce_ranks(
        &mut self, ty: Scalar, source: &str, destination: &str, mask: &str,
        count: &str, operation: &str, scan: bool, exclusive: bool, prefix: bool,
    ) -> Result<()> {
        let rank = self.reg(Scalar::U32);
        let accumulator = self.reg(ty);
        if prefix {
            self.line(format!("mov.u32 {rank}, %laneid;"));
        } else {
            self.line(format!("mov.u32 {rank}, %lanemask_lt;"));
            self.line(format!("and.b32 {rank}, {rank}, {mask};"));
            self.line(format!("popc.b32 {rank}, {rank};"));
        }
        self.line(format!("mov.{} {accumulator}, {source};", ty.storage()));
        for offset in [1, 2, 4, 8, 16] {
            if offset >= self.plane_width { break; }
            let partner = self.reg(Scalar::U32);
            let valid = self.reg(Scalar::Pred);
            if scan {
                self.line(format!("setp.ge.u32 {valid}, {rank}, {offset};"));
                self.line(format!("sub.u32 {partner}, {rank}, {offset};"));
            } else {
                self.line(format!("xor.b32 {partner}, {rank}, {offset};"));
                self.line(format!("setp.lt.u32 {valid}, {partner}, {count};"));
            }
            self.line(format!("selp.u32 {partner}, {partner}, {rank}, {valid};"));
            let partner_lane = if prefix { partner } else { self.plane_lane_at_rank(mask, &partner) };
            let shuffled = self.shuffle_register_masked(ty, &accumulator, &partner_lane, "idx", mask)?;
            let combined = self.numeric_binary(ty, operation, &accumulator, &shuffled)?;
            self.line(format!("@{valid} mov.{} {accumulator}, {combined};", ty.storage()));
        }

        if exclusive {
            let has_previous = self.reg(Scalar::Pred);
            let previous = self.reg(Scalar::U32);
            self.line(format!("setp.ne.u32 {has_previous}, {rank}, 0;"));
            self.line(format!("sub.u32 {previous}, {rank}, 1;"));
            self.line(format!("selp.u32 {previous}, {previous}, {rank}, {has_previous};"));
            let previous_lane = if prefix { previous } else { self.plane_lane_at_rank(mask, &previous) };
            let shuffled = self.shuffle_register_masked(ty, &accumulator, &previous_lane, "idx", mask)?;
            let identity = if operation == "add" { 0 } else { 1 };
            let identity = if ty.float() {
                ty.constant(ConstantValue::Float(identity as f64))?
            } else {
                identity.to_string()
            };
            self.line(format!("selp.{} {accumulator}, {shuffled}, {identity}, {has_previous};", ty.storage()));
        } else if !scan {
            let first_lane = if prefix { "0".to_owned() } else {
                let first = self.reg(Scalar::U32);
                self.line(format!("brev.b32 {first}, {mask};"));
                self.line(format!("clz.b32 {first}, {first};"));
                first
            };
            let result = self.shuffle_register_masked(ty, &accumulator, &first_lane, "idx", mask)?;
            self.line(format!("mov.{} {accumulator}, {result};", ty.storage()));
        }
        self.line(format!("mov.{} {destination}, {accumulator};", ty.storage()));
        Ok(())
    }
}
