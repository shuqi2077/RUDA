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
        let rank = self.reg(Scalar::U32);
        let count = self.reg(Scalar::U32);
        let lower_lanes = self.reg(Scalar::U32);
        let accumulator = self.reg(ty);
        self.line(format!("mov.u32 {lower_lanes}, %lanemask_lt;"));
        self.line(format!("and.b32 {lower_lanes}, {lower_lanes}, {mask};"));
        self.line(format!("popc.b32 {rank}, {lower_lanes};"));
        self.line(format!("popc.b32 {count}, {mask};"));
        self.line(format!("mov.{} {accumulator}, {source};", ty.storage()));

        for offset in [1, 2, 4, 8, 16] {
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
            let partner_lane = self.plane_lane_at_rank(mask, &partner);
            let shuffled = self.shuffle_register_masked(ty, &accumulator, &partner_lane, "idx", mask)?;
            let skip = self.label();
            self.line(format!("@!{valid} bra {skip};"));
            let combined = self.numeric_binary(ty, operation, &accumulator, &shuffled)?;
            self.line(format!("mov.{} {accumulator}, {combined};", ty.storage()));
            self.line(format!("{skip}:"));
        }

        if exclusive {
            let has_previous = self.reg(Scalar::Pred);
            let previous = self.reg(Scalar::U32);
            self.line(format!("setp.ne.u32 {has_previous}, {rank}, 0;"));
            self.line(format!("sub.u32 {previous}, {rank}, 1;"));
            self.line(format!("selp.u32 {previous}, {previous}, {rank}, {has_previous};"));
            let previous_lane = self.plane_lane_at_rank(mask, &previous);
            let shuffled = self.shuffle_register_masked(ty, &accumulator, &previous_lane, "idx", mask)?;
            let identity = if operation == "add" { 0 } else { 1 };
            let identity = if ty.float() {
                ty.constant(ConstantValue::Float(identity as f64))?
            } else {
                identity.to_string()
            };
            self.line(format!("selp.{} {accumulator}, {shuffled}, {identity}, {has_previous};", ty.storage()));
        } else if !scan {
            let first_lane = self.reg(Scalar::U32);
            self.line(format!("brev.b32 {first_lane}, {mask};"));
            self.line(format!("clz.b32 {first_lane}, {first_lane};"));
            let result = self.shuffle_register_masked(ty, &accumulator, &first_lane, "idx", mask)?;
            self.line(format!("mov.{} {accumulator}, {result};", ty.storage()));
        }
        self.line(format!("mov.{} {destination}, {accumulator};", ty.storage()));
        Ok(())
    }
}
