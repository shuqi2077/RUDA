use super::*;
use ruda_core::ir::ConstantValue;

impl Emitter {
    pub(super) fn plane_reduce(&mut self, input: Variable, out: Variable, operation: &str, scan: bool, exclusive: bool) -> Result<()> {
        let ty = Scalar::of(out.ty)?;
        if input.ty != out.ty || ty == Scalar::Pred {
            return Err(invalid("plane reduction requires matching numeric input/output types"));
        }
        let source = self.value(input)?;
        let mask = self.plane_active_mask();
        let full = self.reg(Scalar::Pred);
        let partial = self.label();
        let done = self.label();
        self.line(format!("setp.eq.u32 {full}, {mask}, 0xffffffff;"));
        self.line(format!("@!{full} bra {partial};"));
        let mut accumulator = self.reg(ty);
        self.line(format!("mov.{} {accumulator}, {source};", ty.storage()));
        let lane = if scan {
            let lane = self.reg(Scalar::U32);
            self.line(format!("mov.u32 {lane}, %laneid;"));
            Some(lane)
        } else { None };
        // Preserve the C++ backend's ascending butterfly / inclusive-scan order.
        for offset in [1, 2, 4, 8, 16] {
            if offset >= self.plane_width { break; }
            let shuffled = self.shuffle_register(ty, &accumulator, &offset.to_string(), if scan { "up" } else { "bfly" })?;
            if let Some(lane) = &lane {
                let valid = self.reg(Scalar::Pred);
                let skip = self.label();
                self.line(format!("setp.ge.u32 {valid}, {lane}, {offset};"));
                self.line(format!("@!{valid} bra {skip};"));
                let combined = self.numeric_binary(ty, operation, &accumulator, &shuffled)?;
                self.line(format!("mov.{} {accumulator}, {combined};", ty.storage()));
                self.line(format!("{skip}:"));
            } else {
                accumulator = self.numeric_binary(ty, operation, &accumulator, &shuffled)?;
            }
        }
        if exclusive {
            accumulator = self.shuffle_register(ty, &accumulator, "1", "up")?;
            let first = self.reg(Scalar::Pred);
            let lane = lane.expect("exclusive scan has a lane register");
            self.line(format!("setp.eq.u32 {first}, {lane}, 0;"));
            let identity = if operation == "add" { 0 } else { 1 };
            let identity = if ty.float() { ty.constant(ConstantValue::Float(identity as f64))? } else { identity.to_string() };
            self.line(format!("selp.{} {accumulator}, {identity}, {accumulator}, {first};", ty.storage()));
        }
        let destination = self.destination(out)?;
        self.line(format!("mov.{} {destination}, {accumulator};", ty.storage()));
        self.line(format!("bra {done};"));
        self.line(format!("{partial}:"));
        self.plane_reduce_active(ty, &source, &destination, &mask, operation, scan, exclusive)?;
        self.line(format!("{done}:"));
        Ok(())
    }
}
