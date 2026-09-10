use super::{Result, emit::Emitter, invalid, types::Scalar};
use ruda_core::ir::{Plane, Variable};

mod shuffle;
mod reduce;
mod active;
mod reduce_active;

impl Emitter {
    pub fn plane(&mut self, operation: Plane, out: Variable) -> Result<()> {
        use Plane::*;
        match operation {
            Elect => self.plane_elect(out),
            All(op) => self.plane_vote(op.input, out, "all"),
            Any(op) => self.plane_vote(op.input, out, "any"),
            Ballot(op) => self.plane_vote(op.input, out, "ballot"),
            Broadcast(op) | Shuffle(op) => self.plane_shuffle(op, out, "idx"),
            ShuffleXor(op) => self.plane_shuffle(op, out, "bfly"),
            ShuffleUp(op) => self.plane_shuffle(op, out, "up"),
            ShuffleDown(op) => self.plane_shuffle(op, out, "down"),
            Sum(op) => self.plane_reduce(op.input, out, "add", false, false),
            Prod(op) => self.plane_reduce(op.input, out, "mul", false, false),
            Min(op) => self.plane_reduce(op.input, out, "min", false, false),
            Max(op) => self.plane_reduce(op.input, out, "max", false, false),
            InclusiveSum(op) => self.plane_reduce(op.input, out, "add", true, false),
            InclusiveProd(op) => self.plane_reduce(op.input, out, "mul", true, false),
            ExclusiveSum(op) => self.plane_reduce(op.input, out, "add", true, true),
            ExclusiveProd(op) => self.plane_reduce(op.input, out, "mul", true, true),
        }
    }

    fn plane_vote(&mut self, input: Variable, out: Variable, operation: &str) -> Result<()> {
        if Scalar::of(input.ty)? != Scalar::Pred {
            return Err(invalid("plane vote requires a predicate input"));
        }
        let expected = if operation == "ballot" { Scalar::U32 } else { Scalar::Pred };
        if Scalar::of(out.ty)? != expected {
            return Err(invalid("plane vote output type"));
        }
        let source = self.value(input)?;
        let mask = self.plane_active_mask();
        let destination = self.destination(out)?;
        let suffix = if operation == "ballot" { "b32" } else { "pred" };
        self.line(format!("vote.sync.{operation}.{suffix} {destination}, {source}, {mask};"));
        Ok(())
    }

    fn plane_elect(&mut self, out: Variable) -> Result<()> {
        if Scalar::of(out.ty)? != Scalar::Pred {
            return Err(invalid("plane election requires a predicate output"));
        }
        let destination = self.destination(out)?;
        let mask = self.plane_active_mask();
        if self.target.sm >= 90 && self.target.version >= (7, 8) {
            self.line(format!("elect.sync _|{destination}, {mask};"));
        } else {
            let reversed = self.reg(Scalar::U32);
            let leader = self.reg(Scalar::U32);
            let lane = self.reg(Scalar::U32);
            self.line(format!("brev.b32 {reversed}, {mask};"));
            self.line(format!("clz.b32 {leader}, {reversed};"));
            self.line(format!("mov.u32 {lane}, %laneid;"));
            self.line(format!("setp.eq.u32 {destination}, {lane}, {leader};"));
        }
        Ok(())
    }
}
