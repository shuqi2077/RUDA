use super::*;

impl Emitter {
    pub(super) fn plane_active_mask(&mut self) -> String {
        let mask = self.reg(Scalar::U32);
        if self.target.version >= (6, 2) {
            self.line(format!("activemask.b32 {mask};"));
        } else {
            let predicate = self.reg(Scalar::Pred);
            self.line(format!("setp.eq.u32 {predicate}, 0, 0;"));
            self.line(format!("vote.ballot.b32 {mask}, {predicate};"));
        }
        mask
    }

    pub(super) fn plane_lane_at_rank(&mut self, mask: &str, rank: &str) -> String {
        let remaining = self.reg(Scalar::U32);
        let bits = self.reg(Scalar::U32);
        let lane = self.reg(Scalar::U32);
        let lower = self.reg(Scalar::U32);
        let count = self.reg(Scalar::U32);
        let shifted = self.reg(Scalar::U32);
        let next_rank = self.reg(Scalar::U32);
        let next_lane = self.reg(Scalar::U32);
        let upper = self.reg(Scalar::Pred);
        self.line(format!("mov.u32 {remaining}, {rank};"));
        self.line(format!("mov.u32 {bits}, {mask};"));
        self.line(format!("mov.u32 {lane}, 0;"));
        for width in [16, 8, 4, 2, 1] {
            let low_mask = (1u32 << width) - 1;
            self.line(format!("and.b32 {lower}, {bits}, {low_mask};"));
            self.line(format!("popc.b32 {count}, {lower};"));
            self.line(format!("setp.ge.u32 {upper}, {remaining}, {count};"));
            self.line(format!("shr.u32 {shifted}, {bits}, {width};"));
            self.line(format!("selp.u32 {bits}, {shifted}, {lower}, {upper};"));
            self.line(format!("sub.u32 {next_rank}, {remaining}, {count};"));
            self.line(format!("selp.u32 {remaining}, {next_rank}, {remaining}, {upper};"));
            self.line(format!("add.u32 {next_lane}, {lane}, {width};"));
            self.line(format!("selp.u32 {lane}, {next_lane}, {lane}, {upper};"));
        }
        lane
    }
}
