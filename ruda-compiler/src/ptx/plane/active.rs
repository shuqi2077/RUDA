use super::*;

impl Emitter {
    pub(in crate::ptx) fn plane_member_mask(&mut self) -> String {
        self.static_warp_mask.map_or_else(|| self.plane_active_mask(), |mask| format!("0x{mask:08x}"))
    }

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

    // Select the zero-based rank-th set bit. Discard the lower half only
    // when moving up. When staying in the lower half, the next (narrower)
    // low mask already excludes all upper bits. Avoid a per-lane fns here.
    pub(super) fn plane_lane_at_rank(&mut self, mask: &str, rank: &str) -> String {
        let remaining = self.reg(Scalar::U32);
        let bits = self.reg(Scalar::U32);
        let lane = self.reg(Scalar::U32);
        let count = self.reg(Scalar::U32);
        let upper = self.reg(Scalar::Pred);
        self.line(format!("mov.u32 {remaining}, {rank};"));
        self.line(format!("mov.u32 {bits}, {mask};"));
        self.line(format!("mov.u32 {lane}, 0;"));
        for width in [16, 8, 4, 2, 1] {
            let low_mask = (1u32 << width) - 1;
            self.line(format!("and.b32 {count}, {bits}, {low_mask};"));
            self.line(format!("popc.b32 {count}, {count};"));
            self.line(format!("setp.ge.u32 {upper}, {remaining}, {count};"));
            self.line(format!("@{upper} add.u32 {lane}, {lane}, {width};"));
            if width > 1 {
                self.line(format!("@{upper} shr.u32 {bits}, {bits}, {width};"));
                self.line(format!("@{upper} sub.u32 {remaining}, {remaining}, {count};"));
            }
        }
        lane
    }
}
