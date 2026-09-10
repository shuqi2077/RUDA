// SPDX-License-Identifier: Apache-2.0
//! Dependency-free launch geometry for the opt-in warp-shared direct solvers.
//! This describes source-level storage/work, NOT measured bandwidth or latency.
use core::fmt;

/// Which scratch layout is needed. Both algorithms preserve the serial APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarpDirectKind { Cholesky, Lu }

/// Immutable validated plan; one 32-thread block owns one system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarpDirectPlan {
    batch: usize,
    order: usize,
    rhs: usize,
    pitch: usize,
    shared_bytes: usize,
    kind: WarpDirectKind,
}

/// Capability snapshot supplied by a real runtime, not inferred from its name.
#[derive(Clone, Copy, Debug)]
pub struct WarpDirectLimits {
    pub plane_min: u32,
    pub plane_max: u32,
    pub plane_ops: bool,
    pub max_threads: u32,
    pub max_block_x: u32,
    pub max_grid_x: u32,
    pub shared_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarpPlanError { InvalidShape, SizeOverflow, Unsupported(&'static str) }
impl fmt::Display for WarpPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => f.write_str("require n=1..32 and nrhs=1..8"),
            Self::SizeOverflow => f.write_str("32-bit device indexing/byte count overflow"),
            Self::Unsupported(s) => write!(f, "warp solver requires {s}"),
        }
    }
}
impl std::error::Error for WarpPlanError {}

impl WarpDirectPlan {
    pub const THREADS: u32 = 32;
    pub fn new(batch: usize, order: usize, rhs: usize, kind: WarpDirectKind) -> Result<Self, WarpPlanError> {
        if !(1..=32).contains(&order) || !(1..=8).contains(&rhs) {
            return Err(WarpPlanError::InvalidShape);
        }
        for per_system in [order * order, order * rhs, order, 1] {
            let elements = batch.checked_mul(per_system).ok_or(WarpPlanError::SizeOverflow)?;
            if elements > u32::MAX as usize { return Err(WarpPlanError::SizeOverflow); }
            elements.checked_mul(4).ok_or(WarpPlanError::SizeOverflow)?;
        }
        // Odd pitch is relatively prime to 32: rows at a fixed column map to
        // distinct 32-bit shared-memory banks. Padding cells are never read.
        let pitch = order + usize::from(order % 2 == 0);
        let pivots = if kind == WarpDirectKind::Lu { order } else { 0 };
        let shared_bytes = 4 * (order * pitch + order * rhs + pivots);
        Ok(Self { batch, order, rhs, pitch, shared_bytes, kind })
    }
    /// Empty batches require no launch or plane capability, but shapes remain checked.
    pub fn check_device(&self, limits: WarpDirectLimits) -> Result<(), WarpPlanError> {
        if self.batch == 0 { return Ok(()); }
        if limits.plane_min != 32 || limits.plane_max != 32 || !limits.plane_ops {
            return Err(WarpPlanError::Unsupported("a fixed 32-lane plane with plane operations"));
        }
        if limits.max_threads < 32 || limits.max_block_x < 32 {
            return Err(WarpPlanError::Unsupported("32 X threads per block"));
        }
        if self.batch > limits.max_grid_x as usize {
            return Err(WarpPlanError::Unsupported("one X block per system; split this batch explicitly"));
        }
        if self.shared_bytes > limits.shared_bytes {
            return Err(WarpPlanError::Unsupported("sufficient per-block shared memory"));
        }
        Ok(())
    }
    pub fn batch(&self) -> usize { self.batch }
    pub fn order(&self) -> usize { self.order }
    pub fn rhs(&self) -> usize { self.rhs }
    pub fn pitch(&self) -> usize { self.pitch }
    pub fn shared_bytes(&self) -> usize { self.shared_bytes }
    pub fn kind(&self) -> WarpDirectKind { self.kind }
    /// Successful-path logical global payload: A/B loaded once, factors/X stored
    /// once, and status/pivots stored once. Excludes allocator, cache-line and
    /// compiler-inserted traffic. Does NOT assert actual DRAM bytes transferred.
    pub fn logical_global_bytes_per_system(&self) -> usize {
        8 * (self.order * self.order + self.order * self.rhs) + 4
            + if self.kind == WarpDirectKind::Lu { 4 * self.order } else { 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn limits() -> WarpDirectLimits {
        WarpDirectLimits { plane_min:32, plane_max:32, plane_ops:true, max_threads:1024,
            max_block_x:1024, max_grid_x:65535, shared_bytes:48*1024 }
    }
    #[test] fn invalid_orders_and_rhs_are_rejected() {
        for (n,r) in [(0,1),(33,1),(1,0),(1,9)] {
            assert_eq!(WarpDirectPlan::new(1,n,r,WarpDirectKind::Lu),Err(WarpPlanError::InvalidShape));
        }
    }
    #[test] fn overflow_is_rejected_without_allocating() {
        assert!(WarpDirectPlan::new(usize::MAX,32,8,WarpDirectKind::Lu).is_err());
        assert!(WarpDirectPlan::new(u32::MAX as usize/1024+1,32,8,WarpDirectKind::Lu).is_err());
    }
    #[test] fn shared_bounds_for_every_shape() {
        for n in 1..=32 { for r in 1..=8 {
            for kind in [WarpDirectKind::Cholesky,WarpDirectKind::Lu] {
                let p=WarpDirectPlan::new(3,n,r,kind).unwrap();
                assert!(p.pitch()>=n && p.pitch()<=n+1 && p.pitch()%2==1);
                assert!(p.shared_bytes()<=5376);
                assert_eq!(p.batch(),3); assert_eq!(p.kind(),kind);
                p.check_device(limits()).unwrap();
            }
        }}
    }
    #[test] fn columns_have_no_bank_duplicates_in_this_layout_model() {
        for n in 1..=32 {
            let p=WarpDirectPlan::new(1,n,1,WarpDirectKind::Lu).unwrap();
            for c in 0..n {
                let mut banks=std::collections::BTreeSet::new();
                for row in 0..n { assert!(banks.insert((row*p.pitch()+c)%32)); }
            }
        }
    }
    #[test] fn strided_copy_covers_only_logical_elements_once() {
        for len in 1..=1024 {
            let mut hits=vec![0;len];
            for lane in 0..32 { for i in (lane..len).step_by(32) { hits[i]+=1; } }
            assert!(hits.iter().all(|&x|x==1));
        }
    }
    #[test] fn empty_batch_has_no_hardware_requirement() {
        let p=WarpDirectPlan::new(0,32,8,WarpDirectKind::Lu).unwrap();
        p.check_device(WarpDirectLimits{plane_min:0,plane_max:0,plane_ops:false,
            max_threads:0,max_block_x:0,max_grid_x:0,shared_bytes:0}).unwrap();
    }
    #[test] fn each_hardware_requirement_is_checked() {
        let p=WarpDirectPlan::new(65,32,8,WarpDirectKind::Lu).unwrap();
        let mut bad=limits();bad.plane_min=16;assert!(p.check_device(bad).is_err());
        bad=limits();bad.plane_max=64;assert!(p.check_device(bad).is_err());
        bad=limits();bad.plane_ops=false;assert!(p.check_device(bad).is_err());
        bad=limits();bad.max_threads=31;assert!(p.check_device(bad).is_err());
        bad=limits();bad.max_block_x=31;assert!(p.check_device(bad).is_err());
        bad=limits();bad.max_grid_x=64;assert!(p.check_device(bad).is_err());
        bad=limits();bad.shared_bytes=5375;assert!(p.check_device(bad).is_err());
        bad=limits();bad.shared_bytes=5376;p.check_device(bad).unwrap();
    }
    #[test] fn maximum_layout_and_payload_are_explicit() {
        let a=WarpDirectPlan::new(1,32,8,WarpDirectKind::Cholesky).unwrap();
        let b=WarpDirectPlan::new(1,32,8,WarpDirectKind::Lu).unwrap();
        assert_eq!(a.shared_bytes(),5248);assert_eq!(b.shared_bytes(),5376);
        assert_eq!(a.logical_global_bytes_per_system(),10244);
        assert_eq!(b.logical_global_bytes_per_system(),10372);
    }
}
