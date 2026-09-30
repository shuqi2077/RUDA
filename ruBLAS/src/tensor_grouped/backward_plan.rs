//! Host-only checked launch geometry. No runtime, allocation or GPU fallback here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BackwardTilePlan {
    pub dinput_grid: [u32; 3],
    pub dweight_grid: [u32; 3],
}

impl BackwardTilePlan {
    pub const TILE: usize = 16;
    // Two 16x16 half/bfloat matrices plus a 16x16 FP32 store tile.
    pub const SHARED_BYTES: usize = 2048;

    pub fn new(experts: usize, columns: usize, inner: usize, limits: [u32; 3])
        -> Result<Self, &'static str>
    {
        if experts == 0 || columns == 0 || inner == 0 {
            return Err("grouped backward dimensions must be positive");
        }
        if experts.checked_mul(columns).and_then(|x| x.checked_mul(inner))
            .is_none_or(|x| x > u32::MAX as usize)
        {
            return Err("grouped backward weights exceed U32 indexing");
        }
        let x = u32::try_from(inner.div_ceil(Self::TILE))
            .map_err(|_| "grouped backward X grid overflows U32")?;
        let y = u32::try_from(experts)
            .map_err(|_| "grouped backward Y grid overflows U32")?;
        let z = u32::try_from(columns.div_ceil(Self::TILE))
            .map_err(|_| "grouped backward Z grid overflows U32")?;
        if x > limits[0] || y > limits[1] || z > limits[2] {
            return Err("grouped backward grid exceeds device limits");
        }
        Ok(Self { dinput_grid: [x, y, 1], dweight_grid: [x, y, z] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn v31_geometry_tails() {
        let p = BackwardTilePlan::new(4, 33, 17, [100,100,100]).unwrap();
        assert_eq!(p.dinput_grid, [2,4,1]);
        assert_eq!(p.dweight_grid, [2,4,3]);
    }
    #[test] fn v31_geometry_exact_tiles() {
        let p = BackwardTilePlan::new(1,16,32,[2,1,1]).unwrap();
        assert_eq!(p.dweight_grid,[2,1,1]);
    }
    #[test] fn v31_geometry_zero_rejected() {
        for dims in [[0,1,1],[1,0,1],[1,1,0]] {
            assert!(BackwardTilePlan::new(dims[0],dims[1],dims[2],[1,1,1]).is_err());
        }
    }
    #[test] fn v31_geometry_limits_checked() {
        for limits in [[1,4,3],[2,3,3],[2,4,2]] {
            assert!(BackwardTilePlan::new(4,33,17,limits).is_err());
        }
    }
    #[test] fn v31_geometry_index_overflow_rejected() {
        assert!(BackwardTilePlan::new(u32::MAX as usize,2,1,[u32::MAX;3]).is_err());
        assert!(BackwardTilePlan::new(usize::MAX,usize::MAX,2,[u32::MAX;3]).is_err());
    }
    #[test] fn v31_geometry_shared_storage() {
        assert_eq!(BackwardTilePlan::SHARED_BYTES,2*16*16*2+16*16*4);
    }
}
