use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::RudaDim;

use crate::kernel_ir::{
    components::global::multi_stage::LoadMaxRoundPlaneCount,
    definition::{MatmulElems, MatmulSetupError, MatmulVectorSizes, TilingScheme},
};

#[allow(unused_variables)]
pub fn ruda_dim_validation(ruda_dim: RudaDim) -> Result<(), MatmulSetupError> {
    #[cfg(target_os = "macos")]
    {
        if ruda_dim.num_elems() >= 512 {
            use crate::kernel_ir::definition::{MatmulAvailabilityError, MatmulSetupError};

            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::RudaDimTooBig(ruda_dim),
            ));
        }
    }

    Ok(())
}

/// Maximal number of planes each reader can handle to divide its workload evenly
pub struct MaxGlobalReaderPlanes {
    pub lhs: u32,
    pub rhs: u32,
}

impl MaxGlobalReaderPlanes {
    /// Create a MaxGlobalReaderPlanes
    pub fn new<LL: LoadMaxRoundPlaneCount, RL: LoadMaxRoundPlaneCount>(
        tiling_scheme: &TilingScheme,
        vector_sizes: &MatmulVectorSizes,
        plane_dim: u32,
        dtypes: &MatmulElems,
    ) -> Self {
        MaxGlobalReaderPlanes {
            lhs: LL::max_round_plane_count(
                tiling_scheme.tile_size.m * tiling_scheme.tile_size.k,
                (tiling_scheme.partition_size.m
                    * tiling_scheme.stage_size.m
                    * tiling_scheme.partition_size.k
                    * tiling_scheme.stage_size.k) as u32,
                vector_sizes.lhs,
                plane_dim,
                dtypes.lhs_global,
            ),
            rhs: RL::max_round_plane_count(
                tiling_scheme.tile_size.k * tiling_scheme.tile_size.n,
                (tiling_scheme.partition_size.k
                    * tiling_scheme.stage_size.k
                    * tiling_scheme.partition_size.n
                    * tiling_scheme.stage_size.n) as u32,
                vector_sizes.rhs,
                plane_dim,
                dtypes.rhs_global,
            ),
        }
    }
}
