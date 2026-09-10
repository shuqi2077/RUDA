use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::barrier::Barrier;
use ruda_kernel::dsl::prelude::barrier::BarrierToken;
use ruda_kernel::dsl::prelude::*;

use crate::kernel_ir::components::{
    global::{GlobalConfig, SharedGlobalMatmulConfig, read::SyncStrategy},
    stage::StageConfig,
};
use crate::kernel_ir::definition::{LhsS, MatmulTypes, RhsS};

/// Asynchronous barrier for TMA loads
pub struct AsyncTma {}

#[ruda]
impl SyncStrategy for AsyncTma {
    type Barrier = Shared<Barrier>;

    fn create_barrier() -> Self::Barrier {
        let bar = Barrier::shared(RUDA_DIM, UNIT_POS == 0);
        sync_async_proxy_shared();
        bar
    }

    fn sync<MP: MatmulTypes, S: StageConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] config: SharedGlobalMatmulConfig<S>,
    ) {
        let lhs_elem_size = LhsS::<MP>::type_size().comptime();
        let rhs_elem_size = RhsS::<MP>::type_size().comptime();
        let lhs_bytes =
            config.lhs_reader_config().smem_config.elements_per_stage() * lhs_elem_size as u32;
        let rhs_bytes =
            config.rhs_reader_config().smem_config.elements_per_stage() * rhs_elem_size as u32;
        let num_bytes = lhs_bytes + rhs_bytes;
        let token = arrive_tma(barrier, num_bytes);
        barrier.wait(token);
    }
}

#[ruda]
/// Barrier for TMA
pub fn arrive_tma(barrier: &Barrier, #[comptime] num_bytes: u32) -> BarrierToken {
    let expected = select(UNIT_POS == 0, num_bytes, 0);
    barrier.arrive_and_expect_tx(1, expected)
}
