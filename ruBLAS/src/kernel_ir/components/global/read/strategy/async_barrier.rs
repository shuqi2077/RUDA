use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::barrier::Barrier;
use ruda_kernel::dsl::prelude::*;

use crate::kernel_ir::{
    components::{
        global::{SharedGlobalMatmulConfig, read::SyncStrategy},
        stage::StageConfig,
    },
    definition::MatmulTypes,
};

/// Asynchronous barrier for `async_memcpy`
pub struct AsyncBarrier {}

#[ruda]
impl SyncStrategy for AsyncBarrier {
    type Barrier = Shared<Barrier>;

    fn create_barrier() -> Self::Barrier {
        Barrier::shared(RUDA_DIM, UNIT_POS == 0)
    }

    fn sync<MP: MatmulTypes, S: StageConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig<S>,
    ) {
        barrier.arrive_and_wait();
    }
}

/// Asynchronous barrier for `async_copy`
pub struct AsyncCopy {}

#[ruda]
impl SyncStrategy for AsyncCopy {
    type Barrier = Shared<Barrier>;

    fn create_barrier() -> Self::Barrier {
        Barrier::shared(RUDA_DIM, UNIT_POS == 0)
    }

    fn sync<MP: MatmulTypes, S: StageConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] _config: SharedGlobalMatmulConfig<S>,
    ) {
        barrier.commit_copy_async();
        barrier.arrive_and_wait();
    }
}
