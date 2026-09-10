pub mod row_fp {
    use super::*;
    use rublas::kernel_ir::definition::{MatmulProblem, TilingScheme};
    use ruda_kernel::tiling::ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation};

    fn hyperruda_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HyperrudaBlueprint {
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::RowMajor)
            .ruda_count_strategy(RudaCountStrategy::FromProblem)
            .build()
    }

    include!("partition_buffering.rs");
}

mod swizzlecol_fp {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation};

    fn hyperruda_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HyperrudaBlueprint {
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::SwizzleCol(2))
            .ruda_count_strategy(RudaCountStrategy::FromProblem)
            .build()
    }

    include!("partition_buffering.rs");
}

mod col_fl {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation};

    fn hyperruda_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HyperrudaBlueprint {
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::ColMajor)
            .ruda_count_strategy(RudaCountStrategy::Flattened)
            .build()
    }

    include!("partition_buffering.rs");
}

mod swizzlerow_fl {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation};

    fn hyperruda_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HyperrudaBlueprint {
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(2))
            .ruda_count_strategy(RudaCountStrategy::Flattened)
            .build()
    }

    include!("partition_buffering.rs");
}

mod row_sm_exact {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation};

    fn hyperruda_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HyperrudaBlueprint {
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::RowMajor)
            .ruda_count_strategy(RudaCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Exact,
                rudas_first: false,
            })
            .build()
    }

    include!("partition_buffering.rs");
}

mod row_sm_full {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation};

    fn hyperruda_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HyperrudaBlueprint {
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::RowMajor)
            .ruda_count_strategy(RudaCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Full,
                rudas_first: false,
            })
            .build()
    }

    include!("partition_buffering.rs");
}

mod swizzlerow_ruda_full {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation};

    fn hyperruda_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HyperrudaBlueprint {
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(2))
            .ruda_count_strategy(RudaCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Full,
                rudas_first: true,
            })
            .build()
    }

    include!("partition_buffering.rs");
}

mod swizzlerow_spread {
    use super::*;
    use rublas::kernel_ir::definition::TilingScheme;
    use ruda_kernel::tiling::ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation};

    fn hyperruda_blueprint(
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
    ) -> HyperrudaBlueprint {
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(2))
            .ruda_count_strategy(RudaCountStrategy::Spread)
            .build()
    }

    include!("partition_buffering.rs");
}
