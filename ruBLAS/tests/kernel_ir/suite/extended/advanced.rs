//! Forced-blueprint tests for non-default hyperruda / specialization /
//! partition-buffering knobs. All of these are applied to one representative
//! routine (SimpleCyclicCmma or DoubleCyclicCmma as appropriate) — the point
//! is to exercise each knob at least once, not to cover every combo.
//!
//! Swizzle knobs are not covered here: CMMA does not support swizzling, and
//! MMA-based swizzling depends on the `alignment` client feature, so those
//! cases live in `full/` with the platform-specific routines instead.

use rublas::kernel_ir::{
    components::{
        global::{InputLoadFlow, LoadFlows},
        stage::PartitionBuffering,
    },
    launch::Strategy,
    routines::BlueprintStrategy,
};
use ruda_kernel::tiling::{
    PartitionSize, StageSize, SwizzleModes,
    ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation},
    stage::SwizzleMode,
};

use super::common::{
    client, default_hyperruda, default_tile_size, f16_elems, plane_blueprint_with, problem, row_row,
};
use crate::suite::test_matmul_strategy;

fn run_with(
    partition: PartitionSize,
    stage: StageSize,
    swizzle: SwizzleModes,
    hyperruda: HyperrudaBlueprint,
    buffering: PartitionBuffering,
    specialization: LoadFlows,
    strategy: impl FnOnce(rublas::kernel_ir::definition::TilingBlueprint) -> Strategy,
) {
    let c = client();
    let p = problem(256, 256, 256, row_row(), f16_elems());
    let bp = plane_blueprint_with(
        &c,
        &p,
        default_tile_size(),
        partition,
        stage,
        swizzle,
        hyperruda,
        buffering,
        specialization,
    );
    test_matmul_strategy(c, p, strategy(bp));
}

fn default_swizzle() -> SwizzleModes {
    SwizzleModes {
        lhs: SwizzleMode::None,
        rhs: SwizzleMode::None,
        ..Default::default()
    }
}

fn both_main() -> LoadFlows {
    LoadFlows {
        lhs: InputLoadFlow::MainOnly,
        rhs: InputLoadFlow::MainOnly,
    }
}

/// Default partition/stage for single-partition knob tests (hyperruda).
fn simple_partition() -> PartitionSize {
    PartitionSize { m: 1, n: 1, k: 1 }
}

fn simple_stage() -> StageSize {
    StageSize { m: 2, n: 2, k: 1 }
}

/// Partition/stage suitable for specialization: matches the Specialized routine
/// default (partition (1, 4, 2), stage (4, 1, 1)) so there are enough main-flow
/// planes for the load-only planes to pair against.
fn specialized_partition() -> PartitionSize {
    PartitionSize { m: 1, n: 4, k: 2 }
}

fn specialized_stage() -> StageSize {
    StageSize { m: 4, n: 1, k: 1 }
}

// -- Hyperruda global order --------------------------------------------------

#[test]
fn hyperruda_swizzle_col() {
    run_with(
        simple_partition(),
        simple_stage(),
        default_swizzle(),
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::SwizzleCol(2))
            .ruda_count_strategy(RudaCountStrategy::FromProblem)
            .build(),
        PartitionBuffering::Single,
        both_main(),
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn hyperruda_col_flattened() {
    run_with(
        simple_partition(),
        simple_stage(),
        default_swizzle(),
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::ColMajor)
            .ruda_count_strategy(RudaCountStrategy::Flattened)
            .build(),
        PartitionBuffering::Single,
        both_main(),
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn hyperruda_sm_exact() {
    run_with(
        simple_partition(),
        simple_stage(),
        default_swizzle(),
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::RowMajor)
            .ruda_count_strategy(RudaCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Exact,
                rudas_first: false,
            })
            .build(),
        PartitionBuffering::Single,
        both_main(),
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn hyperruda_spread() {
    run_with(
        simple_partition(),
        simple_stage(),
        default_swizzle(),
        HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(2))
            .ruda_count_strategy(RudaCountStrategy::Spread)
            .build(),
        PartitionBuffering::Single,
        both_main(),
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

// -- Load specialization (applied on a routine that supports it) -------------
//
// These use the same partition/stage shape the `Specialized*` routines default
// to (partition (1, 4, 2), stage (4, 1, 1)); smaller shapes don't leave enough
// main-flow planes for the load-only planes to balance against and produce
// all-zero output.

#[test]
fn specialization_main_load() {
    run_with(
        specialized_partition(),
        specialized_stage(),
        default_swizzle(),
        default_hyperruda(),
        PartitionBuffering::Single,
        LoadFlows {
            lhs: InputLoadFlow::MainOnly,
            rhs: InputLoadFlow::LoadOnly,
        },
        |bp| Strategy::SpecializedCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn specialization_load_main() {
    run_with(
        specialized_partition(),
        specialized_stage(),
        default_swizzle(),
        default_hyperruda(),
        PartitionBuffering::Single,
        LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::MainOnly,
        },
        |bp| Strategy::SpecializedCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn specialization_load_load() {
    run_with(
        specialized_partition(),
        specialized_stage(),
        default_swizzle(),
        default_hyperruda(),
        PartitionBuffering::Single,
        LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::LoadOnly,
        },
        |bp| Strategy::SpecializedCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

// -- Partition buffering -----------------------------------------------------
//
// Partition buffering pipelines inside a partition along n, so it needs at
// least two tiles along n (partition.n >= 2).

#[test]
fn partition_buffering_double() {
    run_with(
        PartitionSize { m: 1, n: 2, k: 1 },
        simple_stage(),
        default_swizzle(),
        default_hyperruda(),
        PartitionBuffering::Double,
        both_main(),
        |bp| Strategy::DoubleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}
