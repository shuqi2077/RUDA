//! Shared helpers for the extended (forced-blueprint) tier.

use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::Runtime;
use ruda_test_runtime::TestRuntime;
use ruda_kernel::dsl::client::ComputeClient;
use ruda_kernel::dsl::ir::AddressType;
use ruda_kernel::dsl::zspace::shape;
use rublas::kernel_ir::{
    components::{
        global::LoadFlows,
        stage::PartitionBuffering,
        tile::{TileMatmul, TileMatmulKind},
    },
    definition::{MatmulElems, MatmulGlobalElems, MatmulProblem, TilingBlueprint, TilingScheme},
};
use ruda_kernel::tiling::{
    MatrixLayout, PartitionSize, StageSize, SwizzleModes, TileSize,
    ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint},
};

pub(crate) fn client() -> ComputeClient<TestRuntime> {
    TestRuntime::client(&Default::default())
}

pub(crate) fn f16_elems() -> MatmulGlobalElems {
    use ruda_kernel::dsl::frontend::RudaPrimitive;
    MatmulElems::from_single_dtype(half::f16::as_type_native_unchecked()).as_global_elems()
}

pub(crate) fn f32_elems() -> MatmulGlobalElems {
    use ruda_kernel::dsl::frontend::RudaPrimitive;
    MatmulElems::from_single_dtype(f32::as_type_native_unchecked()).as_global_elems()
}

pub(crate) fn problem(
    m: usize,
    n: usize,
    k: usize,
    layouts: (MatrixLayout, MatrixLayout),
    elems: MatmulGlobalElems,
) -> MatmulProblem {
    MatmulProblem::from_parameters(
        m,
        n,
        k,
        shape![1],
        shape![1],
        layouts.0,
        layouts.1,
        MatrixLayout::RowMajor,
        None,
        None,
        elems,
        AddressType::U32,
    )
}

pub(crate) fn row_row() -> (MatrixLayout, MatrixLayout) {
    (MatrixLayout::RowMajor, MatrixLayout::RowMajor)
}

/// Tile size that works on both macOS (where CMMA is 8x8x8) and elsewhere.
#[cfg(target_os = "macos")]
pub(crate) fn default_tile_size() -> TileSize {
    TileSize { m: 8, n: 8, k: 8 }
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn default_tile_size() -> TileSize {
    TileSize {
        m: 16,
        n: 16,
        k: 16,
    }
}

pub(crate) fn tiling_scheme(
    tile: TileSize,
    partition: PartitionSize,
    stage: StageSize,
) -> TilingScheme {
    TilingScheme::builder()
        .with_tile_size(tile)
        .with_partition_size(partition)
        .with_stage_size(stage)
        .build()
        .unwrap()
}

pub(crate) fn plane_blueprint(
    client: &ComputeClient<TestRuntime>,
    problem: &MatmulProblem,
    tile: TileSize,
    partition: PartitionSize,
    stage: StageSize,
) -> TilingBlueprint {
    let scheme = tiling_scheme(tile, partition, stage);
    let plane_dim = client.properties().hardware.plane_size_max;
    // Default the partition buffering based on whether there is more than one
    // tile along n inside a partition: double buffering at the partition level
    // requires at least two tiles to pipeline, so partitions with n=1 must use
    // Single (otherwise every test with partition.n=1 fails under strict mode).
    let partition_buffering = if partition.n > 1 {
        PartitionBuffering::Double
    } else {
        PartitionBuffering::Single
    };
    TilingBlueprint::builder(TileMatmulKind::Cmma, scheme, plane_dim, problem)
        .partition_buffering(partition_buffering)
        .build()
}

pub(crate) fn plane_blueprint_with(
    client: &ComputeClient<TestRuntime>,
    problem: &MatmulProblem,
    tile: TileSize,
    partition: PartitionSize,
    stage: StageSize,
    swizzle: SwizzleModes,
    hyperruda: HyperrudaBlueprint,
    partition_buffering: PartitionBuffering,
    specialization: LoadFlows,
) -> TilingBlueprint {
    let scheme = tiling_scheme(tile, partition, stage);
    let plane_dim = client.properties().hardware.plane_size_max;
    TilingBlueprint::builder(TileMatmulKind::Cmma, scheme, plane_dim, problem)
        .shared_swizzle(swizzle)
        .hyperruda_blueprint(hyperruda)
        .partition_buffering(partition_buffering)
        .load_specialization_config(specialization)
        .build()
}

pub(crate) fn default_hyperruda() -> HyperrudaBlueprint {
    HyperrudaBlueprint::builder()
        .global_order(GlobalOrder::RowMajor)
        .ruda_count_strategy(RudaCountStrategy::FromProblem)
        .build()
}
