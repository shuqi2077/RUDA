use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::ir::features::MmaConfig;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::client::ComputeClient;
use ruda_kernel::tiling::{
    ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation},
    tile::Strided,
};
use std::{fmt::Display, marker::PhantomData};

use crate::kernel_ir::{
    components::{
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            PlaneWriterFamily,
            read::{
                FullLoadingStrategy, async_full_tma::AsyncFullTmaLoading,
                sync_full_cyclic::SyncFullCyclicLoading,
            },
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{ColMajorTilingOrder, PartitionBuffering, PlaneMatmulFamily, RowMajorTilingOrder},
        tile::TileMatmulKind,
    },
    routines::{
        Routine, TilingArgs,
        selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane},
    },
};
use crate::kernel_ir::{
    definition::{
        MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes, MultiRowStrategy,
        TilingBlueprint, TilingScheme, adjust_dtypes,
    },
    routines::ExpandInfo,
};
use crate::kernel_ir::{
    routines::{BlueprintStrategy, DeviceSettings, LaunchInfo},
    {components::batch::BatchMatmulFamily, launch::RuntimeConfig},
};

/// Plane accelerated single stage matmul with configurable readers (default to cyclic)
pub struct SimpleAlgorithm<
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    AL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    const FIRST_K_TILE: u32 = 0,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _al: PhantomData<AL>,
}

pub type SimpleTmaAlgorithm = SimpleAlgorithm<
    AsyncFullTmaLoading,
    AsyncFullTmaLoading,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
>;
pub type SimpleBarrierAlgorithm<L> = SimpleAlgorithm<L, L>;

pub type SimpleResidueFirstAlgorithm = SimpleAlgorithm<
    SyncFullCyclicLoading<ColMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    SyncFullCyclicLoading<RowMajorTilingOrder>,
    32,
>;

#[derive(Debug, Clone)]
pub struct SimpleArgs {
    /// Which tile matmul variant to dispatch on. Also stored into the blueprint.
    pub tile_matmul: TileMatmulKind,
    // Uses an optimized multi rows strategy.
    pub multi_rows: bool,
}

impl Default for SimpleArgs {
    fn default() -> Self {
        Self {
            tile_matmul: TileMatmulKind::Cmma,
            multi_rows: false,
        }
    }
}

impl TilingArgs for SimpleArgs {
    fn set_tile_matmul(&mut self, kind: TileMatmulKind) {
        self.tile_matmul = kind;
    }
}

impl Display for SimpleArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.multi_rows { "_multi_rows" } else { "" })
    }
}

impl<RC, LL, RL, AL, const FIRST_K_TILE: u32> Routine<RC> for SimpleAlgorithm<LL, RL, AL, FIRST_K_TILE>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC, TileKind = Strided>,
    RL: FullLoadingStrategy<RC, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, TileKind = Strided>,
{
    type Strategy = SimpleArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        SimpleMatmulFamily<
            PlaneMatmulFamily<LL::Stage, RL::Stage, Option<AL::Stage>>,
            RC,
            LL,
            RL,
            AL,
            PlaneWriterFamily,
            FIRST_K_TILE,
        >,
        RowMajorGlobalPartitionMatmul,
    >;
    type Blueprint = TilingBlueprint;
    type Config = <Self::BatchMatmul as BatchMatmulFamily<RC>>::Config;

    fn expand_blueprint<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        strategy: &BlueprintStrategy<RC, Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let mut dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let client = &device_settings.client;

        let tile_matmul = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.tile_matmul,
            BlueprintStrategy::Inferred(args) => args.tile_matmul,
        };

        if tile_matmul.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (mut blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(strategy) => {
                if strategy.multi_rows {
                    infer_blueprint_multi_rows::<R>(
                        tile_matmul,
                        client,
                        problem,
                        device_settings.plane_dim,
                        dtypes,
                        &device_settings.vector_sizes,
                    )
                } else {
                    infer_blueprint_plane::<R>(
                        tile_matmul,
                        client,
                        problem,
                        device_settings.plane_dim,
                        dtypes,
                        &device_settings.vector_sizes,
                        PlaneTilingBlueprintOptions {
                            partition_buffering: Some(PartitionBuffering::Single),
                            tiny_selection_enabled: true,
                            swizzled: tile_matmul.should_swizzle(client),
                            ..Default::default()
                        },
                    )
                }?
            }
        };
        if FIRST_K_TILE != 0 {
            let tile_k = blueprint.tiling_scheme.tile_size.k();
            if !FIRST_K_TILE.is_multiple_of(tile_k) {
                return Err(MatmulSetupError::InvalidConfig(Box::new("residue-first K tile is incompatible with the matrix instruction".to_string())));
            }
            blueprint.tiling_scheme.partition_size.k = u8::try_from(FIRST_K_TILE / tile_k)
                .map_err(|_| MatmulSetupError::InvalidConfig(Box::new("residue-first K partition exceeds u8".to_string())))?;
            blueprint.tiling_scheme.stage_size.k = 1;
            blueprint.check_k_bounds = true;
        }
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<TilingBlueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        let client = &device_settings.client;

        Self::validate_blueprint(
            client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let rudadim_resource = Self::BatchMatmul::rudadim_resource(
            &blueprint,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        LaunchInfo::new(
            blueprint,
            dtypes,
            problem,
            rudadim_resource,
            device_settings,
        )
    }
}

fn infer_blueprint_multi_rows<R: Runtime>(
    tile_matmul: TileMatmulKind,
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    mut dtypes: MatmulElems,
    vector_sizes: &MatmulVectorSizes,
) -> Result<(TilingBlueprint, MatmulElems), MatmulSetupError> {
    adjust_dtypes(client, &mut dtypes, tile_matmul.requires_accelerator());

    let supported = |m: u32, n: u32, k: u32| {
        tile_matmul.is_supported(
            client,
            MmaConfig {
                a_type: dtypes.lhs_register,
                b_type: dtypes.rhs_register,
                cd_type: dtypes.acc_register,
                m,
                n,
                k,
            },
        )
    };
    let ruda_count_strategy = match client.properties().hardware.num_streaming_multiprocessors {
        Some(num_sms) => RudaCountStrategy::Sm {
            num_sms,
            sm_usage: SmAllocation::Exact,
            rudas_first: true,
        },
        None => RudaCountStrategy::Flattened,
    };

    if supported(8, 32, 16) {
        // A lot of multi-rows balanced with a
        // tile size of (8, 32, 16)
        let tiling_scheme = TilingScheme::builder()
            .with_tile_size((8, 32, 16).into())
            .with_partition_size((4, 4, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap();

        let hyperruda = HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(4))
            .ruda_count_strategy(ruda_count_strategy)
            .build();

        Ok((
            TilingBlueprint::builder(tile_matmul, tiling_scheme, plane_dim, problem)
                .partition_buffering(PartitionBuffering::Single)
                .hyperruda_blueprint(hyperruda)
                .build(),
            dtypes,
        ))
    } else if supported(8, 8, 8) {
        let tiling_scheme = TilingScheme::builder()
            .with_tile_size((8, 8, 8).into())
            .with_partition_size((4, 8, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap();
        let hyperruda = HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(4))
            .ruda_count_strategy(ruda_count_strategy)
            .build();

        Ok((
            TilingBlueprint::builder(tile_matmul, tiling_scheme, plane_dim, problem)
                .partition_buffering(PartitionBuffering::Single)
                .hyperruda_blueprint(hyperruda)
                .build(),
            dtypes,
        ))
    } else {
        infer_blueprint_plane::<R>(
            tile_matmul,
            client,
            problem,
            plane_dim,
            dtypes,
            vector_sizes,
            PlaneTilingBlueprintOptions {
                partition_buffering: Some(PartitionBuffering::Single),
                multi_row_strategy: MultiRowStrategy::Always(2),
                partition_k: Some(2),
                ..Default::default()
            },
        )
    }
}
