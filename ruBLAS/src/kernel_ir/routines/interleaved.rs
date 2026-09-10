use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::ir::features::MmaConfig;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::client::ComputeClient;
use ruda_kernel::tiling::{
    ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation},
    tile::Strided,
};
use std::{fmt::Display, marker::PhantomData};

use crate::kernel_ir::definition::{
    RudaMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes,
    MultiRowStrategy, TilingBlueprint, TilingScheme, adjust_dtypes,
};
use crate::kernel_ir::{
    components::{
        batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
        global::{
            PlaneWriterFamily,
            read::{FullLoadingStrategy, sync_full_cyclic::SyncFullCyclicLoading},
            single_stage::simple::SimpleMatmulFamily,
        },
        stage::{ColMajorTilingOrder, PartitionBuffering, PlaneMatmulFamily, RowMajorTilingOrder},
        tile::TileMatmulKind,
    },
    routines::{
        Routine,
        selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane},
    },
};
use crate::kernel_ir::{
    routines::ExpandInfo,
    routines::{BlueprintStrategy, DeviceSettings, LaunchInfo},
    {components::batch::BatchMatmulFamily, launch::RuntimeConfig},
};

/// Plane accelerated single stage matmul with configurable readers (default to cyclic)
pub struct InterleavedAlgorithm<
    LL = SyncFullCyclicLoading<ColMajorTilingOrder>,
    RL = SyncFullCyclicLoading<RowMajorTilingOrder>,
    AL = SyncFullCyclicLoading<RowMajorTilingOrder>,
> {
    pub _ll: PhantomData<LL>,
    pub _rl: PhantomData<RL>,
    pub _al: PhantomData<AL>,
}

#[derive(Default, Debug, Clone)]
pub struct InterleavedArgs {
    // Uses an optimized multi rows strategy.
    pub multi_rows: bool,
}

impl Display for InterleavedArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.multi_rows { "_multi_rows" } else { "" })
    }
}

impl<LL, RL, AL, RC> Routine<RC> for InterleavedAlgorithm<LL, RL, AL>
where
    RC: RuntimeConfig,
    LL: FullLoadingStrategy<RC, TileKind = Strided>,
    RL: FullLoadingStrategy<RC, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
    AL: FullLoadingStrategy<RC, TileKind = Strided, SyncStrategy = LL::SyncStrategy>,
{
    type Strategy = InterleavedArgs;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        SimpleMatmulFamily<
            PlaneMatmulFamily<LL::Stage, RL::Stage, Option<AL::Stage>>,
            RC,
            LL,
            RL,
            AL,
            PlaneWriterFamily,
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
        let tile_matmul = TileMatmulKind::Interleaved;

        if tile_matmul.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let client = &device_settings.client;
        let (blueprint, dtypes) = match strategy {
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

    fn launch<MA: crate::kernel_ir::launch::MatmulArgs<Config = RC>, R: Runtime>(
        client: &ComputeClient<R>,
        ruda_dim: ruda_kernel::dsl::RudaDim,
        ruda_count: ruda_kernel::dsl::RudaCount,
        address_type: ruda_kernel::dsl::prelude::AddressType,
        input: crate::kernel_ir::launch::InputRuntimeArg<MA, R>,
        output: crate::kernel_ir::launch::OutputRuntimeArg<MA, R>,
        config: crate::kernel_ir::launch::ConfigRuntimeArg<MA, R>,
        ruda_count_input: RudaMappingLaunch<R>,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        unsafe {
            Self::BatchMatmul::launch_unchecked::<MA, R>(
                client,
                ruda_dim,
                ruda_count,
                address_type,
                input,
                output,
                config,
                ruda_count_input,
                blueprint,
                dtypes,
                vector_sizes,
            )?
        }
        Ok(())
    }

    fn num_stages() -> crate::kernel_ir::components::stage::NumStages {
        Self::BatchMatmul::num_stages()
    }

    fn device_settings<R: Runtime>(
        client: &ComputeClient<R>,
        vector_sizes: MatmulVectorSizes,
    ) -> DeviceSettings<R> {
        // Sometimes the GPU doesn't support plane instructions and doesn't report the
        // plane size, but we can still execute algorithms that don't use plane instructions.
        //
        // In this case, we set a plane size for the selector to work, defaulting to 32 as it
        // is a common plane size.
        let plane_dim = match client.properties().hardware.plane_size_max {
            0 => 32,
            plane_dim => plane_dim,
        };

        DeviceSettings {
            client: client.clone(),
            plane_dim,
            vector_sizes,
            max_ruda_count: client.properties().hardware.max_ruda_count,
        }
    }

    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        Self::BatchMatmul::validate_blueprint(client, blueprint, problem, dtypes, vector_sizes)
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
            TilingBlueprint::builder(
                TileMatmulKind::Interleaved,
                tiling_scheme,
                plane_dim,
                problem,
            )
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
            TilingBlueprint::builder(
                TileMatmulKind::Interleaved,
                tiling_scheme,
                plane_dim,
                problem,
            )
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
