use ruda_kernel::dsl as kernel_dsl;
use std::{fmt::Display, marker::PhantomData};

use ruda_kernel::dsl::Runtime;
use ruda_kernel::dsl::client::ComputeClient;
use ruda_kernel::dsl::ir::features::MmaConfig;
use ruda_kernel::tiling::{
    MatrixLayout,
    ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation},
};

use crate::kernel_ir::components::{
    global::read::sync_full_strided::SyncFullStridedLoading,
    stage::{PlaneMatmulFamily, StageFamily},
};
use crate::kernel_ir::definition::{
    MatmulProblem, MatmulSetupError, MatmulVectorSizes, SwizzleModes, TilingBlueprint,
    adjust_dtypes,
};
use crate::kernel_ir::{
    components::batch::{PartitionedBatchMatmulFamily, RowMajorGlobalPartitionMatmul},
    components::global::PlaneWriterFamily,
    components::global::read::FullLoadingStrategy,
    components::tile::TileMatmulKind,
};
use crate::kernel_ir::{
    components::global::{
        multi_stage::specialized::SpecializedMatmulFamily,
        read::{AsyncPartialLoadingStrategy, async_partial_tma::AsyncPartialTmaLoading},
    },
    definition::{MatmulElems, MultiRowStrategy, TilingScheme},
};
use crate::kernel_ir::{
    components::{
        global::{InputLoadFlow, LoadFlows},
        stage::PartitionBuffering,
    },
    routines::selector::select_swizzle,
};
use crate::kernel_ir::{
    launch::RuntimeConfig,
    routines::selector::{PlaneTilingBlueprintOptions, infer_blueprint_plane},
    routines::{BlueprintStrategy, DeviceSettings, LaunchInfo, TilingArgs, base},
    {components::batch::BatchMatmulFamily, routines::ExpandInfo},
};

/// Plane accelerated specialized matmul with TMA readers
pub struct SpecializedAlgorithm<L = AsyncPartialTmaLoading, AL = SyncFullStridedLoading> {
    pub _phantom: PhantomData<(L, AL)>,
}

#[derive(Clone)]
pub struct SpecializedStrategy {
    pub tile_matmul: TileMatmulKind,
}

impl Default for SpecializedStrategy {
    fn default() -> Self {
        Self {
            tile_matmul: TileMatmulKind::Cmma,
        }
    }
}

impl TilingArgs for SpecializedStrategy {
    fn set_tile_matmul(&mut self, kind: TileMatmulKind) {
        self.tile_matmul = kind;
    }
}

impl Display for SpecializedStrategy {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl From<()> for SpecializedStrategy {
    fn from(_value: ()) -> Self {
        Self::default()
    }
}

impl<RC, L, AL> base::Routine<RC> for SpecializedAlgorithm<L, AL>
where
    RC: RuntimeConfig,
    L: AsyncPartialLoadingStrategy<RC, Stage: StageFamily>,
    AL: FullLoadingStrategy<RC, Stage: StageFamily>,
{
    type Strategy = SpecializedStrategy;
    type BatchMatmul = PartitionedBatchMatmulFamily<
        RC,
        SpecializedMatmulFamily<
            PlaneMatmulFamily<L::Stage, L::Stage, Option<AL::Stage>>,
            RC,
            L,
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

        let tile_matmul = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.tile_matmul,
            BlueprintStrategy::Inferred(args) => args.tile_matmul,
        };

        if tile_matmul.can_cast_stage_element() {
            dtypes.adjust_stage_dtypes();
        }

        let (blueprint, dtypes) = match strategy {
            BlueprintStrategy::Forced(blueprint) => (blueprint.clone(), dtypes),
            BlueprintStrategy::Inferred(_) => infer_blueprint_plane::<R>(
                tile_matmul,
                &device_settings.client,
                problem,
                device_settings.plane_dim,
                dtypes,
                &device_settings.vector_sizes,
                PlaneTilingBlueprintOptions {
                    specialized: true,
                    multi_row_strategy: MultiRowStrategy::Adaptive {
                        minimum_stage_count: 8,
                    },
                    swizzled: tile_matmul.should_swizzle(&device_settings.client),
                    ..Default::default()
                },
            )?,
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        Self::validate_blueprint(
            &device_settings.client,
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

#[allow(unused, reason = "needs more tuning")]
fn infer_blueprint_specialized<R: Runtime>(
    tile_matmul: TileMatmulKind,
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    plane_dim: u32,
    swizzle: bool,
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

    let tiling_scheme = if supported(16, 8, 16) {
        TilingScheme::builder()
            .with_tile_size((16, 8, 16).into())
            .with_partition_size((1, 8, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap()
    } else if supported(16, 16, 16) {
        TilingScheme::builder()
            .with_tile_size((16, 16, 16).into())
            .with_partition_size((1, 4, 2).into())
            .with_stage_size((4, 1, 1).into())
            .build()
            .unwrap()
    } else {
        return infer_blueprint_plane::<R>(
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
        );
    };

    let hyperruda = HyperrudaBlueprint::builder()
        .global_order(GlobalOrder::SwizzleRow(4))
        .ruda_count_strategy(ruda_count_strategy)
        .build();

    let mut builder = TilingBlueprint::builder(tile_matmul, tiling_scheme, plane_dim, problem)
        .partition_buffering(PartitionBuffering::Single)
        .hyperruda_blueprint(hyperruda)
        .load_specialization_config(LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::LoadOnly,
        });

    if swizzle {
        let lhs_swizzle_dim = match problem.lhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_k() as usize,
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_m() as usize,
        };
        let rhs_swizzle_dim = match problem.rhs_layout {
            MatrixLayout::RowMajor => tiling_scheme.elements_per_stage_along_n() as usize,
            MatrixLayout::ColMajor => tiling_scheme.elements_per_stage_along_k() as usize,
        };

        let lhs = select_swizzle(lhs_swizzle_dim, dtypes.lhs_stage, vector_sizes.lhs);
        let rhs = select_swizzle(rhs_swizzle_dim, dtypes.rhs_stage, vector_sizes.rhs);
        builder = builder.shared_swizzle(SwizzleModes {
            lhs,
            rhs,
            ..Default::default()
        });
    }

    Ok((builder.build(), dtypes))
}
