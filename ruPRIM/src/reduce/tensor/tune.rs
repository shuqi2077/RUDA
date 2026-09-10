#![allow(missing_docs)]

use super::SumAutotuneKey;
use super::key::SumTuneKey;
use ruda_kernel::dsl::{Runtime, RudaTuneId as TuneId};
use ruda_kernel::tensor::RudaTensor;
use ruda::runtime::{
    client::ComputeClient,
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup},
};
use crate::reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy, tune_key::ReduceAutotuneKey},
    routines::{BlueprintStrategy, ruda::RudaStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};

/// Executes autotune on reduce operations.
pub fn autotune_reduce<R: Runtime>(
    client: &ComputeClient<R>,
    input: RudaTensor<R>,
    output: RudaTensor<R>,
    axis: usize,
    config: ReduceOperationConfig,
    dtypes: ReduceDtypes,
) {
    use reduce_ops::*;

    static TUNER: LocalTuner<ReduceAutotuneKey, TuneId> = LocalTuner::new("ruda_tensor_device::kernel::reduce::tune-reduce-dim");

    let tunables = TUNER.init(|| {
        const PRIORITY_MAX: i8 = 2;
        const PRIORITY_MIN: i8 = 1;
        const PRIORITY_SKIP: i8 = -1;

        let mut set = TunableSet::new(create_key::<R>, reduce_input_gen::<R>);

        let default_group =
            TuneGroup::<ReduceAutotuneKey>::new("default_reduce", |_key| PRIORITY_MAX);
        let vectorized_parallel_group =
            TuneGroup::<ReduceAutotuneKey>::new("vectorized_parallel_reduce", |key| {
                if key.axis_is_contiguous {
                    PRIORITY_MAX
                } else {
                    // We disable the tunable with the setting [vector_size.parallel_output_vectorization]
                    // when the reduce isn't parallel, since it would duplicate tunables.
                    PRIORITY_SKIP
                }
            });

        enum ReduceProps {
            GreatWithLowReduceCount,
            GreatWithHighReduceCount,
            Balanced,
        }

        for (vectorization, vector_size_ident) in [
            (
                VectorizationStrategy {
                    parallel_output_vectorization: true,
                },
                "_vectorized_parallel_reduce",
            ),
            (
                VectorizationStrategy {
                    parallel_output_vectorization: false,
                },
                "",
            ),
        ] {
            for (name, routine, props) in [
                (
                    "unit",
                    RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
                    ReduceProps::GreatWithHighReduceCount,
                ),
                (
                    "plane",
                    RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
                        independent: true,
                    })),
                    ReduceProps::Balanced,
                ),
                (
                    "ruda",
                    RoutineStrategy::Ruda(BlueprintStrategy::Inferred(RudaStrategy {
                        use_planes: true,
                    })),
                    ReduceProps::GreatWithLowReduceCount,
                ),
            ] {
                let name = format!("{name}{vector_size_ident}");
                let mut tunable = Tunable::new(
                    &name,
                    move |(input, output, axis, config, dtypes): (
                        RudaTensor<R>,
                        RudaTensor<R>,
                        usize,
                        ReduceOperationConfig,
                        ReduceDtypes,
                    )| {
                        let strategy = ReduceStrategy {
                            routine: routine.clone(),
                            vectorization,
                        };
                        crate::reduce::reduce::<R>(
                            &output.client,
                            input.binding(),
                            output.clone().binding(),
                            axis,
                            strategy,
                            config,
                            dtypes,
                        )
                        .map_err(|e| format!("{e}"))
                    },
                );
                if vectorization.parallel_output_vectorization {
                    tunable = tunable.group(&vectorized_parallel_group, |_| PRIORITY_MAX);
                }

                tunable = tunable.group(&default_group, move |key| match props {
                    ReduceProps::GreatWithLowReduceCount => {
                        if key.vector_count < 128 {
                            PRIORITY_MAX
                        } else {
                            // When you have a high level of vector to reduce, it is normally
                            // better to use another routine.
                            PRIORITY_MIN
                        }
                    }
                    ReduceProps::GreatWithHighReduceCount => {
                        if key.vector_count > 64 {
                            PRIORITY_MAX
                        } else {
                            // Bellow 64 it is normally better to use another routine
                            PRIORITY_MIN
                        }
                    }
                    ReduceProps::Balanced => PRIORITY_MAX,
                });
                set = set.with(tunable);
            }
        }

        set
    });

    TUNER.execute(
        &TuneId::new(&input.client, &input.device),
        client,
        tunables,
        (input, output, axis, config, dtypes),
    );
}

pub(crate) fn create_key<Run: Runtime>(
    (input, output, axis, _config, dtypes): &(
        RudaTensor<Run>,
        RudaTensor<Run>,
        usize,
        ReduceOperationConfig,
        ReduceDtypes,
    ),
) -> ReduceAutotuneKey {
    let elem_input = input.dtype.into();
    let elem_output = output.dtype.into();
    let elem_acc = dtypes.accumulation.elem_type();

    ReduceAutotuneKey::generate(
        elem_input,
        elem_output,
        elem_acc,
        input.meta.shape(),
        input.meta.strides()[*axis] == 1,
        *axis,
    )
}

mod reduce_ops {
    #![allow(missing_docs)]

    use crate::reduce::ReduceDtypes;

    use super::*;

    pub(crate) fn reduce_input_gen<Run: Runtime>(
        _key: &ReduceAutotuneKey,
        (input, output, dim, config, dtypes): &(
            RudaTensor<Run>,
            RudaTensor<Run>,
            usize,
            ReduceOperationConfig,
            ReduceDtypes,
        ),
    ) -> (
        RudaTensor<Run>,
        RudaTensor<Run>,
        usize,
        ReduceOperationConfig,
        ReduceDtypes,
    ) {
        (input.clone(), output.copy(), *dim, *config, *dtypes)
    }
}

/// Executes autotune on reduce operations.
#[cfg(feature = "tensor-reduce-autotune")]
pub fn autotune_sum<R: Runtime>(
    client: &ComputeClient<R>,
    input: RudaTensor<R>,
) -> RudaTensor<R> {
    use sum_ops::*;

    static TUNER: LocalTuner<SumTuneKey, TuneId> = LocalTuner::new("ruda_tensor_device::kernel::reduce::tune-autotune-sum");

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key_sum::<R>, sum_input_gen::<R>)
            .with(Tunable::new("sum_chained", sum_chained::<R>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 1>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 2>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 4>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 8>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 16>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 32>))
            .with(Tunable::new("sum_one_shot", sum_one_shot::<R, 64>))
    });

    TUNER.execute(
        &TuneId::new(&input.client, &input.device),
        client,
        tunables,
        input,
    )
}

pub(crate) fn create_key_sum<Run: Runtime>(input: &RudaTensor<Run>) -> SumTuneKey {
    SumTuneKey::Sum(SumAutotuneKey::generate(input))
}

impl SumAutotuneKey {
    #[allow(unused)]
    pub(crate) fn generate<Run: Runtime>(input: &RudaTensor<Run>) -> Self {
        let dtype = input.dtype;
        let length = input.meta.num_elements();
        Self::new(dtype, length)
    }
}
mod sum_ops {
    #![allow(missing_docs)]
    use ruda_kernel::tensor::initialization::zeros_client;

    use super::*;

    pub(crate) fn sum_input_gen<Run: Runtime>(
        _key: &SumTuneKey,
        input: &RudaTensor<Run>,
    ) -> RudaTensor<Run> {
        input.clone()
    }

    pub(crate) fn sum_one_shot<Run: Runtime, const C: u32>(
        input: RudaTensor<Run>,
    ) -> Result<RudaTensor<Run>, String> {
        let client = input.client.clone();
        let device = input.device.clone();
        let output = zeros_client(client.clone(), device, [1].into(), input.dtype);
        let dtype = input.dtype;

        crate::reduce::shared_sum::<Run>(
            &output.client,
            input.binding(),
            output.clone().binding(),
            C,
            dtype.into(),
        )
        .map_err(|e| e.to_string())
        .map(|_| output)
    }

    #[cfg(feature = "tensor-reduce-autotune")]
    pub(crate) fn sum_chained<Run: Runtime>(
        input: RudaTensor<Run>,
    ) -> Result<RudaTensor<Run>, String> {
        crate::reduce::tensor::reduce::<Run>(
            input,
            None,
            crate::reduce::tensor::KernelReduceStrategy::Autotune,
            crate::reduce::components::instructions::ReduceOperationConfig::Sum,
        )
        .map_err(|e| e.to_string())
    }
}
