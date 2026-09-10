use ruda_kernel::{dsl::{Runtime, RudaTuneId}, tensor::RudaTensor};
use super::{AttentionStrategy, attention};
use ruda_core::tensor::spatial::AttentionModuleOptions;
use ruda_kernel::dsl::tune::{LocalTuner, Tunable, TunableSet, TuneGroup};
use crate::attention::kernel_ir::{
    launch::AttentionAutotuneKey, routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};

/// Executes autotune on attention operations
pub fn attention_autotune<R: Runtime>(
    query: RudaTensor<R>,
    key: RudaTensor<R>,
    value: RudaTensor<R>,
    mask: Option<RudaTensor<R>>,
    attn_bias: Option<RudaTensor<R>>,
    options: AttentionModuleOptions,
) -> RudaTensor<R> {
    let client = query.client.clone();

    static TUNER: LocalTuner<AttentionAutotuneKey, RudaTuneId> = LocalTuner::new("ruda_tensor_device::kernel::attention::tune");

    let tunables = TUNER.init(|| {
        const PRIORITY_MAX: i8 = 3;
        const PRIORITY_MIN: i8 = 0;

        let flash_attention =
            TuneGroup::<AttentionAutotuneKey>::new("flash_attention", |_key| PRIORITY_MAX);

        let fallback = TuneGroup::<AttentionAutotuneKey>::new("fallback", |key| {
            if key.seq_q > 4096 {
                PRIORITY_MIN
            } else {
                PRIORITY_MAX
            }
        });

        let mut set = TunableSet::new(create_key::<R>, input_gen::<R>);

        // First entry should always work, since it is considered the fallback.
        set = set.with(
            Tunable::new(
                "fallback",
                |(query, key, value, mask, attn_bias, options)| {
                    attention::<R>(
                        query,
                        key,
                        value,
                        mask,
                        attn_bias,
                        options,
                        AttentionStrategy::Fallback,
                    )
                    .map_err(|err| std::format!("{err:?}"))
                },
            )
            .group(&fallback, |_key| PRIORITY_MAX),
        );

        let seq_q = 1;
        let seq_kv = 1;
        for num_planes in [2, 4, 8] {
            let name = format!("blackbox_accelerated_{num_planes}_planes_p_{seq_q}-{seq_kv}");
            set = set.with(
                Tunable::new(
                    &name,
                    move |(query, key, value, mask, attn_bias, options)| {
                        attention::<R>(
                            query,
                            key,
                            value,
                            mask,
                            attn_bias,
                            options,
                            AttentionStrategy::FlashBlackboxAccelerated(
                                BlackboxAcceleratedStrategy {
                                    num_planes,
                                    seq_q,
                                    seq_kv,
                                },
                            ),
                        )
                        .map_err(|err| std::format!("{err:?}"))
                    },
                )
                .group(&flash_attention, |_key| PRIORITY_MAX),
            );
        }

        set = set.with(
            Tunable::new("unit", |(query, key, value, mask, attn_bias, options)| {
                attention::<R>(
                    query,
                    key,
                    value,
                    mask,
                    attn_bias,
                    options,
                    AttentionStrategy::FlashUnit,
                )
                .map_err(|err| std::format!("{err:?}"))
            })
            .group(&flash_attention, |_key| PRIORITY_MIN),
        );

        set
    });

    TUNER.execute(
        &RudaTuneId::new(&client, &query.device),
        &client,
        tunables,
        (query, key, value, mask, attn_bias, options),
    )
}

#[allow(clippy::type_complexity)]
fn create_key<R: Runtime>(
    (query, key, value, mask, _attn_bias, _options): &(
        RudaTensor<R>,
        RudaTensor<R>,
        RudaTensor<R>,
        Option<RudaTensor<R>>,
        Option<RudaTensor<R>>,
        AttentionModuleOptions,
    ),
) -> AttentionAutotuneKey {
    let total_batches = query.meta.shape[0] * query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let head_dim = query.meta.shape[3];
    let seq_kv = value.meta.shape[2];
    let val_dim = value.meta.shape[3];

    AttentionAutotuneKey::generate(
        query.dtype.into(),
        key.dtype.into(),
        value.dtype.into(),
        query.dtype.into(),
        total_batches,
        seq_q,
        head_dim,
        seq_kv,
        val_dim,
        mask.is_some(),
    )
}

#[allow(clippy::type_complexity)]
fn input_gen<R: Runtime>(
    _key: &AttentionAutotuneKey,
    (query, key, value, mask, attn_bias, options): &(
        RudaTensor<R>,
        RudaTensor<R>,
        RudaTensor<R>,
        Option<RudaTensor<R>>,
        Option<RudaTensor<R>>,
        AttentionModuleOptions,
    ),
) -> (
    RudaTensor<R>,
    RudaTensor<R>,
    RudaTensor<R>,
    Option<RudaTensor<R>>,
    Option<RudaTensor<R>>,
    AttentionModuleOptions,
) {
    (
        query.clone(),
        key.clone(),
        value.clone(),
        mask.clone(),
        attn_bias.clone(),
        *options,
    )
}
