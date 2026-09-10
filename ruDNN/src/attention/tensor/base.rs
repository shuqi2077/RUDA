use ruda_kernel::{dsl::Runtime, tensor::{RudaTensor, allocation::empty_device_dtype}};
use ruda_core::tensor::{DType, Shape, spatial::AttentionModuleOptions};
use crate::attention::fallback::attention_fallback;
use super::{attention_autotune, ops::RudaAttentionOps};
use crate::attention::kernel_ir::launch;
use crate::attention::kernel_ir::{
    definition::{
        AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions, AttentionSetupError,
    },
    routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};

#[derive(Debug)]
/// Strategy used to select which attention implementation to run.
pub enum AttentionStrategy {
    /// Flash Attention using accelerated inner matmuls.
    FlashBlackboxAccelerated(BlackboxAcceleratedStrategy),

    /// Flash Attention using unit inner matmuls.
    FlashUnit,

    /// Fallback implementation using multiple separate kernels.
    Fallback,

    /// Automatically benchmark and select the best strategy at runtime.
    #[cfg(feature = "tensor-attention-autotune")]
    Autotune,
}

impl Default for AttentionStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "tensor-attention-autotune")]
        return AttentionStrategy::Autotune;

        // if autotune is disabled, default to fallback to make sure it runs
        #[cfg(not(feature = "tensor-attention-autotune"))]
        AttentionStrategy::Fallback
    }
}

#[allow(clippy::too_many_arguments)]
/// Launch an attention kernel with given strategy
pub fn attention<R: Runtime>(
    query: RudaTensor<R>,
    key: RudaTensor<R>,
    value: RudaTensor<R>,
    mask: Option<RudaTensor<R>>,
    attn_bias: Option<RudaTensor<R>>,
    options: AttentionModuleOptions,
    strategy: AttentionStrategy,
) -> Result<RudaTensor<R>, AttentionSetupError> {
    match strategy {
        AttentionStrategy::FlashBlackboxAccelerated(strategy) => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            launch::Strategy::BlackboxAccelerated(
                crate::attention::kernel_ir::launch::BlueprintStrategy::Inferred(strategy),
            ),
        ),
        AttentionStrategy::FlashUnit => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            launch::Strategy::Unit(crate::attention::kernel_ir::launch::BlueprintStrategy::Inferred(())),
        ),
        AttentionStrategy::Fallback => Ok(attention_fallback::<RudaAttentionOps<R>>(
            query, key, value, mask, attn_bias, options,
        )),
        #[cfg(feature = "tensor-attention-autotune")]
        AttentionStrategy::Autotune => Ok(attention_autotune(
            query, key, value, mask, attn_bias, options,
        )),
    }
}

#[allow(clippy::too_many_arguments)]
/// Launch a flash attention kernel
pub fn flash_attention<R: Runtime>(
    query: RudaTensor<R>,
    key: RudaTensor<R>,
    value: RudaTensor<R>,
    mask: Option<RudaTensor<R>>,
    _attn_bias: Option<RudaTensor<R>>,
    options: AttentionModuleOptions,
    strategy: launch::Strategy,
) -> Result<RudaTensor<R>, AttentionSetupError> {
    let client = query.client.clone();
    let out = init_attention_output(&query, &value);

    let dtypes = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
        out: out.dtype.into(),
    };

    crate::attention::kernel_ir::launch::launch_ref::<R>(
        strategy,
        &client,
        query.binding(),
        key.binding(),
        value.binding(),
        mask.map(|mask| mask.binding()),
        out.clone().binding(),
        &dtypes,
        AttentionOptions {
            causal: options.is_causal,
            accumulator_precision: AccumulatorPrecision::Strict(ruda_kernel::dsl::ir::StorageType::Scalar(
                ruda_kernel::dsl::ir::ElemType::Float(ruda_kernel::dsl::ir::FloatKind::F32),
            )),
        },
    )?;

    Ok(out)
}

pub(crate) fn init_attention_output<R: Runtime>(
    query: &RudaTensor<R>,
    value: &RudaTensor<R>,
) -> RudaTensor<R> {
    let num_batches = query.meta.shape[0];
    let num_heads = query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let val_dim = value.meta.shape[3];
    let out_shape = Shape::new([num_batches, num_heads, seq_q, val_dim]);

    empty_device_dtype::<R>(
        query.client.clone(),
        query.device.clone(),
        out_shape,
        query.dtype,
    )
}
