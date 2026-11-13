//! Opt-in native sampling. Logits never leave the device; only the selected I32
//! token is read back. Request-local ChaCha produces one uploaded uniform per
//! step, without changing global device RNG state.
//!
//! FP32 reductions and nucleus boundary ties differ from the legacy CPU/F64
//! sampler. Existing generation APIs retain their original behavior.
use super::{
    CausalModel, CausalModelLimits, GenerationControl, GenerationError, GreedyGenerationConfig,
    SamplingConfig, SamplingGenerationConfig, TokenGenerationOutput, generate_with_selector_stream,
};
use chacha20::ChaCha12Rng;
use ruda::runtime::server::ComputeServer;
use ruda_core::{
    rand::{RngExt, SeedableRng, get_seeded_rng},
    tensor::{DType, Shape},
};
use ruda_kernel::tensor::RudaTensor;
use ruda_tensor::{
    DeviceOps,
    api::{Int, Tensor},
};
use ruda_tensor_device::{BoolElement, DeviceBackend, DeviceRuntime, FloatElement, IntElement};
use rurand::tensor::sampling::{FilterOrder, SamplingFilter, SamplingPlan};
use std::ops::ControlFlow;

pub struct DeviceTokenSampler<R: DeviceRuntime> {
    filter: SamplingFilter,
    rng: ChaCha12Rng,
    plan: Option<SamplingPlan<R>>,
}
impl<R: DeviceRuntime> DeviceTokenSampler<R> {
    pub fn new(config: SamplingConfig) -> Result<Self, GenerationError> {
        config.validate()?;
        let top_k =
            u32::try_from(config.top_k).map_err(|_| GenerationError("top-k exceeds U32".into()))?;
        Self::with_filter(
            SamplingFilter {
                temperature: config.temperature as f32,
                top_k,
                top_p: config.top_p as f32,
                min_p: 0.0,
            },
            config.seed,
        )
    }
    pub fn with_filter(filter: SamplingFilter, seed: Option<u64>) -> Result<Self, GenerationError> {
        filter
            .validate()
            .map_err(|e| GenerationError(e.to_string()))?;
        let rng = seed
            .map(ChaCha12Rng::seed_from_u64)
            .unwrap_or_else(|| ChaCha12Rng::from_rng(&mut get_seeded_rng()));
        Ok(Self {
            filter,
            rng,
            plan: None,
        })
    }
    /// Input must be a contiguous [1,vocabulary] device tensor. Returns device
    /// IDs; callers may keep them entirely on-device for batched scheduling.
    pub fn sample(&mut self, row: &RudaTensor<R>) -> Result<RudaTensor<R>, GenerationError> {
        if row.meta.num_dims() != 2 || row.meta.shape()[0] != 1 {
            return Err(GenerationError("expected [1,vocabulary] logits".into()));
        }
        if self
            .plan
            .as_ref()
            .is_none_or(|p| p.vocabulary() != row.meta.shape()[1])
        {
            self.plan = Some(
                SamplingPlan::new(row, &[self.filter], None, true, FilterOrder::TopKFirst)
                    .map_err(|e| GenerationError(e.to_string()))?,
            );
        }
        let u = self.rng.random::<f32>();
        let uniforms = RudaTensor::new_contiguous(
            row.client.clone(),
            row.device.clone(),
            Shape::from([1]),
            row.client.create_from_slice(&u.to_ne_bytes()),
            DType::F32,
        );
        Ok(self
            .plan
            .as_ref()
            .unwrap()
            .sample(row, &uniforms)
            .map_err(|e| GenerationError(e.to_string()))?
            .token_ids)
    }
}

/// Cached causal generation with native device filtering/selection, and no
/// vocabulary-sized readback. Supports temperature/top-k/top-p; penalty-aware
/// legacy generation is not silently redirected to this different sampler.
pub fn generate_causal_sampled_device<R, F, I, BT, M>(
    model: &M,
    limits: &CausalModelLimits,
    prompt: &[i32],
    config: &SamplingGenerationConfig,
    device: &R::Device,
) -> Result<TokenGenerationOutput, GenerationError>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
    M: CausalModel<DeviceBackend<R, F, I, BT>>,
{
    let mut sampler = DeviceTokenSampler::<R>::new(config.sampling)?;
    let generation = GreedyGenerationConfig {
        max_new_tokens: config.max_new_tokens,
        eos_token_ids: config.eos_token_ids.clone(),
    };
    Ok(generate_with_selector_stream(
        model,
        limits,
        prompt,
        &generation,
        &GenerationControl::default(),
        device,
        |_, logits: Tensor<DeviceBackend<R, F, I, BT>, 3>| {
            let [batch, sequence, vocabulary] = logits.dims();
            if batch != 1 || sequence == 0 || vocabulary == 0 {
                return Err(GenerationError("expected nonempty batch-one logits".into()));
            }
            let row = logits
                .slice([0..1, sequence - 1..sequence, 0..vocabulary])
                .reshape([1, vocabulary])
                .into_primitive()
                .tensor();
            // Framework tensors can have aligned/pitched singleton strides.
            // Pack only this final row, on-device, before flat-array sampling.
            let row = ruda_kernel::tensor::contiguous::into_contiguous(row);
            let ids = sampler.sample(&row)?;
            let data = Tensor::<DeviceBackend<R, F, I, BT>, 1, Int>::from_primitive(ids)
                .try_into_data()
                .map_err(|e| GenerationError(format!("cannot read sampled token: {e}")))?;
            let ids = data
                .to_vec::<i32>()
                .map_err(|e| GenerationError(format!("cannot decode sampled token: {e}")))?;
            let id = *ids
                .first()
                .ok_or_else(|| GenerationError("empty sampled token".into()))?;
            if id < 0 {
                return Err(GenerationError(
                    "invalid or fully masked sampling distribution".into(),
                ));
            }
            Ok(id)
        },
        |_| ControlFlow::Continue(()),
    )?
    .output)
}
