// SPDX-License-Identifier: Apache-2.0
//! Explicit parameter routing. Reuses existing optimizers; never guesses roles
//! from rank alone (embeddings and output heads are also matrices).
use alloc::{format, string::String, vec::Vec};
use core::marker::PhantomData;
use hashbrown::{HashMap, HashSet};
use ruda_model::{
    config::Config,
    module::{AutodiffModule, ModuleVisitor, Param, ParamId},
    record::{PrecisionSettings, Record},
    tensor::{Tensor, backend::AutodiffBackend},
};
use crate::{
    AdamW, AdamWConfig, GradientsParams, LearningRate, MultiGradientsParams,
    Optimizer, adaptor::OptimizerAdaptor, record::{AdaptorRecord, AdaptorRecordV1},
};
use super::{Muon, MuonConfig, MuonError};

type Manifest = Vec<(u64, Vec<usize>, bool, String)>;
type MuonRecords<B: AutodiffBackend> = HashMap<ParamId, AdaptorRecord<Muon<<B as AutodiffBackend>::InnerBackend>, B>>;
type AdamRecords<B: AutodiffBackend> = HashMap<ParamId, AdaptorRecord<AdamW, B>>;

/// Muon for explicitly selected hidden matrices and AdamW for the remainder.
///
/// `Optimizer::step(lr, ...)` uses `lr` for Muon and `lr * adamw_lr_ratio`
/// for AdamW. `try_step_with_lrs` accepts independent rates instead.
/// This is a high-level tensor optimizer, not the experimental fused AdamW API.
#[derive(Config, Debug)]
pub struct MuonAdamWConfig {
    /// Muon settings, with legacy numerical defaults preserved.
    #[config(default = "MuonConfig::new()")]
    muon: MuonConfig,
    /// Settings for all parameters not explicitly selected for Muon.
    #[config(default = "AdamWConfig::new()")]
    adamw: AdamWConfig,
    /// AdamW/Muon learning-rate ratio. 0.015 maps 0.02 to 0.0003.
    #[config(default = 0.015)]
    adamw_lr_ratio: f64,
}

impl MuonAdamWConfig {
    /// Validate configuration and resolve selected parameter IDs against a model.
    /// Do not select embeddings, output heads, normalization gains or biases.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self, module: &M, muon_parameters: &[ParamId],
    ) -> Result<MuonAdamW<M, B>, MuonError> {
        self.muon.validate()?;
        self.adamw.validate_hyperparameters().map_err(MuonError::InvalidConfig)?;
        valid_lr(self.adamw_lr_ratio)?;
        if muon_parameters.is_empty() { return Err(MuonError::EmptyMuonGroup); }
        let mut selected = HashSet::new();
        for id in muon_parameters {
            if !selected.insert(*id) { return Err(MuonError::DuplicateParameter(id.val())); }
        }
        let muon: OptimizerAdaptor<Muon<B::InnerBackend>, M, B> = self.muon.try_init()?;
        let manifest = inspect_module(module, &selected, muon.optim(), None, 0.0)?;
        Ok(MuonAdamW {
            muon, adamw: self.adamw.init(), selected, manifest,
            config_key: format!("muon-adamw-v1:{self:?}"),
            adamw_lr_ratio: self.adamw_lr_ratio,
        })
    }
}

/// Mixed optimizer with fixed, explicit parameter identities.
/// Missing gradients skip both momentum and decay for that parameter.
/// A tied parameter is updated once, by its `ParamId`.
#[derive(Clone)]
pub struct MuonAdamW<M: AutodiffModule<B>, B: AutodiffBackend> {
    muon: OptimizerAdaptor<Muon<B::InnerBackend>, M, B>,
    adamw: OptimizerAdaptor<AdamW, M, B>,
    selected: HashSet<ParamId>,
    manifest: Manifest,
    config_key: String,
    adamw_lr_ratio: f64,
}

/// Versioned optimizer record. Save the model record with it so ParamIds survive.
/// Configuration and routing are checked on load; lower-precision record settings
/// can round momentum, so use full-precision records for continuation comparisons.
#[derive(Clone)]
pub struct MuonAdamWRecord<B: AutodiffBackend> {
    version: u32,
    config_key: String,
    manifest: Manifest,
    muon: MuonRecords<B>,
    adamw: AdamRecords<B>,
}

impl<B: AutodiffBackend> MuonAdamWRecord<B> {
    /// Number of hidden-matrix momentum states saved.
    pub fn muon_state_count(&self) -> usize { self.muon.len() }
    /// Number of auxiliary AdamW parameter states saved.
    pub fn adamw_state_count(&self) -> usize { self.adamw.len() }
}

impl<B: AutodiffBackend> Record<B> for MuonAdamWRecord<B> {
    type Item<S: PrecisionSettings> = (
        u32, String, Manifest,
        <MuonRecords<B> as Record<B>>::Item<S>,
        <AdamRecords<B> as Record<B>>::Item<S>,
    );
    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.version, self.config_key, self.manifest,
         <MuonRecords<B> as Record<B>>::into_item::<S>(self.muon),
         <AdamRecords<B> as Record<B>>::into_item::<S>(self.adamw))
    }
    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            version: item.0, config_key: item.1, manifest: item.2,
            muon: <MuonRecords<B> as Record<B>>::from_item::<S>(item.3, device),
            adamw: <AdamRecords<B> as Record<B>>::from_item::<S>(item.4, device),
        }
    }
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> MuonAdamW<M, B> {
    /// Number of distinct parameters routed to Muon, not number of matrix elements.
    pub fn muon_parameter_count(&self) -> usize { self.selected.len() }

    /// Update with independent learning rates. Metadata for both groups is
    /// checked before either group submits an update. Device runtime errors are
    /// still asynchronous and this method is NOT a two-phase device transaction.
    pub fn try_step_with_lrs(
        &mut self, muon_lr: LearningRate, adamw_lr: LearningRate,
        module: M, grads: GradientsParams,
    ) -> Result<M, MuonError> {
        valid_lr(muon_lr)?;
        valid_lr(adamw_lr)?;
        let manifest = inspect_module(&module, &self.selected, self.muon.optim(), Some(&grads), muon_lr)?;
        if manifest != self.manifest { return Err(MuonError::ModelChanged); }
        let mut split = SplitGradients::<B> {
            selected: &self.selected, source: grads,
            muon: GradientsParams::new(), adamw: GradientsParams::new(),
            seen: HashSet::new(), backend: PhantomData,
        };
        module.visit(&mut split);
        if !split.source.is_empty() { return Err(MuonError::UnusedGradients); }
        let module = self.muon.step(muon_lr, module, split.muon);
        Ok(self.adamw.step(adamw_lr, module, split.adamw))
    }

    /// An explicit all-group skip makes no optimizer update and changes no state.
    /// The caller must decide skip consistently across all replicas and must
    /// unscale/check finite gradients before a non-skipped update.
    pub fn try_step_or_skip(
        &mut self, muon_lr: LearningRate, adamw_lr: LearningRate,
        module: M, grads: GradientsParams, skip_update: bool,
    ) -> Result<M, MuonError> {
        if skip_update { return Ok(module); }
        self.try_step_with_lrs(muon_lr, adamw_lr, module, grads)
    }

    /// Restore only compatible grouping/configuration. Does not silently move a
    /// momentum buffer between SGD and EMA conventions or between parameter roles.
    pub fn try_load_record(mut self, record: MuonAdamWRecord<B>) -> Result<Self, MuonError> {
        if record.version != 1 || record.config_key != self.config_key || record.manifest != self.manifest {
            return Err(MuonError::IncompatibleRecord);
        }
        let known: HashSet<_> = self.manifest.iter().map(|(id, _, _, _)| ParamId::from(*id)).collect();
        if record.muon.keys().any(|id| !self.selected.contains(id))
            || record.adamw.keys().any(|id| !known.contains(id) || self.selected.contains(id)) {
            return Err(MuonError::IncompatibleRecord);
        }
        for (id, state) in &record.muon {
            let expected = self.manifest.iter().find(|entry| entry.0 == id.val())
                .ok_or(MuonError::IncompatibleRecord)?;
            match state {
                AdaptorRecord::V1(AdaptorRecordV1::Rank2(state)) => {
                    let v = state.momentum.velocity();
                    if v.shape().to_vec() != expected.1 || format!("{:?}", v.dtype()) != expected.3 {
                        return Err(MuonError::IncompatibleRecord);
                    }
                }
                _ => return Err(MuonError::IncompatibleRecord),
            }
        }
        for (id, state) in &record.adamw {
            let expected = self.manifest.iter().find(|entry| entry.0 == id.val())
                .ok_or(MuonError::IncompatibleRecord)?;
            macro_rules! check_adam {
                ($state:expr) => {{
                    let m = &$state.momentum;
                    if m.time == 0 || m.moment_1.shape().to_vec() != expected.1
                        || m.moment_2.shape().to_vec() != expected.1
                        || format!("{:?}", m.moment_1.dtype()) != expected.3
                        || format!("{:?}", m.moment_2.dtype()) != expected.3
                        || m.max_moment_2.as_ref().is_some_and(|v|
                            v.shape().to_vec() != expected.1 || format!("{:?}", v.dtype()) != expected.3) {
                        return Err(MuonError::IncompatibleRecord);
                    }
                }};
            }
            match state {
                AdaptorRecord::V1(state) => match state {
                    AdaptorRecordV1::Rank0(v) => check_adam!(v),
                    AdaptorRecordV1::Rank1(v) => check_adam!(v),
                    AdaptorRecordV1::Rank2(v) => check_adam!(v),
                    AdaptorRecordV1::Rank3(v) => check_adam!(v),
                    AdaptorRecordV1::Rank4(v) => check_adam!(v),
                    AdaptorRecordV1::Rank5(v) => check_adam!(v),
                    AdaptorRecordV1::Rank6(v) => check_adam!(v),
                    AdaptorRecordV1::Rank7(v) => check_adam!(v),
                    AdaptorRecordV1::Rank8(v) => check_adam!(v),
                },
            }
        }
        self.muon = self.muon.load_record(record.muon);
        self.adamw = self.adamw.load_record(record.adamw);
        Ok(self)
    }
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> Optimizer<M, B> for MuonAdamW<M, B> {
    type Record = MuonAdamWRecord<B>;
    fn step(&mut self, lr: LearningRate, module: M, grads: GradientsParams) -> M {
        self.try_step_with_lrs(lr, lr * self.adamw_lr_ratio, module, grads)
            .unwrap_or_else(|error| panic!("{error}"))
    }
    fn step_multi(&mut self, _lr: LearningRate, _module: M, _grads: MultiGradientsParams) -> M {
        // Orthogonalization is nonlinear: orthogonalizing each shard separately
        // is not equivalent to orthogonalizing the full matrix.
        panic!("{}", MuonError::UnsupportedDistributed)
    }
    fn to_record(&self) -> Self::Record {
        MuonAdamWRecord {
            version: 1, config_key: self.config_key.clone(), manifest: self.manifest.clone(),
            muon: self.muon.to_record(), adamw: self.adamw.to_record(),
        }
    }
    fn load_record(self, record: Self::Record) -> Self {
        self.try_load_record(record).unwrap_or_else(|error| panic!("{error}"))
    }
}

fn valid_lr(lr: f64) -> Result<(), MuonError> {
    if !lr.is_finite() || lr < 0.0 || !(lr as f32).is_finite() {
        Err(MuonError::InvalidConfig("learning rates/ratios must be finite, nonnegative and FP32-representable"))
    } else { Ok(()) }
}

struct Inspect<'a, B: AutodiffBackend> {
    selected: &'a HashSet<ParamId>,
    muon: &'a Muon<B::InnerBackend>,
    grads: Option<&'a GradientsParams>,
    lr: f64,
    entries: HashMap<ParamId, (Vec<usize>, bool, String)>,
    seen_grads: usize,
    error: Option<MuonError>,
}

fn inspect_module<B: AutodiffBackend, M: AutodiffModule<B>>(
    module: &M, selected: &HashSet<ParamId>, muon: &Muon<B::InnerBackend>,
    grads: Option<&GradientsParams>, lr: f64,
) -> Result<Manifest, MuonError> {
    let mut inspect = Inspect { selected, muon, grads, lr, entries: HashMap::new(), seen_grads: 0, error: None };
    module.visit(&mut inspect);
    if let Some(error) = inspect.error { return Err(error); }
    for id in selected {
        if !inspect.entries.contains_key(id) { return Err(MuonError::UnknownParameter(id.val())); }
    }
    if grads.is_some_and(|g| g.len() != inspect.seen_grads) { return Err(MuonError::UnusedGradients); }
    let mut manifest: Manifest = inspect.entries.into_iter()
        .map(|(id, (shape, _, dtype))| (id.val(), shape, selected.contains(&id), dtype)).collect();
    manifest.sort_by_key(|entry| entry.0);
    Ok(manifest)
}

impl<B: AutodiffBackend> ModuleVisitor<B> for Inspect<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if self.error.is_some() { return; }
        let tensor = param.val();
        let shape = tensor.shape().to_vec();
        let trainable = param.is_require_grad();
        let dtype = format!("{:?}", tensor.dtype());
        if let Some(prior) = self.entries.get(&param.id) {
            if prior != &(shape, trainable, dtype) { self.error = Some(MuonError::ModelChanged); }
            return;
        }
        self.entries.insert(param.id, (shape, trainable, dtype));
        if D > 8 { self.error = Some(MuonError::InvalidConfig("optimizer records support at most rank 8")); return; }
        #[cfg(feature = "distributed")]
        if tensor.is_distributed() { self.error = Some(MuonError::UnsupportedDistributed); return; }
        let selected = self.selected.contains(&param.id);
        if selected && !trainable { self.error = Some(MuonError::FrozenParameter(param.id.val())); return; }
        let inner = tensor.inner();
        if selected {
            // Also validates rank/empty shape/stable-normalization dtype at init.
            if let Err(error) = self.muon.validate_step(self.lr, &inner, &inner, None) {
                self.error = Some(error); return;
            }
        }
        if !trainable { return; }
        if let Some(grad) = self.grads.and_then(|g| g.get::<B::InnerBackend, D>(param.id)) {
            self.seen_grads += 1;
            if inner.shape() != grad.shape() { self.error = Some(MuonError::ShapeMismatch("gradient")); return; }
            if inner.dtype() != grad.dtype() { self.error = Some(MuonError::DTypeMismatch("gradient")); return; }
            if inner.device() != grad.device() { self.error = Some(MuonError::DeviceMismatch("gradient")); }
        }
    }
}

struct SplitGradients<'a, B: AutodiffBackend> {
    selected: &'a HashSet<ParamId>,
    source: GradientsParams,
    muon: GradientsParams,
    adamw: GradientsParams,
    seen: HashSet<ParamId>,
    backend: PhantomData<B>,
}
impl<B: AutodiffBackend> ModuleVisitor<B> for SplitGradients<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if !param.is_require_grad() || !self.seen.insert(param.id) { return; }
        if let Some(grad) = self.source.remove::<B::InnerBackend, D>(param.id) {
            if self.selected.contains(&param.id) { self.muon.register(param.id, grad); }
            else { self.adamw.register(param.id, grad); }
        }
    }
}

#[cfg(test)]
mod tests;
