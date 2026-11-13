#![allow(unsafe_code)]
//! Native device filtering and inverse-CDF sampling. No logits/probabilities are
//! read on the host. Supplied uniforms make RNG ownership explicit to callers.
use ruda_core::{
    device::Device,
    tensor::{DType, Shape},
};
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SamplingError(pub &'static str);
impl core::fmt::Display for SamplingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for SamplingError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingFilter {
    pub temperature: f32,
    /// Zero disables top-k. All values tied at the boundary are retained.
    pub top_k: u32,
    pub top_p: f32,
    /// Relative to the maximum weight, applied after top-k/top-p.
    pub min_p: f32,
}
impl Default for SamplingFilter {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        }
    }
}
impl SamplingFilter {
    pub fn validate(self) -> Result<(), SamplingError> {
        if !self.temperature.is_finite()
            || self.temperature <= 0.0
            || !self.top_p.is_finite()
            || self.top_p <= 0.0
            || self.top_p > 1.0
            || !self.min_p.is_finite()
            || !(0.0..=1.0).contains(&self.min_p)
        {
            return Err(SamplingError("invalid temperature/top-p/min-p"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterOrder {
    TopKFirst,
    Joint,
}

pub struct DeviceDistribution<R: Runtime> {
    pub probabilities: RudaTensor<R>,
    /// U32, one for a valid finite nonempty distribution and zero otherwise.
    pub valid: RudaTensor<R>,
}
pub struct SampledTokens<R: Runtime> {
    /// I32; invalid rows produce -1, never a silent token zero.
    pub token_ids: RudaTensor<R>,
    pub valid: RudaTensor<R>,
}

pub struct SpeculativeTokens<R: Runtime> {
    /// [batch, draft_length+1], padded with -1 after the emitted prefix.
    pub token_ids: RudaTensor<R>,
    /// Independently accepted draft positions, including after the first rejection.
    pub accepted: RudaTensor<R>,
    /// Accepted draft prefix actually emitted, excluding the replacement/bonus.
    pub emitted_draft: RudaTensor<R>,
    /// Total emitted tokens, equal to emitted_draft+1 for valid rows.
    pub emitted: RudaTensor<R>,
    pub valid: RudaTensor<R>,
}
impl<R: Runtime> SpeculativeTokens<R> {
    /// Add verification statistics to caller-owned I32/U32 running counters.
    /// Inputs may not alias and must be exclusive for this queued update.
    pub unsafe fn accumulate_counts(
        &self,
        accepted: &RudaTensor<R>,
        emitted_draft: &RudaTensor<R>,
    ) -> Result<(), SamplingError> {
        let rows = self.accepted.meta.num_elements();
        for t in [accepted, emitted_draft] {
            if !compatible(t, &self.accepted)
                || !t.is_contiguous()
                || !matches!(t.dtype, DType::I32 | DType::U32)
                || t.meta.shape()[..] != [rows]
            {
                return Err(SamplingError(
                    "speculative counters must be same-queue I32/U32 [batch]",
                ));
            }
        }
        if rows > 0 {
            let dim = RudaDim::new(accepted.client.properties(), rows);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&accepted.client, rows, dim);
            accumulate_counts::launch::<R>(
                &accepted.client,
                count,
                dim,
                self.accepted.clone().into_array_arg(),
                self.emitted_draft.clone().into_array_arg(),
                self.valid.clone().into_array_arg(),
                accepted.clone().into_array_arg(),
                emitted_draft.clone().into_array_arg(),
                rows as u32,
            );
        }
        Ok(())
    }
}
#[ruda(launch)]
pub fn accumulate_counts(
    accepted: &Array<u32>,
    emitted: &Array<u32>,
    valid: &Array<u32>,
    accepted_total: &mut Array<u32>,
    emitted_total: &mut Array<u32>,
    n: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize && valid[i] != 0 {
        accepted_total[i] += accepted[i];
        emitted_total[i] += emitted[i];
    }
}

/// Rejection-corrected chain verification, entirely on the device. Draft
/// probabilities are [batch,steps,vocabulary], target probabilities are
/// [batch,steps+1,vocabulary], draft IDs are I32 [batch,steps]. Both uniform
/// tensors are F32 [batch,steps+1] in [0,1). Inputs are normalized probabilities
/// (not logits). A rejection samples max(target-draft,0); an accepted whole
/// chain emits one bonus target token. RNG ownership remains with the caller.
pub fn chain_speculative_sampling<R: Runtime>(
    draft: &RudaTensor<R>,
    draft_ids: &RudaTensor<R>,
    target: &RudaTensor<R>,
    accept_uniforms: &RudaTensor<R>,
    sample_uniforms: &RudaTensor<R>,
) -> Result<SpeculativeTokens<R>, SamplingError> {
    execute_chain(
        draft,
        draft_ids,
        target,
        accept_uniforms,
        sample_uniforms,
        None,
    )
}
pub unsafe fn chain_speculative_sampling_into<R: Runtime>(
    draft: &RudaTensor<R>,
    draft_ids: &RudaTensor<R>,
    target: &RudaTensor<R>,
    accept_uniforms: &RudaTensor<R>,
    sample_uniforms: &RudaTensor<R>,
    output: &SpeculativeTokens<R>,
) -> Result<(), SamplingError> {
    execute_chain(
        draft,
        draft_ids,
        target,
        accept_uniforms,
        sample_uniforms,
        Some(output),
    )
    .map(|_| ())
}
pub fn allocate_speculative_output<R: Runtime>(
    like: &RudaTensor<R>,
    batch: usize,
    steps: usize,
) -> Result<SpeculativeTokens<R>, SamplingError> {
    if steps >= u32::MAX as usize
        || batch
            .checked_mul(steps + 1)
            .filter(|n| *n <= u32::MAX as usize)
            .is_none()
    {
        return Err(SamplingError("speculative output capacity exceeds U32"));
    }
    Ok(SpeculativeTokens {
        token_ids: alloc(like, [batch, steps + 1], DType::I32),
        accepted: alloc(like, [batch], DType::U32),
        emitted_draft: alloc(like, [batch], DType::U32),
        emitted: alloc(like, [batch], DType::U32),
        valid: alloc(like, [batch], DType::U32),
    })
}
fn execute_chain<R: Runtime>(
    draft: &RudaTensor<R>,
    draft_ids: &RudaTensor<R>,
    target: &RudaTensor<R>,
    accept_uniforms: &RudaTensor<R>,
    sample_uniforms: &RudaTensor<R>,
    destination: Option<&SpeculativeTokens<R>>,
) -> Result<SpeculativeTokens<R>, SamplingError> {
    if draft.meta.num_dims() != 3
        || target.meta.num_dims() != 3
        || draft.dtype != DType::F32
        || target.dtype != DType::F32
    {
        return Err(SamplingError(
            "speculative probabilities must be rank-three F32",
        ));
    }
    let (batch, steps, width) = (
        draft.meta.shape()[0],
        draft.meta.shape()[1],
        draft.meta.shape()[2],
    );
    if steps >= u32::MAX as usize
        || width == 0
        || width > i32::MAX as usize
        || target.meta.shape()[..] != [batch, steps + 1, width]
        || draft_ids.dtype != DType::I32
        || draft_ids.meta.shape()[..] != [batch, steps]
    {
        return Err(SamplingError(
            "invalid speculative probability/token shapes",
        ));
    }
    for t in [draft, draft_ids, target, accept_uniforms, sample_uniforms] {
        if !compatible(draft, t)
            || !t.is_contiguous()
            || t.qparams.is_some()
            || t.meta.num_elements() > u32::MAX as usize
        {
            return Err(SamplingError(
                "speculation requires same-queue contiguous U32-indexed tensors",
            ));
        }
    }
    for t in [accept_uniforms, sample_uniforms] {
        if t.dtype != DType::F32 || t.meta.shape()[..] != [batch, steps + 1] {
            return Err(SamplingError("invalid speculative uniforms"));
        }
    }
    let plane = draft.client.properties().hardware.plane_size_max;
    if !matches!(plane, 32 | 64)
        || batch > draft.client.properties().hardware.max_ruda_count.0 as usize
    {
        return Err(SamplingError(
            "speculation requires full 32/64-lane plane and legal grid",
        ));
    }
    let output = if let Some(out) = destination {
        SpeculativeTokens {
            token_ids: out.token_ids.clone(),
            accepted: out.accepted.clone(),
            emitted_draft: out.emitted_draft.clone(),
            emitted: out.emitted.clone(),
            valid: out.valid.clone(),
        }
    } else {
        allocate_speculative_output(draft, batch, steps)?
    };
    for (t, dtype, shape) in [
        (&output.token_ids, DType::I32, vec![batch, steps + 1]),
        (&output.accepted, DType::U32, vec![batch]),
        (&output.emitted_draft, DType::U32, vec![batch]),
        (&output.emitted, DType::U32, vec![batch]),
        (&output.valid, DType::U32, vec![batch]),
    ] {
        if !compatible(t, draft)
            || !t.is_contiguous()
            || t.qparams.is_some()
            || t.dtype != dtype
            || t.meta.shape()[..] != shape
        {
            return Err(SamplingError("preallocated speculative output mismatch"));
        }
    }
    let token_ids = output.token_ids;
    let accepted = output.accepted;
    let emitted_draft = output.emitted_draft;
    let emitted = output.emitted;
    let valid = output.valid;
    if batch != 0 {
        verify_chain::launch::<R>(
            &draft.client,
            RudaCount::Static(batch as u32, 1, 1),
            RudaDim::new_1d(plane),
            draft.clone().into_array_arg(),
            draft_ids.clone().into_array_arg(),
            target.clone().into_array_arg(),
            accept_uniforms.clone().into_array_arg(),
            sample_uniforms.clone().into_array_arg(),
            token_ids.clone().into_array_arg(),
            accepted.clone().into_array_arg(),
            emitted_draft.clone().into_array_arg(),
            emitted.clone().into_array_arg(),
            valid.clone().into_array_arg(),
            steps as u32,
            width as u32,
        );
    }
    Ok(SpeculativeTokens {
        token_ids,
        accepted,
        emitted_draft,
        emitted,
        valid,
    })
}

/// Immutable metadata; many output samples may select the same input row.
pub struct SamplingPlan<R: Runtime> {
    metadata: RudaTensor<R>,
    parameters: RudaTensor<R>,
    source_rows: usize,
    rows: usize,
    vocabulary: usize,
    logits: bool,
    order: FilterOrder,
}
fn alloc<R: Runtime>(like: &RudaTensor<R>, shape: impl Into<Shape>, dtype: DType) -> RudaTensor<R> {
    empty_device_contiguous_dtype(
        like.client.clone(),
        like.device.clone(),
        shape.into(),
        dtype,
    )
}
fn compatible<R: Runtime>(a: &RudaTensor<R>, b: &RudaTensor<R>) -> bool {
    a.device.to_id() == b.device.to_id() && a.client.same_execution_queue(&b.client)
}
impl<R: Runtime> SamplingPlan<R> {
    /// Device-resident [rows,2] U32 (source row, top-k) and [rows,3] F32
    /// (temperature, top-p, min-p). Values are reread on every graph replay;
    /// source IDs and parameter ranges are the caller's device-value contract.
    pub fn from_device(
        input: &RudaTensor<R>,
        metadata: &RudaTensor<R>,
        parameters: &RudaTensor<R>,
        logits: bool,
        order: FilterOrder,
    ) -> Result<Self, SamplingError> {
        if input.meta.num_dims() != 2
            || metadata.meta.num_dims() != 2
            || parameters.meta.num_dims() != 2
            || metadata.dtype != DType::U32
            || parameters.dtype != DType::F32
            || metadata.meta.shape()[1] != 2
            || parameters.meta.shape()[..] != [metadata.meta.shape()[0], 3]
            || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return Err(SamplingError("invalid device sampling plan tensors"));
        }
        for t in [input, metadata, parameters] {
            if !compatible(t, input)
                || !t.is_contiguous()
                || t.qparams.is_some()
                || t.meta.num_elements() > u32::MAX as usize
            {
                return Err(SamplingError(
                    "device sampling plan needs contiguous same-queue operands",
                ));
            }
        }
        let rows = metadata.meta.shape()[0];
        let vocabulary = input.meta.shape()[1];
        if vocabulary == 0
            || vocabulary > i32::MAX as usize
            || rows
                .checked_mul(vocabulary)
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
        {
            return Err(SamplingError("device sampling capacity exceeds U32"));
        }
        Ok(Self {
            metadata: metadata.clone(),
            parameters: parameters.clone(),
            source_rows: input.meta.shape()[0],
            rows,
            vocabulary,
            logits,
            order,
        })
    }
    /// Input is [source_rows,vocabulary]. A single filter broadcasts. Indices
    /// are CPU control metadata uploaded once, not device tensor readback.
    pub fn new(
        input: &RudaTensor<R>,
        filters: &[SamplingFilter],
        indices: Option<&[u32]>,
        logits: bool,
        order: FilterOrder,
    ) -> Result<Self, SamplingError> {
        if input.meta.num_dims() != 2
            || !input.is_contiguous()
            || input.qparams.is_some()
            || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return Err(SamplingError(
                "sampling requires contiguous rank-two float input",
            ));
        }
        let source_rows = input.meta.shape()[0];
        let vocabulary = input.meta.shape()[1];
        let rows = indices.map_or(source_rows, |v| v.len());
        if source_rows == 0
            || vocabulary == 0
            || vocabulary > i32::MAX as usize
            || input.meta.num_elements() > u32::MAX as usize
            || rows > u32::MAX as usize / 3
            || rows > u32::MAX as usize
            || source_rows > u32::MAX as usize
            || rows
                .checked_mul(vocabulary)
                .filter(|&n| n <= u32::MAX as usize)
                .is_none()
            || !(filters.len() == 1 || filters.len() == rows)
        {
            return Err(SamplingError(
                "sampling shape/filter count exceeds indexing contract",
            ));
        }
        let mut metadata = Vec::with_capacity(rows * 2);
        let mut parameters = Vec::with_capacity(rows * 3);
        for i in 0..rows {
            let source = indices.map_or(i as u32, |v| v[i]);
            if source as usize >= source_rows {
                return Err(SamplingError("sample source row is out of range"));
            }
            let f = filters[if filters.len() == 1 { 0 } else { i }];
            f.validate()?;
            metadata.extend([source, f.top_k]);
            parameters.extend([f.temperature, f.top_p, f.min_p]);
        }
        let metadata = RudaTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            Shape::from([rows, 2]),
            input
                .client
                .create_from_slice(bytemuck::cast_slice(&metadata)),
            DType::U32,
        );
        let parameters = RudaTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            Shape::from([rows, 3]),
            input
                .client
                .create_from_slice(bytemuck::cast_slice(&parameters)),
            DType::F32,
        );
        Ok(Self {
            metadata,
            parameters,
            source_rows,
            rows,
            vocabulary,
            logits,
            order,
        })
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn vocabulary(&self) -> usize {
        self.vocabulary
    }
    pub fn allocate_workspace(
        &self,
        input: &RudaTensor<R>,
    ) -> Result<SamplingWorkspace<R>, SamplingError> {
        if !compatible(input, &self.metadata)
            || input.meta.shape()[..] != [self.source_rows, self.vocabulary]
        {
            return Err(SamplingError(
                "sampling workspace input shape/queue mismatch",
            ));
        }
        Ok(SamplingWorkspace {
            distribution: DeviceDistribution {
                probabilities: alloc(input, [self.rows, self.vocabulary], DType::F32),
                valid: alloc(input, [self.rows], DType::U32),
            },
            output: SampledTokens {
                token_ids: alloc(input, [self.rows], DType::I32),
                valid: alloc(input, [self.rows], DType::U32),
            },
        })
    }
    pub fn prepare(&self, input: &RudaTensor<R>) -> Result<DeviceDistribution<R>, SamplingError> {
        self.prepare_impl(input, None)
    }
    pub unsafe fn prepare_into(
        &self,
        input: &RudaTensor<R>,
        destination: DeviceDistribution<R>,
    ) -> Result<DeviceDistribution<R>, SamplingError> {
        self.prepare_impl(input, Some(destination))
    }
    /// The retained workspace is exclusive and disjoint from inputs/uniforms.
    pub unsafe fn sample_into<'a>(
        &self,
        input: &RudaTensor<R>,
        uniforms: &RudaTensor<R>,
        workspace: &'a SamplingWorkspace<R>,
    ) -> Result<&'a SampledTokens<R>, SamplingError> {
        unsafe {
            self.prepare_into(
                input,
                DeviceDistribution {
                    probabilities: workspace.distribution.probabilities.clone(),
                    valid: workspace.distribution.valid.clone(),
                },
            )?;
            workspace
                .distribution
                .sample_into(uniforms, &workspace.output)?;
        }
        Ok(&workspace.output)
    }
    fn prepare_impl(
        &self,
        input: &RudaTensor<R>,
        destination: Option<DeviceDistribution<R>>,
    ) -> Result<DeviceDistribution<R>, SamplingError> {
        if !compatible(input, &self.metadata)
            || !input.is_contiguous()
            || input.qparams.is_some()
            || input.meta.shape()[..] != [self.source_rows, self.vocabulary]
            || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return Err(SamplingError("input no longer matches sampling plan"));
        }
        let plane = input.client.properties().hardware.plane_size_max;
        if !matches!(plane, 32 | 64)
            || self.rows > input.client.properties().hardware.max_ruda_count.0 as usize
        {
            return Err(SamplingError(
                "sampling requires a full 32/64-lane plane and legal grid",
            ));
        }
        let destination = destination.unwrap_or_else(|| DeviceDistribution {
            probabilities: alloc(input, [self.rows, self.vocabulary], DType::F32),
            valid: alloc(input, [self.rows], DType::U32),
        });
        for (t, dtype, shape) in [
            (
                &destination.probabilities,
                DType::F32,
                vec![self.rows, self.vocabulary],
            ),
            (&destination.valid, DType::U32, vec![self.rows]),
        ] {
            if !compatible(t, input)
                || !t.is_contiguous()
                || t.qparams.is_some()
                || t.dtype != dtype
                || t.meta.shape()[..] != shape
            {
                return Err(SamplingError(
                    "preallocated distribution shape/dtype/queue mismatch",
                ));
            }
        }
        let probabilities = destination.probabilities;
        let valid = destination.valid;
        if self.rows != 0 {
            weights::launch::<R>(
                &input.client,
                RudaCount::Static(self.rows as u32, 1, 1),
                RudaDim::new_1d(plane),
                input.clone().into_array_arg(),
                self.metadata.clone().into_array_arg(),
                self.parameters.clone().into_array_arg(),
                probabilities.clone().into_array_arg(),
                valid.clone().into_array_arg(),
                self.vocabulary as u32,
                self.logits,
                input.dtype.into(),
            );
            filter::launch::<R>(
                &input.client,
                RudaCount::Static(self.rows as u32, 1, 1),
                RudaDim::new_1d(plane),
                probabilities.clone().into_array_arg(),
                valid.clone().into_array_arg(),
                self.metadata.clone().into_array_arg(),
                self.parameters.clone().into_array_arg(),
                self.vocabulary as u32,
                self.order == FilterOrder::Joint,
            );
        }
        Ok(DeviceDistribution {
            probabilities,
            valid,
        })
    }
    pub fn sample(
        &self,
        input: &RudaTensor<R>,
        uniforms: &RudaTensor<R>,
    ) -> Result<SampledTokens<R>, SamplingError> {
        self.prepare(input)?.sample(uniforms)
    }
}

/// Fixed-address filtering and draw destinations for native graph replay.
pub struct SamplingWorkspace<R: Runtime> {
    pub distribution: DeviceDistribution<R>,
    pub output: SampledTokens<R>,
}
/// Public softmax and renormalization building blocks use the same native
/// filtering path as sampling; logits/probabilities never leave the device.
pub fn softmax<R: Runtime>(
    input: &RudaTensor<R>,
    temperature: f32,
) -> Result<DeviceDistribution<R>, SamplingError> {
    SamplingPlan::new(
        input,
        &[SamplingFilter {
            temperature,
            ..Default::default()
        }],
        None,
        true,
        FilterOrder::TopKFirst,
    )?
    .prepare(input)
}
pub fn top_k_renorm_probs<R: Runtime>(
    input: &RudaTensor<R>,
    k: u32,
) -> Result<DeviceDistribution<R>, SamplingError> {
    SamplingPlan::new(
        input,
        &[SamplingFilter {
            top_k: k,
            ..Default::default()
        }],
        None,
        false,
        FilterOrder::TopKFirst,
    )?
    .prepare(input)
}
pub fn top_p_renorm_probs<R: Runtime>(
    input: &RudaTensor<R>,
    p: f32,
) -> Result<DeviceDistribution<R>, SamplingError> {
    SamplingPlan::new(
        input,
        &[SamplingFilter {
            top_p: p,
            ..Default::default()
        }],
        None,
        false,
        FilterOrder::TopKFirst,
    )?
    .prepare(input)
}
/// Mask logits directly, before exponentiation. Selection therefore cannot lose
/// small finite logits through softmax underflow. Equal boundary values stay.
pub fn top_k_mask_logits<R: Runtime>(
    input: &RudaTensor<R>,
    k: &RudaTensor<R>,
) -> Result<RudaTensor<R>, SamplingError> {
    if input.meta.num_dims() != 2
        || !input.is_contiguous()
        || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
        || input.qparams.is_some()
        || !compatible(input, k)
        || k.dtype != DType::U32
        || !k.is_contiguous()
        || !(k.meta.num_elements() == 1 || k.meta.shape()[..] == [input.meta.shape()[0]])
        || input.meta.num_elements() > u32::MAX as usize
    {
        return Err(SamplingError(
            "mask logits needs dense [rows,V] and U32 scalar/per-row K",
        ));
    }
    let rows = input.meta.shape()[0];
    let width = input.meta.shape()[1];
    let lanes = input.client.properties().hardware.plane_size_max;
    if width == 0
        || !matches!(lanes, 32 | 64)
        || rows > input.client.properties().hardware.max_ruda_count.0 as usize
    {
        return Err(SamplingError("mask logits dimensions/grid invalid"));
    }
    let out = alloc(input, input.meta.shape().clone(), input.dtype);
    if rows > 0 {
        mask_logits::launch::<R>(
            &input.client,
            RudaCount::Static(rows as u32, 1, 1),
            RudaDim::new_1d(lanes),
            input.clone().into_array_arg(),
            k.clone().into_array_arg(),
            out.clone().into_array_arg(),
            width as u32,
            k.meta.num_elements() == 1,
            input.dtype.into(),
        );
    }
    Ok(out)
}
#[ruda]
fn signed_float_key(x: f32) -> u32 {
    let mut bits = u32::reinterpret(x);
    if x == 0.0 {
        bits = 0u32;
    }
    let mut key = bits ^ 0x80000000u32;
    if bits & 0x80000000u32 != 0 {
        key = !bits;
    }
    if x != x {
        key = u32::MAX;
    }
    key
}
#[ruda(launch)]
pub fn mask_logits<F: Float>(
    input: &Array<F>,
    ks: &Array<u32>,
    out: &mut Array<F>,
    width: u32,
    #[comptime] scalar: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let mut ki = row;
    if comptime!(scalar) {
        ki = 0;
    }
    let k = ks[ki];
    let mut remaining = k;
    let mut cutoff: u32 = 0u32;
    let mut mask: u32 = 0u32;
    if k > 0 && k < width {
        for b in 0u32..32 {
            let bit = 1u32 << (31 - b);
            let mut count: u32 = 0u32;
            let mut j = lane;
            while j < width as usize {
                let key = signed_float_key(f32::cast_from(input[row * width as usize + j]));
                if key & mask == cutoff && key & bit != 0 {
                    count += 1;
                }
                j += PLANE_DIM as usize;
            }
            count = plane_sum(count);
            if count >= remaining {
                cutoff |= bit;
            } else {
                remaining -= count;
            }
            mask |= bit;
        }
    }
    let mut j = lane;
    while j < width as usize {
        let x = f32::cast_from(input[row * width as usize + j]);
        let mut value = x;
        if signed_float_key(x) < cutoff {
            value = f32::NEG_INFINITY;
        }
        out[row * width as usize + j] = F::cast_from(value);
        j += PLANE_DIM as usize;
    }
}

/// Counter-based independent uniforms. `offset` is measured in scalar draws;
/// Philox4x32-10 assigns four consecutive draws to one counter block.
pub fn uniforms<R: Runtime>(
    like: &RudaTensor<R>,
    count: usize,
    seed: u64,
    offset: u64,
) -> Result<RudaTensor<R>, SamplingError> {
    if count > u32::MAX as usize || offset.checked_add(count as u64).is_none() {
        return Err(SamplingError("uniform counter range overflow"));
    }
    let out = alloc(like, [count], DType::F32);
    if count > 0 {
        let dim = RudaDim::new(like.client.properties(), count);
        let grid = kernel_dsl::calculate_ruda_count_elemwise(&like.client, count, dim);
        philox_uniforms::launch::<R>(
            &like.client,
            grid,
            dim,
            out.clone().into_array_arg(),
            count as u32,
            seed,
            offset,
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn philox_uniforms(out: &mut Array<f32>, n: u32, seed: u64, offset: u64) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let draw = offset + i as u64;
        let counter = draw / 4;
        let mut a = counter as u32;
        let mut b = (counter >> 32) as u32;
        let mut c: u32 = 0u32;
        let mut d: u32 = 0u32;
        let mut k0 = seed as u32;
        let mut k1 = (seed >> 32) as u32;
        for round in 0u32..10 {
            let p0 = a as u64 * 0xD2511F53u64;
            let p1 = c as u64 * 0xCD9E8D57u64;
            let na = (p1 >> 32) as u32 ^ b ^ k0;
            let nb = p1 as u32;
            let nc = (p0 >> 32) as u32 ^ d ^ k1;
            let nd = p0 as u32;
            a = na;
            b = nb;
            c = nc;
            d = nd;
            k0 = k0 + 0x9E3779B9u32;
            k1 = k1 + 0xBB67AE85u32;
        }
        let mut word = a;
        if draw % 4 == 1 {
            word = b;
        }
        if draw % 4 == 2 {
            word = c;
        }
        if draw % 4 == 3 {
            word = d;
        }
        out[i] = (word >> 8) as f32 * 0.000000059604644775390625;
    }
}
impl<R: Runtime> DeviceDistribution<R> {
    /// Uniforms must be F32 [rows] in [0,1). Invalid device values are reported
    /// in the returned validity tensor, without synchronizing the host.
    pub fn sample(&self, uniforms: &RudaTensor<R>) -> Result<SampledTokens<R>, SamplingError> {
        self.sample_impl(uniforms, None)
    }
    pub unsafe fn sample_into(
        &self,
        uniforms: &RudaTensor<R>,
        output: &SampledTokens<R>,
    ) -> Result<(), SamplingError> {
        self.sample_impl(
            uniforms,
            Some(SampledTokens {
                token_ids: output.token_ids.clone(),
                valid: output.valid.clone(),
            }),
        )
        .map(|_| ())
    }
    fn sample_impl(
        &self,
        uniforms: &RudaTensor<R>,
        destination: Option<SampledTokens<R>>,
    ) -> Result<SampledTokens<R>, SamplingError> {
        if self.probabilities.meta.num_dims() != 2
            || self.probabilities.dtype != DType::F32
            || !self.probabilities.is_contiguous()
            || self.probabilities.qparams.is_some()
            || self.probabilities.meta.num_elements() > u32::MAX as usize
        {
            return Err(SamplingError(
                "distribution must be contiguous F32 [rows,vocabulary]",
            ));
        }
        let rows = self.probabilities.meta.shape()[0];
        let vocabulary = self.probabilities.meta.shape()[1];
        let plane = uniforms.client.properties().hardware.plane_size_max;
        if vocabulary == 0
            || vocabulary > i32::MAX as usize
            || rows > uniforms.client.properties().hardware.max_ruda_count.0 as usize
            || !matches!(plane, 32 | 64)
            || !compatible(&self.probabilities, &self.valid)
            || self.valid.dtype != DType::U32
            || !self.valid.is_contiguous()
            || self.valid.meta.shape()[..] != [rows]
        {
            return Err(SamplingError(
                "invalid distribution validity tensor or launch dimensions",
            ));
        }
        if !compatible(&self.probabilities, uniforms)
            || uniforms.dtype != DType::F32
            || !uniforms.is_contiguous()
            || uniforms.meta.shape()[..] != [rows]
        {
            return Err(SamplingError(
                "uniforms must be same-queue contiguous F32 [rows]",
            ));
        }
        let destination = destination.unwrap_or_else(|| SampledTokens {
            token_ids: alloc(uniforms, [rows], DType::I32),
            valid: alloc(uniforms, [rows], DType::U32),
        });
        for (t, dtype) in [
            (&destination.token_ids, DType::I32),
            (&destination.valid, DType::U32),
        ] {
            if !compatible(t, uniforms)
                || !t.is_contiguous()
                || t.qparams.is_some()
                || t.dtype != dtype
                || t.meta.shape()[..] != [rows]
            {
                return Err(SamplingError(
                    "preallocated sample output shape/dtype/queue mismatch",
                ));
            }
        }
        let token_ids = destination.token_ids;
        let valid = destination.valid;
        if rows != 0 {
            let plane = uniforms.client.properties().hardware.plane_size_max;
            draw::launch::<R>(
                &uniforms.client,
                RudaCount::Static(rows as u32, 1, 1),
                RudaDim::new_1d(plane),
                self.probabilities.clone().into_array_arg(),
                self.valid.clone().into_array_arg(),
                uniforms.clone().into_array_arg(),
                token_ids.clone().into_array_arg(),
                valid.clone().into_array_arg(),
                vocabulary as u32,
            );
        }
        Ok(SampledTokens { token_ids, valid })
    }
}

#[ruda(launch)]
pub fn weights<F: Float>(
    input: &Array<F>,
    metadata: &Array<u32>,
    parameters: &Array<f32>,
    out: &mut Array<f32>,
    valid: &mut Array<u32>,
    width: u32,
    #[comptime] logits: bool,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let lanes = PLANE_DIM as usize;
    let base = metadata[row * 2] as usize * width as usize;
    let mut maximum = f32::NEG_INFINITY;
    let mut bad: u32 = 0u32;
    {
        let mut j = lane;
        while j < width as usize {
            let x = f32::cast_from(input[base + j]);
            if x != x || x == f32::INFINITY {
                bad = 1u32;
            }
            if comptime!(!logits) {
                if x < 0.0 {
                    bad = 1u32;
                }
            }
            maximum = f32::max(maximum, x);
            j += lanes;
        }
    }
    bad = plane_max(bad);
    maximum = plane_max(maximum);
    let mut total: f32 = 0.0f32;
    {
        let mut j = lane;
        while j < width as usize {
            let mut x = f32::cast_from(input[base + j]);
            if comptime!(logits) {
                if x == f32::NEG_INFINITY || maximum == f32::NEG_INFINITY {
                    x = 0.0;
                } else {
                    x = ((x - maximum) / parameters[row * 3]).exp();
                }
            }
            // Radix ordering assumes nonnegative F32 bit patterns; normalize -0.
            if x == 0.0 {
                x = 0.0;
            }
            if bad != 0 {
                x = 0.0;
            }
            out[row * width as usize + j] = x;
            total += x;
            j += lanes;
        }
    }
    total = plane_sum(total);
    if lane == 0 {
        valid[row] = u32::cast_from(bad == 0 && total > 0.0 && total != f32::INFINITY);
    }
}

// Radix selection is bounded by 32 passes, not a vocabulary-square sort.
// Positive F32 bit patterns have the same order as their numeric values.
#[ruda(launch)]
pub fn filter(
    weights: &mut Array<f32>,
    valid: &mut Array<u32>,
    metadata: &Array<u32>,
    parameters: &Array<f32>,
    width: u32,
    #[comptime] joint: bool,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let lanes = PLANE_DIM as usize;
    let base = row * width as usize;
    let k = metadata[row * 2 + 1];
    let p = parameters[row * 3 + 1];
    let min_p = parameters[row * 3 + 2];
    let mut cutoff: u32 = 0u32;
    let mut mask: u32 = 0u32;
    let mut remaining = k;
    if k > 0 && k < width {
        for b in 0u32..32 {
            let bit = 1u32 << (31 - b);
            let mut n: u32 = 0u32;
            {
                let mut j = lane;
                while j < width as usize {
                    let key = u32::reinterpret(weights[base + j]);
                    if key & mask == cutoff && key & bit != 0 {
                        n += 1;
                    }
                    j += lanes;
                }
            }
            n = plane_sum(n);
            if n >= remaining {
                cutoff |= bit;
            } else {
                remaining -= n;
            }
            mask |= bit;
        }
    }
    let mut total: f32 = 0.0f32;
    let mut maximum: f32 = 0.0f32;
    {
        let mut j = lane;
        while j < width as usize {
            let w = weights[base + j];
            maximum = f32::max(maximum, w);
            if comptime!(joint) {
                total += w;
            } else {
                if u32::reinterpret(w) >= cutoff {
                    total += w;
                }
            }
            j += lanes;
        }
    }
    total = plane_sum(total);
    maximum = plane_max(maximum);
    let mut nucleus: u32 = 0u32;
    if p < 1.0 && total > 0.0 {
        let mut needed = p * total;
        mask = 0;
        for b in 0u32..32 {
            let bit = 1u32 << (31 - b);
            let mut mass: f32 = 0.0f32;
            {
                let mut j = lane;
                while j < width as usize {
                    let w = weights[base + j];
                    let key = u32::reinterpret(w);
                    let mut eligible = true;
                    if comptime!(!joint) {
                        eligible = key >= cutoff;
                    }
                    if eligible && key & mask == nucleus && key & bit != 0 {
                        mass += w;
                    }
                    j += lanes;
                }
            }
            mass = plane_sum(mass);
            if mass > 0.0 && mass >= needed {
                nucleus |= bit;
            } else {
                needed -= mass;
            }
            mask |= bit;
        }
    }
    let boundary = u32::max(cutoff, nucleus);
    let mut retained: f32 = 0.0f32;
    {
        let mut j = lane;
        while j < width as usize {
            let w = weights[base + j];
            if u32::reinterpret(w) >= boundary && w >= min_p * maximum {
                retained += w;
            }
            j += lanes;
        }
    }
    retained = plane_sum(retained);
    let good = valid[row] != 0 && retained > 0.0 && retained != f32::INFINITY;
    {
        let mut j = lane;
        while j < width as usize {
            let w = weights[base + j];
            let mut value: f32 = 0.0f32;
            if good && u32::reinterpret(w) >= boundary && w >= min_p * maximum {
                value = w / retained;
            }
            weights[base + j] = value;
            j += lanes;
        }
    }
    if lane == 0 {
        valid[row] = u32::cast_from(good);
    }
}

#[ruda(launch)]
pub fn draw(
    probs: &Array<f32>,
    input_valid: &Array<u32>,
    uniforms: &Array<f32>,
    out: &mut Array<i32>,
    valid: &mut Array<u32>,
    width: u32,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let lanes = PLANE_DIM as usize;
    let u = uniforms[row];
    let mut total: f32 = 0.0f32;
    let mut last: u32 = 0u32;
    {
        let mut j = lane;
        while j < width as usize {
            let w = probs[row * width as usize + j];
            total += w;
            if w > 0.0 {
                last = j as u32;
            }
            j += lanes;
        }
    }
    total = plane_sum(total);
    last = plane_max(last);
    let good = input_valid[row] != 0 && u >= 0.0 && u < 1.0 && total > 0.0;
    let mut selected = last;
    let mut target = u * total;
    let mut found = false;
    let mut start: usize = 0usize;
    while start < width as usize {
        let j = start + lane;
        let mut w: f32 = 0.0f32;
        if j < width as usize {
            w = probs[row * width as usize + j];
        }
        let prefix = plane_inclusive_sum(w);
        let chunk = plane_sum(w);
        let mut candidate = u32::MAX;
        if w > 0.0 && target < prefix {
            candidate = j as u32;
        }
        candidate = plane_min(candidate);
        if !found && candidate != u32::MAX {
            selected = candidate;
            found = true;
        }
        target -= chunk;
        start += lanes;
    }
    if lane == 0 {
        let mut id: i32 = -1i32;
        if good {
            id = i32::cast_from(selected);
        }
        out[row] = id;
        valid[row] = u32::cast_from(good);
    }
}

#[ruda(launch)]
pub fn verify_chain(
    draft: &Array<f32>,
    ids: &Array<i32>,
    target: &Array<f32>,
    accept_u: &Array<f32>,
    sample_u: &Array<f32>,
    out: &mut Array<i32>,
    accepted: &mut Array<u32>,
    emitted_draft: &mut Array<u32>,
    emitted: &mut Array<u32>,
    valid: &mut Array<u32>,
    steps: u32,
    width: u32,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let lanes = PLANE_DIM as usize;
    let mut good = true;
    let mut done = false;
    let mut n: u32 = 0u32;
    let mut count: u32 = 0u32;
    // The alignment statistic intentionally considers *all* draft positions,
    // whereas the emitted draft prefix stops at the first rejection.
    let mut independent: u32 = 0u32;
    for step in 0..steps as usize {
        let id = ids[row * steps as usize + step];
        let u = accept_u[row * (steps as usize + 1) + step];
        if id < 0 || id >= width as i32 || !(u >= 0.0 && u < 1.0) {
            good = false;
        } else {
            let q = draft[(row * steps as usize + step) * width as usize + id as usize];
            let p = target[(row * (steps as usize + 1) + step) * width as usize + id as usize];
            if !(q > 0.0 && q < f32::INFINITY && p >= 0.0 && p < f32::INFINITY) {
                good = false;
            } else {
                if u < f32::min(1.0, p / q) {
                    independent += 1;
                }
            }
        }
    }
    if lane == 0 {
        for step in 0..steps as usize + 1 {
            out[row * (steps as usize + 1) + step] = -1;
        }
    }
    for step in 0..steps as usize + 1 {
        if !done {
            let tbase = (row * (steps as usize + 1) + step) * width as usize;
            let qbase = (row * steps as usize + step) * width as usize;
            let mut tsum: f32 = 0.0f32;
            let mut qsum: f32 = 0.0f32;
            let mut bad: u32 = 0u32;
            let mut j = lane;
            while j < width as usize {
                let p = target[tbase + j];
                tsum += p;
                if p != p || p < 0.0 || p == f32::INFINITY {
                    bad = 1u32;
                }
                if step < steps as usize {
                    let q = draft[qbase + j];
                    qsum += q;
                    if q != q || q < 0.0 || q == f32::INFINITY {
                        bad = 1u32;
                    }
                }
                j += lanes;
            }
            tsum = plane_sum(tsum);
            qsum = plane_sum(qsum);
            bad = plane_max(bad);
            if bad != 0 || (tsum - 1.0).abs() > 0.001 {
                good = false;
            }
            if step < steps as usize && (qsum - 1.0).abs() > 0.001 {
                good = false;
            }
            let mut accept = false;
            let mut candidate: i32 = -1i32;
            if step < steps as usize {
                candidate = ids[row * steps as usize + step];
                let u = accept_u[row * (steps as usize + 1) + step];
                if candidate < 0 || candidate >= width as i32 || !(u >= 0.0 && u < 1.0) {
                    good = false;
                }
                if good {
                    let q = draft[qbase + candidate as usize];
                    let p = target[tbase + candidate as usize];
                    if q <= 0.0 {
                        good = false;
                    } else {
                        accept = u < f32::min(1.0, p / q);
                    }
                }
            }
            if good && accept {
                if lane == 0 {
                    out[row * (steps as usize + 1) + step] = candidate;
                }
                n += 1;
                count += 1;
            } else {
                let u = sample_u[row * (steps as usize + 1) + step];
                if !(u >= 0.0 && u < 1.0) {
                    good = false;
                }
                let mut mass: f32 = 0.0f32;
                let mut last: u32 = 0u32;
                j = lane;
                while j < width as usize {
                    let mut w = target[tbase + j];
                    if step < steps as usize {
                        w = f32::max(0.0, w - draft[qbase + j]);
                    }
                    mass += w;
                    if w > 0.0 {
                        last = j as u32;
                    }
                    j += lanes;
                }
                mass = plane_sum(mass);
                last = plane_max(last);
                if mass <= 0.0 || mass != mass || mass == f32::INFINITY {
                    good = false;
                }
                let mut threshold = u * mass;
                let mut found = false;
                let mut selected = last;
                let mut start: usize = 0usize;
                while start < width as usize {
                    j = start + lane;
                    let mut w: f32 = 0.0f32;
                    if j < width as usize {
                        w = target[tbase + j];
                        if step < steps as usize {
                            w = f32::max(0.0, w - draft[qbase + j]);
                        }
                    }
                    let prefix = plane_inclusive_sum(w);
                    let chunk = plane_sum(w);
                    let mut id = u32::MAX;
                    if w > 0.0 && threshold < prefix {
                        id = j as u32;
                    }
                    id = plane_min(id);
                    if !found && id != u32::MAX {
                        selected = id;
                        found = true;
                    }
                    threshold -= chunk;
                    start += lanes;
                }
                if lane == 0 && good {
                    out[row * (steps as usize + 1) + step] = selected as i32;
                }
                if good {
                    count += 1;
                }
                done = true;
            }
        }
    }
    if lane == 0 {
        if !good {
            for step in 0..steps as usize + 1 {
                out[row * (steps as usize + 1) + step] = -1;
            }
            n = 0;
            count = 0;
        }
        let mut accepted_count: u32 = 0u32;
        if good {
            accepted_count = independent;
        }
        accepted[row] = accepted_count;
        emitted_draft[row] = n;
        emitted[row] = count;
        valid[row] = u32::cast_from(good);
    }
}

/// Retained native Philox draws. Output ownership/counter progression belongs
/// to the caller; this function never mutates global generator state.
pub unsafe fn uniforms_into<R: Runtime>(
    out: &RudaTensor<R>,
    seed: u64,
    offset: u64,
) -> Result<(), SamplingError> {
    let n = out.meta.num_elements();
    if out.dtype != DType::F32
        || !out.is_contiguous()
        || out.qparams.is_some()
        || n > u32::MAX as usize
        || offset.checked_add(n as u64).is_none()
    {
        return Err(SamplingError("uniform destination/counter range invalid"));
    }
    if n > 0 {
        let dim = RudaDim::new(out.client.properties(), n);
        let grid = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
        philox_uniforms::launch::<R>(
            &out.client,
            grid,
            dim,
            out.clone().into_array_arg(),
            n as u32,
            seed,
            offset,
        );
    }
    Ok(())
}
