//! Four-stream mHC mapping/collapse, residual mixing and deterministic training.
//! Parameters and saved-record layout follow the public prepare/finish split;
//! no architecture-specific generated CUDA program is embedded or invoked.
use super::{InferenceError, alloc, check, float, plane};
use ruda_core::tensor::DType;
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, contiguous::into_contiguous},
};
#[derive(Clone, Copy, Debug)]
pub struct MhcOptions {
    pub rms_epsilon: f32,
    pub coefficient_epsilon: f32,
    pub sinkhorn_iterations: usize,
}
impl Default for MhcOptions {
    fn default() -> Self {
        Self {
            rms_epsilon: 1e-5,
            coefficient_epsilon: 1e-6,
            sinkhorn_iterations: 20,
        }
    }
}
pub struct PreparedMhc<R: Runtime> {
    pub collapsed: RudaTensor<R>,
    pub statistics: RudaTensor<R>,
    pub options: MhcOptions,
}
pub struct MhcBackwardPrepare<R: Runtime> {
    pub branch_gradient: RudaTensor<R>,
    pub statistics_gradient: RudaTensor<R>,
}
pub struct MhcParameterGradients<R: Runtime> {
    pub mapping: RudaTensor<R>,
    pub scale: RudaTensor<R>,
    pub base: RudaTensor<R>,
}
pub struct MhcGradients<R: Runtime> {
    pub input: RudaTensor<R>,
    pub parameters: MhcParameterGradients<R>,
}
/// Persistent prepare/finish and backward workspaces. All numerical stages are
/// allocation-free after this construction; outputs are reused on every call.
pub struct MhcPlan<R: Runtime> {
    pack: super::PreparedPack<R>,
    mapping: RudaTensor<R>,
    scale: RudaTensor<R>,
    base: RudaTensor<R>,
    pub prepared: PreparedMhc<R>,
    pub output: RudaTensor<R>,
    pub backward_prepared: MhcBackwardPrepare<R>,
    pub gradients: MhcGradients<R>,
    dz: RudaTensor<R>,
    da: RudaTensor<R>,
    tokens: usize,
    hidden: usize,
    lanes: u32,
}
impl<R: Runtime> MhcPlan<R> {
    pub fn new(
        input: &RudaTensor<R>,
        mapping: &RudaTensor<R>,
        scale: &RudaTensor<R>,
        base: &RudaTensor<R>,
        options: MhcOptions,
    ) -> Result<Self, InferenceError> {
        let pack = super::PreparedPack::prepare(input)?;
        let x = &pack.output;
        let (tokens, hidden) = input_shape(x)?;
        parameters(x, mapping, scale, base, hidden)?;
        if !options.rms_epsilon.is_finite()
            || options.rms_epsilon <= 0.0
            || !options.coefficient_epsilon.is_finite()
            || options.coefficient_epsilon <= 0.0
            || !(1..=64).contains(&options.sinkhorn_iterations)
        {
            return Err(InferenceError("invalid prepared mHC recipe"));
        }
        let lanes = plane(x, tokens, 1)?;
        let prepared = PreparedMhc {
            collapsed: alloc(x, [tokens, hidden], x.dtype),
            statistics: alloc(x, [tokens, 64], DType::F32),
            options,
        };
        let output = alloc(x, x.meta.shape().clone(), x.dtype);
        let backward_prepared = MhcBackwardPrepare {
            branch_gradient: alloc(x, [tokens, hidden], x.dtype),
            statistics_gradient: alloc(x, [tokens, 32], DType::F32),
        };
        let gradients = MhcGradients {
            input: alloc(x, x.meta.shape().clone(), x.dtype),
            parameters: MhcParameterGradients {
                mapping: alloc(x, [24, 4 * hidden], DType::F32),
                scale: alloc(x, [3], DType::F32),
                base: alloc(x, [24], DType::F32),
            },
        };
        let dz = alloc(x, [tokens, 24], DType::F32);
        let da = alloc(x, [tokens, 24], DType::F32);
        Ok(Self {
            pack,
            mapping: mapping.clone(),
            scale: scale.clone(),
            base: base.clone(),
            prepared,
            output,
            backward_prepared,
            gradients,
            dz,
            da,
            tokens,
            hidden,
            lanes,
        })
    }
    pub unsafe fn forward_prepare(&self) -> &PreparedMhc<R> {
        unsafe {
            self.pack.run();
        }
        let x = &self.pack.output;
        let o = self.prepared.options;
        if self.tokens > 0 {
            prepare::launch::<R>(
                &x.client,
                RudaCount::Static(self.tokens as u32, 1, 1),
                RudaDim::new_1d(self.lanes),
                x.clone().into_array_arg(),
                self.mapping.clone().into_array_arg(),
                self.scale.clone().into_array_arg(),
                self.base.clone().into_array_arg(),
                self.prepared.collapsed.clone().into_array_arg(),
                self.prepared.statistics.clone().into_array_arg(),
                self.hidden as u32,
                o.rms_epsilon,
                o.coefficient_epsilon,
                o.sinkhorn_iterations as u32,
                x.dtype.into(),
            );
        }
        &self.prepared
    }
    pub unsafe fn forward_finish(
        &self,
        branch: &RudaTensor<R>,
    ) -> Result<&RudaTensor<R>, InferenceError> {
        let x = &self.pack.output;
        check(branch, x)?;
        if branch.dtype != x.dtype || branch.meta.shape()[..] != [self.tokens, self.hidden] {
            return Err(InferenceError("prepared mHC branch mismatch"));
        }
        let n = x.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(x.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
            finish::launch::<R>(
                &x.client,
                count,
                dim,
                x.clone().into_array_arg(),
                branch.clone().into_array_arg(),
                self.prepared.statistics.clone().into_array_arg(),
                self.output.clone().into_array_arg(),
                self.tokens as u32,
                self.hidden as u32,
                x.dtype.into(),
            );
        }
        Ok(&self.output)
    }
    pub unsafe fn backward_prepare(
        &self,
        branch: &RudaTensor<R>,
        dy: &RudaTensor<R>,
    ) -> Result<&MhcBackwardPrepare<R>, InferenceError> {
        let x = &self.pack.output;
        for t in [branch, dy] {
            check(t, x)?;
        }
        if branch.dtype != x.dtype
            || branch.meta.shape()[..] != [self.tokens, self.hidden]
            || dy.dtype != x.dtype
            || dy.meta.shape() != x.meta.shape()
        {
            return Err(InferenceError(
                "prepared mHC branch/output gradient mismatch",
            ));
        }
        if self.tokens > 0 {
            prepare_gradient::launch::<R>(
                &x.client,
                RudaCount::Static(self.tokens as u32, 1, 1),
                RudaDim::new_1d(self.lanes),
                x.clone().into_array_arg(),
                branch.clone().into_array_arg(),
                dy.clone().into_array_arg(),
                self.prepared.statistics.clone().into_array_arg(),
                self.backward_prepared
                    .branch_gradient
                    .clone()
                    .into_array_arg(),
                self.backward_prepared
                    .statistics_gradient
                    .clone()
                    .into_array_arg(),
                self.hidden as u32,
                x.dtype.into(),
            );
        }
        Ok(&self.backward_prepared)
    }
    pub unsafe fn backward_finish(
        &self,
        dy: &RudaTensor<R>,
        du: &RudaTensor<R>,
        accumulate: Option<&MhcParameterGradients<R>>,
    ) -> Result<&MhcGradients<R>, InferenceError> {
        let x = &self.pack.output;
        for t in [dy, du] {
            check(t, x)?;
        }
        if dy.dtype != x.dtype
            || dy.meta.shape() != x.meta.shape()
            || du.dtype != x.dtype
            || du.meta.shape()[..] != [self.tokens, self.hidden]
        {
            return Err(InferenceError(
                "prepared mHC input/output gradient mismatch",
            ));
        }
        if let Some(old) = accumulate {
            parameters(x, &old.mapping, &old.scale, &old.base, self.hidden)?;
        }
        let o = self.prepared.options;
        if self.tokens > 0 {
            finish_gradient::launch::<R>(
                &x.client,
                RudaCount::Static(self.tokens as u32, 1, 1),
                RudaDim::new_1d(self.lanes),
                x.clone().into_array_arg(),
                dy.clone().into_array_arg(),
                du.clone().into_array_arg(),
                self.mapping.clone().into_array_arg(),
                self.scale.clone().into_array_arg(),
                self.base.clone().into_array_arg(),
                self.prepared.statistics.clone().into_array_arg(),
                self.backward_prepared
                    .statistics_gradient
                    .clone()
                    .into_array_arg(),
                self.gradients.input.clone().into_array_arg(),
                self.dz.clone().into_array_arg(),
                self.da.clone().into_array_arg(),
                self.hidden as u32,
                o.coefficient_epsilon,
                o.sinkhorn_iterations as u32,
                x.dtype.into(),
            );
        }
        let n = 24 * 4 * self.hidden + 27;
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        parameter_gradient::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            self.prepared.statistics.clone().into_array_arg(),
            self.dz.clone().into_array_arg(),
            self.da.clone().into_array_arg(),
            accumulate
                .map(|v| &v.mapping)
                .unwrap_or(&self.mapping)
                .clone()
                .into_array_arg(),
            accumulate
                .map(|v| &v.scale)
                .unwrap_or(&self.scale)
                .clone()
                .into_array_arg(),
            accumulate
                .map(|v| &v.base)
                .unwrap_or(&self.base)
                .clone()
                .into_array_arg(),
            self.gradients.parameters.mapping.clone().into_array_arg(),
            self.gradients.parameters.scale.clone().into_array_arg(),
            self.gradients.parameters.base.clone().into_array_arg(),
            self.tokens as u32,
            self.hidden as u32,
            accumulate.is_some(),
            x.dtype.into(),
        );
        Ok(&self.gradients)
    }
}
fn input_shape<R: Runtime>(x: &RudaTensor<R>) -> Result<(usize, usize), InferenceError> {
    check(x, x)?;
    float(x)?;
    if x.meta.num_dims() != 3 || x.meta.shape()[1] != 4 || x.meta.shape()[2] == 0 {
        return Err(InferenceError("mHC input must be [tokens,4,hidden]"));
    }
    Ok((x.meta.shape()[0], x.meta.shape()[2]))
}
fn parameters<R: Runtime>(
    x: &RudaTensor<R>,
    mapping: &RudaTensor<R>,
    scale: &RudaTensor<R>,
    base: &RudaTensor<R>,
    hidden: usize,
) -> Result<(), InferenceError> {
    for t in [mapping, scale, base] {
        check(t, x)?;
        if t.dtype != DType::F32 {
            return Err(InferenceError("mHC parameters must be F32"));
        }
    }
    if mapping.meta.shape()[..] != [24, 4 * hidden]
        || scale.meta.shape()[..] != [3]
        || base.meta.shape()[..] != [24]
    {
        return Err(InferenceError("mHC parameter dimensions mismatch"));
    }
    Ok(())
}
pub fn forward_prepare<R: Runtime>(
    input: &RudaTensor<R>,
    mapping: &RudaTensor<R>,
    scale: &RudaTensor<R>,
    base: &RudaTensor<R>,
    options: MhcOptions,
) -> Result<PreparedMhc<R>, InferenceError> {
    let x = into_contiguous(input.clone());
    let (tokens, hidden) = input_shape(&x)?;
    parameters(&x, mapping, scale, base, hidden)?;
    if !options.rms_epsilon.is_finite()
        || options.rms_epsilon <= 0.0
        || !options.coefficient_epsilon.is_finite()
        || options.coefficient_epsilon <= 0.0
        || !(1..=64).contains(&options.sinkhorn_iterations)
    {
        return Err(InferenceError("invalid mHC normalization/iteration recipe"));
    }
    let collapsed = alloc(&x, [tokens, hidden], x.dtype);
    let statistics = alloc(&x, [tokens, 64], DType::F32);
    let lanes = plane(&x, tokens, 1)?;
    if tokens != 0 {
        prepare::launch::<R>(
            &x.client,
            RudaCount::Static(tokens as u32, 1, 1),
            RudaDim::new_1d(lanes),
            x.clone().into_array_arg(),
            mapping.clone().into_array_arg(),
            scale.clone().into_array_arg(),
            base.clone().into_array_arg(),
            collapsed.clone().into_array_arg(),
            statistics.clone().into_array_arg(),
            hidden as u32,
            options.rms_epsilon,
            options.coefficient_epsilon,
            options.sinkhorn_iterations as u32,
            x.dtype.into(),
        );
    }
    Ok(PreparedMhc {
        collapsed,
        statistics,
        options,
    })
}
pub fn forward_stats<R: Runtime>(
    input: &RudaTensor<R>,
    mapping: &RudaTensor<R>,
    scale: &RudaTensor<R>,
    base: &RudaTensor<R>,
    options: MhcOptions,
) -> Result<RudaTensor<R>, InferenceError> {
    Ok(forward_prepare(input, mapping, scale, base, options)?.statistics)
}
pub fn forward_finish<R: Runtime>(
    x: &RudaTensor<R>,
    branch: &RudaTensor<R>,
    statistics: &RudaTensor<R>,
) -> Result<RudaTensor<R>, InferenceError> {
    let (tokens, hidden) = input_shape(x)?;
    check(branch, x)?;
    check(statistics, x)?;
    if branch.dtype != x.dtype
        || branch.meta.shape()[..] != [tokens, hidden]
        || statistics.dtype != DType::F32
        || statistics.meta.shape()[..] != [tokens, 64]
    {
        return Err(InferenceError("mHC finish operands mismatch"));
    }
    let out = alloc(x, x.meta.shape().clone(), x.dtype);
    let n = x.meta.num_elements();
    if n != 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        finish::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            branch.clone().into_array_arg(),
            statistics.clone().into_array_arg(),
            out.clone().into_array_arg(),
            tokens as u32,
            hidden as u32,
            x.dtype.into(),
        );
    }
    Ok(out)
}
pub fn backward_prepare<R: Runtime>(
    x: &RudaTensor<R>,
    branch: &RudaTensor<R>,
    grad: &RudaTensor<R>,
    statistics: &RudaTensor<R>,
) -> Result<MhcBackwardPrepare<R>, InferenceError> {
    let (tokens, hidden) = input_shape(x)?;
    for t in [branch, grad, statistics] {
        check(t, x)?;
    }
    if grad.dtype != x.dtype
        || grad.meta.shape() != x.meta.shape()
        || branch.dtype != x.dtype
        || branch.meta.shape()[..] != [tokens, hidden]
        || statistics.dtype != DType::F32
        || statistics.meta.shape()[..] != [tokens, 64]
    {
        return Err(InferenceError("mHC backward-prepare operands mismatch"));
    }
    let db = alloc(x, [tokens, hidden], x.dtype);
    let gs = alloc(x, [tokens, 32], DType::F32);
    let lanes = plane(x, tokens, 1)?;
    if tokens != 0 {
        prepare_gradient::launch::<R>(
            &x.client,
            RudaCount::Static(tokens as u32, 1, 1),
            RudaDim::new_1d(lanes),
            x.clone().into_array_arg(),
            branch.clone().into_array_arg(),
            grad.clone().into_array_arg(),
            statistics.clone().into_array_arg(),
            db.clone().into_array_arg(),
            gs.clone().into_array_arg(),
            hidden as u32,
            x.dtype.into(),
        );
    }
    Ok(MhcBackwardPrepare {
        branch_gradient: db,
        statistics_gradient: gs,
    })
}
pub fn backward_finish<R: Runtime>(
    x: &RudaTensor<R>,
    grad_y: &RudaTensor<R>,
    grad_u: &RudaTensor<R>,
    mapping: &RudaTensor<R>,
    scale: &RudaTensor<R>,
    base: &RudaTensor<R>,
    saved: &PreparedMhc<R>,
    prepared: &MhcBackwardPrepare<R>,
    accumulate: Option<&MhcParameterGradients<R>>,
) -> Result<MhcGradients<R>, InferenceError> {
    let (tokens, hidden) = input_shape(x)?;
    parameters(x, mapping, scale, base, hidden)?;
    for t in [
        grad_y,
        grad_u,
        &saved.statistics,
        &prepared.statistics_gradient,
    ] {
        check(t, x)?;
    }
    if grad_y.dtype != x.dtype
        || grad_y.meta.shape() != x.meta.shape()
        || grad_u.dtype != x.dtype
        || grad_u.meta.shape()[..] != [tokens, hidden]
        || saved.statistics.dtype != DType::F32
        || saved.statistics.meta.shape()[..] != [tokens, 64]
        || prepared.statistics_gradient.dtype != DType::F32
        || prepared.statistics_gradient.meta.shape()[..] != [tokens, 32]
    {
        return Err(InferenceError("mHC backward-finish operands mismatch"));
    }
    if let Some(old) = accumulate {
        parameters(x, &old.mapping, &old.scale, &old.base, hidden)?;
    }
    let dx = alloc(x, x.meta.shape().clone(), x.dtype);
    let dz = alloc(x, [tokens, 24], DType::F32);
    let da = alloc(x, [tokens, 24], DType::F32);
    let lanes = plane(x, tokens, 1)?;
    if tokens != 0 {
        finish_gradient::launch::<R>(
            &x.client,
            RudaCount::Static(tokens as u32, 1, 1),
            RudaDim::new_1d(lanes),
            x.clone().into_array_arg(),
            grad_y.clone().into_array_arg(),
            grad_u.clone().into_array_arg(),
            mapping.clone().into_array_arg(),
            scale.clone().into_array_arg(),
            base.clone().into_array_arg(),
            saved.statistics.clone().into_array_arg(),
            prepared.statistics_gradient.clone().into_array_arg(),
            dx.clone().into_array_arg(),
            dz.clone().into_array_arg(),
            da.clone().into_array_arg(),
            hidden as u32,
            saved.options.coefficient_epsilon,
            saved.options.sinkhorn_iterations as u32,
            x.dtype.into(),
        );
    }
    let df = alloc(x, [24, 4 * hidden], DType::F32);
    let ds = alloc(x, [3], DType::F32);
    let db = alloc(x, [24], DType::F32);
    let n = 24 * 4 * hidden + 27;
    let dim = RudaDim::new(x.client.properties(), n);
    let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
    parameter_gradient::launch::<R>(
        &x.client,
        count,
        dim,
        x.clone().into_array_arg(),
        saved.statistics.clone().into_array_arg(),
        dz.into_array_arg(),
        da.into_array_arg(),
        accumulate
            .map(|v| &v.mapping)
            .unwrap_or(mapping)
            .clone()
            .into_array_arg(),
        accumulate
            .map(|v| &v.scale)
            .unwrap_or(scale)
            .clone()
            .into_array_arg(),
        accumulate
            .map(|v| &v.base)
            .unwrap_or(base)
            .clone()
            .into_array_arg(),
        df.clone().into_array_arg(),
        ds.clone().into_array_arg(),
        db.clone().into_array_arg(),
        tokens as u32,
        hidden as u32,
        accumulate.is_some(),
        x.dtype.into(),
    );
    Ok(MhcGradients {
        input: dx,
        parameters: MhcParameterGradients {
            mapping: df,
            scale: ds,
            base: db,
        },
    })
}
pub fn head<R: Runtime>(x: &RudaTensor<R>) -> Result<RudaTensor<R>, InferenceError> {
    let (tokens, hidden) = input_shape(x)?;
    let out = alloc(x, [tokens, hidden], x.dtype);
    let n = tokens * hidden;
    if n != 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        head_kernel::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            hidden as u32,
            x.dtype.into(),
        );
    }
    Ok(out)
}
pub fn head_backward<R: Runtime>(
    grad: &RudaTensor<R>,
    accumulate: Option<&RudaTensor<R>>,
) -> Result<RudaTensor<R>, InferenceError> {
    check(grad, grad)?;
    float(grad)?;
    if grad.meta.num_dims() != 2 {
        return Err(InferenceError("mHC head gradient must be [T,H]"));
    }
    let tokens = grad.meta.shape()[0];
    let hidden = grad.meta.shape()[1];
    if let Some(old) = accumulate {
        check(old, grad)?;
        if old.dtype != grad.dtype || old.meta.shape()[..] != [tokens, 4, hidden] {
            return Err(InferenceError("mHC head accumulation shape mismatch"));
        }
    }
    let out = alloc(grad, [tokens, 4, hidden], grad.dtype);
    let n = out.meta.num_elements();
    if n > u32::MAX as usize {
        return Err(InferenceError("mHC head gradient exceeds U32"));
    }
    if n > 0 {
        let dim = RudaDim::new(grad.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&grad.client, n, dim);
        head_gradient::launch::<R>(
            &grad.client,
            count,
            dim,
            grad.clone().into_array_arg(),
            accumulate.unwrap_or(grad).clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            hidden as u32,
            accumulate.is_some(),
            grad.dtype.into(),
        );
    }
    Ok(out)
}

pub struct PreMap<R: Runtime> {
    pub post: RudaTensor<R>,
    pub combination: RudaTensor<R>,
    pub layer_input: RudaTensor<R>,
}
#[derive(Clone, Copy, Debug)]
pub struct PreMapOptions {
    pub rms_epsilon: f32,
    pub pre_epsilon: f32,
    pub sinkhorn_epsilon: f32,
    pub post_multiplier: f32,
    pub iterations: usize,
}
impl Default for PreMapOptions {
    fn default() -> Self {
        Self {
            rms_epsilon: 1e-6,
            pre_epsilon: 1e-6,
            sinkhorn_epsilon: 1e-6,
            post_multiplier: 1.0,
            iterations: 20,
        }
    }
}
/// Split projection partials [splits,T,24], optional split square sums
/// [splits,T]. Omitting square sums computes them from the residual. `prenorm`
/// means the projection already includes inverse RMS scaling.
pub fn pre_map<R: Runtime>(
    residual: &RudaTensor<R>,
    dot_mix: &RudaTensor<R>,
    square_sums: Option<&RudaTensor<R>>,
    scale: &RudaTensor<R>,
    base: &RudaTensor<R>,
    prenorm: bool,
    norm_weight: Option<&RudaTensor<R>>,
    norm_epsilon: f32,
    options: PreMapOptions,
) -> Result<PreMap<R>, InferenceError> {
    let prepared = PreparedMhcPreMap::prepare(
        residual,
        dot_mix,
        square_sums,
        scale,
        base,
        prenorm,
        norm_weight,
        norm_epsilon,
        options,
    )?;
    unsafe {
        prepared.run()?;
    }
    Ok(PreMap {
        post: prepared.output.post.clone(),
        combination: prepared.output.combination.clone(),
        layer_input: prepared.output.layer_input.clone(),
    })
}
/// Prepared external/split-K projection finalizer, including the distinct
/// pre-rounding RMS statistic used by the output-normalization variant.
pub struct PreparedMhcPreMap<R: Runtime> {
    residual: super::PreparedPack<R>,
    parameters: Vec<(super::PreparedPack<R>, RudaTensor<R>)>,
    dot: RudaTensor<R>,
    squares: Option<RudaTensor<R>>,
    scale: RudaTensor<R>,
    base: RudaTensor<R>,
    norm_weight: Option<RudaTensor<R>>,
    pub output: PreMap<R>,
    prenorm: bool,
    norm_epsilon: f32,
    options: PreMapOptions,
    tokens: usize,
    hidden: usize,
    splits: usize,
    lanes: u32,
}
impl<R: Runtime> PreparedMhcPreMap<R> {
    pub fn prepare(
        residual: &RudaTensor<R>,
        dot: &RudaTensor<R>,
        squares: Option<&RudaTensor<R>>,
        scale: &RudaTensor<R>,
        base: &RudaTensor<R>,
        prenorm: bool,
        norm_weight: Option<&RudaTensor<R>>,
        norm_epsilon: f32,
        o: PreMapOptions,
    ) -> Result<Self, InferenceError> {
        let residual = super::PreparedPack::prepare(residual)?;
        let x = &residual.output;
        let (tokens, hidden) = input_shape(x)?;
        let mut parameters = Vec::new();
        let mut parameter = |source: &RudaTensor<R>| -> Result<RudaTensor<R>, InferenceError> {
            super::check_view(source, x)?;
            float(source)?;
            let pack = super::PreparedPack::prepare(source)?;
            let out = alloc(x, source.meta.shape().clone(), DType::F32);
            parameters.push((pack, out.clone()));
            Ok(out)
        };
        let dot = parameter(dot)?;
        let squares = squares.map(&mut parameter).transpose()?;
        let scale = parameter(scale)?;
        let base = parameter(base)?;
        let norm_weight = norm_weight.map(&mut parameter).transpose()?;
        let splits = if dot.meta.num_dims() == 3 {
            dot.meta.shape()[0]
        } else {
            1
        };
        if splits == 0
            || dot.meta.num_elements() != splits * tokens * 24
            || scale.meta.shape()[..] != [3]
            || base.meta.shape()[..] != [24]
            || o.iterations == 0
            || o.iterations > 64
            || !o.rms_epsilon.is_finite()
            || o.rms_epsilon <= 0.0
            || !o.pre_epsilon.is_finite()
            || o.pre_epsilon < 0.0
            || !o.sinkhorn_epsilon.is_finite()
            || o.sinkhorn_epsilon <= 0.0
            || !o.post_multiplier.is_finite()
            || squares
                .as_ref()
                .is_some_and(|s| s.meta.num_elements() != splits * tokens)
            || norm_weight.as_ref().is_some_and(|w| {
                w.meta.shape()[..] != [hidden] || !norm_epsilon.is_finite() || norm_epsilon <= 0.0
            })
        {
            return Err(InferenceError(
                "mHC split projection/options/output norm mismatch",
            ));
        }
        let output = PreMap {
            post: alloc(x, [tokens, 4], DType::F32),
            combination: alloc(x, [tokens, 4, 4], DType::F32),
            layer_input: alloc(x, [tokens, hidden], x.dtype),
        };
        let lanes = plane(x, tokens, 1)?;
        Ok(Self {
            residual,
            parameters,
            dot,
            squares,
            scale,
            base,
            norm_weight,
            output,
            prenorm,
            norm_epsilon,
            options: o,
            tokens,
            hidden,
            splits,
            lanes,
        })
    }
    pub unsafe fn run(&self) -> Result<&PreMap<R>, InferenceError> {
        unsafe {
            self.residual.run();
            for (p, out) in &self.parameters {
                p.run();
                super::norm::cast_into(&p.output, out)?;
            }
        }
        let x = &self.residual.output;
        let o = self.options;
        if self.tokens > 0 {
            pre_external::launch::<R>(
                &x.client,
                RudaCount::Static(self.tokens as u32, 1, 1),
                RudaDim::new_1d(self.lanes),
                x.clone().into_array_arg(),
                self.dot.clone().into_array_arg(),
                self.squares
                    .as_ref()
                    .unwrap_or(&self.dot)
                    .clone()
                    .into_array_arg(),
                self.scale.clone().into_array_arg(),
                self.base.clone().into_array_arg(),
                self.norm_weight
                    .as_ref()
                    .unwrap_or(&self.scale)
                    .clone()
                    .into_array_arg(),
                self.output.post.clone().into_array_arg(),
                self.output.combination.clone().into_array_arg(),
                self.output.layer_input.clone().into_array_arg(),
                self.tokens as u32,
                self.hidden as u32,
                self.splits as u32,
                o.rms_epsilon,
                o.pre_epsilon,
                o.sinkhorn_epsilon,
                o.post_multiplier,
                self.norm_epsilon,
                o.iterations as u32,
                self.squares.is_some(),
                self.prenorm,
                self.norm_weight.is_some(),
                x.dtype.into(),
            );
        }
        Ok(&self.output)
    }
}

pub fn post_map<R: Runtime>(
    residual: &RudaTensor<R>,
    branch: &RudaTensor<R>,
    post: &RudaTensor<R>,
    combination: &RudaTensor<R>,
) -> Result<RudaTensor<R>, InferenceError> {
    let (tokens, hidden) = input_shape(residual)?;
    for t in [branch, post, combination] {
        check(t, residual)?;
    }
    if branch.dtype != residual.dtype
        || branch.meta.shape()[..] != [tokens, hidden]
        || post.dtype != DType::F32
        || post.meta.num_elements() != tokens * 4
        || combination.dtype != DType::F32
        || combination.meta.shape()[..] != [tokens, 4, 4]
    {
        return Err(InferenceError("mHC post-map operands mismatch"));
    }
    let out = alloc(residual, residual.meta.shape().clone(), residual.dtype);
    let n = out.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(out.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
        post_external::launch::<R>(
            &out.client,
            count,
            dim,
            residual.clone().into_array_arg(),
            branch.clone().into_array_arg(),
            post.clone().into_array_arg(),
            combination.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            hidden as u32,
            residual.dtype.into(),
        );
    }
    Ok(out)
}

#[ruda]
fn sigmoid(x: f32) -> f32 {
    let e = (-x.abs()).exp();
    let mut result = e / (1.0 + e);
    if x >= 0.0 {
        result = 1.0 / (1.0 + e);
    }
    result
}
#[ruda]
fn normalize_matrix(matrix: &mut Array<f32>, eps: f32, #[comptime] columns: bool) {
    for axis in 0usize..4 {
        let mut sum = eps;
        for j in 0usize..4 {
            let mut index = axis * 4 + j;
            if comptime!(columns) {
                index = j * 4 + axis;
            }
            sum += matrix[index];
        }
        for j in 0usize..4 {
            let mut index = axis * 4 + j;
            if comptime!(columns) {
                index = j * 4 + axis;
            }
            matrix[index] /= sum;
        }
    }
}
#[ruda]
fn initial_matrix(
    z: &Array<f32>,
    scale: &Array<f32>,
    base: &Array<f32>,
    matrix: &mut Array<f32>,
    eps: f32,
) {
    for i in 0usize..4 {
        let mut maximum = f32::NEG_INFINITY;
        for j in 0usize..4 {
            maximum = f32::max(
                maximum,
                fma(z[8 + i * 4 + j], scale[2], base[8 + i * 4 + j]),
            );
        }
        let mut total: f32 = 0.0f32;
        for j in 0usize..4 {
            let value = (fma(z[8 + i * 4 + j], scale[2], base[8 + i * 4 + j]) - maximum).exp();
            matrix[i * 4 + j] = value;
            total += value;
        }
        for j in 0usize..4 {
            matrix[i * 4 + j] = matrix[i * 4 + j] / total + eps;
        }
    }
}
#[ruda(launch)]
pub fn prepare<F: Float>(
    x: &Array<F>,
    mapping: &Array<f32>,
    scale: &Array<f32>,
    base: &Array<f32>,
    collapsed: &mut Array<F>,
    statistics: &mut Array<f32>,
    hidden: u32,
    rms_eps: f32,
    hc_eps: f32,
    iterations: u32,
    #[define(F)] _dtype: StorageType,
) {
    let token = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let width = hidden as usize * 4;
    let mut square: f32 = 0.0f32;
    let mut c = lane;
    while c < width {
        let v = f32::cast_from(x[token * width + c]);
        square += v * v;
        c += PLANE_DIM as usize;
    }
    let inv = 1.0 / (plane_sum(square) / width as f32 + rms_eps).sqrt();
    let mut z = Array::<f32>::new(24usize);
    for p in 0usize..24 {
        let mut projection: f32 = 0.0f32;
        c = lane;
        while c < width {
            projection = fma(
                f32::cast_from(x[token * width + c]),
                mapping[p * width + c],
                projection,
            );
            c += PLANE_DIM as usize;
        }
        z[p] = plane_sum(projection) * inv;
    }
    let mut matrix = Array::<f32>::new(16usize);
    initial_matrix(&z, scale, base, &mut matrix, hc_eps);
    normalize_matrix(&mut matrix, hc_eps, true);
    for step in 1..iterations {
        normalize_matrix(&mut matrix, hc_eps, false);
        normalize_matrix(&mut matrix, hc_eps, true);
    }
    let mut pre = Array::<f32>::new(4usize);
    for i in 0usize..4 {
        pre[i] = sigmoid(fma(z[i], scale[0], base[i])) + hc_eps;
    }
    c = lane;
    while c < hidden as usize {
        let mut value: f32 = 0.0f32;
        for i in 0usize..4 {
            value = fma(
                pre[i],
                f32::cast_from(x[token * width + i * hidden as usize + c]),
                value,
            );
        }
        collapsed[token * hidden as usize + c] = F::cast_from(value);
        c += PLANE_DIM as usize;
    }
    if lane == 0 {
        let s = token * 64;
        statistics[s] = inv;
        for i in 0usize..24 {
            statistics[s + 1 + i] = z[i];
        }
        for i in 0usize..4 {
            statistics[s + 25 + i] = pre[i];
            statistics[s + 29 + i] = 2.0 * sigmoid(fma(z[i + 4], scale[1], base[i + 4]));
        }
        for i in 0usize..16 {
            statistics[s + 33 + i] = matrix[i];
        }
        for i in 49usize..64 {
            statistics[s + i] = 0.0;
        }
    }
}
#[ruda(launch)]
pub fn finish<F: Float>(
    x: &Array<F>,
    branch: &Array<F>,
    statistics: &Array<f32>,
    out: &mut Array<F>,
    tokens: u32,
    hidden: u32,
    #[define(F)] _dtype: StorageType,
) {
    let id = ABSOLUTE_POS as usize;
    if id < tokens as usize * 4 * hidden as usize {
        let d = id % hidden as usize;
        let j = id / hidden as usize % 4;
        let t = id / (hidden as usize * 4);
        let s = t * 64;
        let mut value = statistics[s + 29 + j] * f32::cast_from(branch[t * hidden as usize + d]);
        for i in 0usize..4 {
            value = fma(
                statistics[s + 33 + i * 4 + j],
                f32::cast_from(x[(t * 4 + i) * hidden as usize + d]),
                value,
            );
        }
        out[id] = F::cast_from(value);
    }
}
#[ruda(launch)]
pub fn prepare_gradient<F: Float>(
    x: &Array<F>,
    branch: &Array<F>,
    dy: &Array<F>,
    statistics: &Array<f32>,
    db: &mut Array<F>,
    gs: &mut Array<f32>,
    hidden: u32,
    #[define(F)] _dtype: StorageType,
) {
    let t = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let mut d = lane;
    while d < hidden as usize {
        let mut value: f32 = 0.0f32;
        for j in 0usize..4 {
            value = fma(
                statistics[t * 64 + 29 + j],
                f32::cast_from(dy[(t * 4 + j) * hidden as usize + d]),
                value,
            );
        }
        db[t * hidden as usize + d] = F::cast_from(value);
        d += PLANE_DIM as usize;
    }
    for j in 0usize..4 {
        let mut sum: f32 = 0.0f32;
        d = lane;
        while d < hidden as usize {
            sum += f32::cast_from(dy[(t * 4 + j) * hidden as usize + d])
                * f32::cast_from(branch[t * hidden as usize + d]);
            d += PLANE_DIM as usize;
        }
        sum = plane_sum(sum);
        if lane == 0 {
            gs[t * 32 + j] = sum;
        }
    }
    for i in 0usize..4 {
        for j in 0usize..4 {
            let mut sum: f32 = 0.0f32;
            d = lane;
            while d < hidden as usize {
                sum += f32::cast_from(x[(t * 4 + i) * hidden as usize + d])
                    * f32::cast_from(dy[(t * 4 + j) * hidden as usize + d]);
                d += PLANE_DIM as usize;
            }
            sum = plane_sum(sum);
            if lane == 0 {
                gs[t * 32 + 4 + i * 4 + j] = sum;
            }
        }
    }
    if lane == 0 {
        for i in 20usize..32 {
            gs[t * 32 + i] = 0.0;
        }
    }
}
#[ruda(launch)]
pub fn finish_gradient<F: Float>(
    x: &Array<F>,
    dy: &Array<F>,
    du: &Array<F>,
    mapping: &Array<f32>,
    scale: &Array<f32>,
    base: &Array<f32>,
    statistics: &Array<f32>,
    gs: &Array<f32>,
    dx: &mut Array<F>,
    dz_out: &mut Array<f32>,
    da_out: &mut Array<f32>,
    hidden: u32,
    hc_eps: f32,
    iterations: u32,
    #[define(F)] _dtype: StorageType,
) {
    let t = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let width = hidden as usize * 4;
    let mut z = Array::<f32>::new(24usize);
    for i in 0usize..24 {
        z[i] = statistics[t * 64 + 1 + i];
    }
    let mut da = Array::<f32>::new(24usize);
    for i in 0usize..4 {
        let mut dot: f32 = 0.0f32;
        let mut d = lane;
        while d < hidden as usize {
            dot += f32::cast_from(du[t * hidden as usize + d])
                * f32::cast_from(x[(t * 4 + i) * hidden as usize + d]);
            d += PLANE_DIM as usize;
        }
        dot = plane_sum(dot);
        let pre = statistics[t * 64 + 25 + i] - hc_eps;
        let post = statistics[t * 64 + 29 + i] * 0.5;
        da[i] = dot * pre * (1.0 - pre);
        da[4 + i] = gs[t * 32 + i] * 2.0 * post * (1.0 - post);
    }
    let mut matrix = Array::<f32>::new(16usize);
    let mut history = Array::<f32>::new(2048usize);
    initial_matrix(&z, scale, base, &mut matrix, hc_eps);
    for j in 0usize..16 {
        history[j] = matrix[j];
    }
    normalize_matrix(&mut matrix, hc_eps, true);
    for j in 0usize..16 {
        history[16 + j] = matrix[j];
    }
    for step in 1..iterations as usize {
        normalize_matrix(&mut matrix, hc_eps, false);
        for j in 0usize..16 {
            history[(step * 2) * 16 + j] = matrix[j];
        }
        normalize_matrix(&mut matrix, hc_eps, true);
        for j in 0usize..16 {
            history[(step * 2 + 1) * 16 + j] = matrix[j];
        }
    }
    let mut grad = Array::<f32>::new(16usize);
    for j in 0usize..16 {
        grad[j] = gs[t * 32 + 4 + j];
    }
    let mut stage = iterations as usize * 2 - 1;
    while stage > 0 {
        for axis in 0usize..4 {
            let mut denom = hc_eps;
            let mut dot: f32 = 0.0f32;
            for j in 0usize..4 {
                let mut index = axis * 4 + j;
                if stage % 2 != 0 {
                    index = j * 4 + axis;
                }
                denom += history[(stage - 1) * 16 + index];
                dot += grad[index] * history[stage * 16 + index];
            }
            for j in 0usize..4 {
                let mut index = axis * 4 + j;
                if stage % 2 != 0 {
                    index = j * 4 + axis;
                }
                grad[index] = (grad[index] - dot) / denom;
            }
        }
        stage -= 1;
    }
    for i in 0usize..4 {
        let mut dot: f32 = 0.0f32;
        for j in 0usize..4 {
            dot += grad[i * 4 + j] * (history[i * 4 + j] - hc_eps);
        }
        for j in 0usize..4 {
            da[8 + i * 4 + j] = (history[i * 4 + j] - hc_eps) * (grad[i * 4 + j] - dot);
        }
    }
    let mut dz = Array::<f32>::new(24usize);
    let mut projection_dot: f32 = 0.0f32;
    for p in 0usize..24 {
        let mut group: usize = 2usize;
        if p < 4 {
            group = 0;
        } else {
            if p < 8 {
                group = 1;
            }
        }
        dz[p] = da[p] * scale[group];
        projection_dot += dz[p] * z[p];
        if lane == 0 {
            dz_out[t * 24 + p] = dz[p];
            da_out[t * 24 + p] = da[p];
        }
    }
    let inv = statistics[t * 64];
    let mut c = lane;
    while c < width {
        let stream = c / hidden as usize;
        let d = c % hidden as usize;
        let mut mapped: f32 = 0.0f32;
        for p in 0usize..24 {
            mapped += dz[p] * mapping[p * width + c];
        }
        let xv = f32::cast_from(x[t * width + c]);
        let mut value = inv * mapped - xv * inv * inv / width as f32 * projection_dot;
        value += statistics[t * 64 + 25 + stream] * f32::cast_from(du[t * hidden as usize + d]);
        for j in 0usize..4 {
            value += statistics[t * 64 + 33 + stream * 4 + j]
                * f32::cast_from(dy[(t * 4 + j) * hidden as usize + d]);
        }
        dx[t * width + c] = F::cast_from(value);
        c += PLANE_DIM as usize;
    }
}
#[ruda(launch)]
pub fn parameter_gradient<F: Float>(
    x: &Array<F>,
    statistics: &Array<f32>,
    dz: &Array<f32>,
    da: &Array<f32>,
    old_mapping: &Array<f32>,
    old_scale: &Array<f32>,
    old_base: &Array<f32>,
    df: &mut Array<f32>,
    ds: &mut Array<f32>,
    db: &mut Array<f32>,
    tokens: u32,
    hidden: u32,
    #[comptime] accumulate: bool,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    let width = hidden as usize * 4;
    let mapping_count = 24 * width;
    if i < mapping_count {
        let p = i / width;
        let c = i % width;
        let mut sum: f32 = 0.0f32;
        for t in 0..tokens as usize {
            sum += dz[t * 24 + p] * statistics[t * 64] * f32::cast_from(x[t * width + c]);
        }
        if comptime!(accumulate) {
            sum += old_mapping[i];
        }
        df[i] = sum;
    } else {
        if i < mapping_count + 24 {
            let p = i - mapping_count;
            let mut sum: f32 = 0.0f32;
            for t in 0..tokens as usize {
                sum += da[t * 24 + p];
            }
            if comptime!(accumulate) {
                sum += old_base[p];
            }
            db[p] = sum;
        } else {
            if i < mapping_count + 27 {
                let group = i - mapping_count - 24;
                let mut begin: usize = 8usize;
                let mut end: usize = 24usize;
                if group == 0 {
                    begin = 0;
                    end = 4;
                }
                if group == 1 {
                    begin = 4;
                    end = 8;
                }
                let mut sum: f32 = 0.0f32;
                for t in 0..tokens as usize {
                    for p in begin..end {
                        sum += da[t * 24 + p] * statistics[t * 64 + 1 + p];
                    }
                }
                if comptime!(accumulate) {
                    sum += old_scale[group];
                }
                ds[group] = sum;
            }
        }
    }
}
#[ruda(launch)]
pub fn head_kernel<F: Float>(
    x: &Array<F>,
    out: &mut Array<F>,
    n: u32,
    hidden: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let token = i / hidden as usize;
        let d = i % hidden as usize;
        let mut sum: f32 = 0.0f32;
        for j in 0usize..4 {
            sum += f32::cast_from(x[(token * 4 + j) * hidden as usize + d]);
        }
        out[i] = F::cast_from(sum * 0.25);
    }
}
#[ruda(launch)]
pub fn head_gradient<F: Float>(
    grad: &Array<F>,
    old: &Array<F>,
    out: &mut Array<F>,
    n: u32,
    hidden: u32,
    #[comptime] accumulate: bool,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let token = i / (4 * hidden as usize);
        let d = i % hidden as usize;
        let mut value = f32::cast_from(grad[token * hidden as usize + d]) * 0.25;
        if comptime!(accumulate) {
            value += f32::cast_from(old[i]);
        }
        out[i] = F::cast_from(value);
    }
}
#[ruda(launch)]
pub fn pre_external<F: Float>(
    residual: &Array<F>,
    dot: &Array<f32>,
    sq: &Array<f32>,
    scale: &Array<f32>,
    base: &Array<f32>,
    norm_weight: &Array<f32>,
    post: &mut Array<f32>,
    combination: &mut Array<f32>,
    layer: &mut Array<F>,
    tokens: u32,
    hidden: u32,
    splits: u32,
    rms_eps: f32,
    pre_eps: f32,
    sink_eps: f32,
    post_mult: f32,
    norm_eps: f32,
    iterations: u32,
    #[comptime] has_sq: bool,
    #[comptime] prenorm: bool,
    #[comptime] normalize_output: bool,
    #[define(F)] _dtype: StorageType,
) {
    let token = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let width = hidden as usize * 4;
    let mut square: f32 = 0.0f32;
    if comptime!(has_sq) {
        for split in 0..splits as usize {
            square += sq[split * tokens as usize + token];
        }
    } else {
        let mut j = lane;
        while j < width {
            let value = f32::cast_from(residual[token * width + j]);
            square += value * value;
            j += PLANE_DIM as usize;
        }
        square = plane_sum(square);
    }
    let mut inv = 1.0 / (square / width as f32 + rms_eps).sqrt();
    if comptime!(prenorm) {
        inv = 1.0;
    }
    let mut z = Array::<f32>::new(24usize);
    for p in 0usize..24 {
        let mut sum: f32 = 0.0f32;
        for split in 0..splits as usize {
            sum += dot[(split * tokens as usize + token) * 24 + p];
        }
        z[p] = sum * inv;
    }
    let mut matrix = Array::<f32>::new(16usize);
    initial_matrix(&z, scale, base, &mut matrix, sink_eps);
    normalize_matrix(&mut matrix, sink_eps, true);
    for step in 1..iterations {
        normalize_matrix(&mut matrix, sink_eps, false);
        normalize_matrix(&mut matrix, sink_eps, true);
    }
    let mut output_square: f32 = 0.0f32;
    let mut d = lane;
    while d < hidden as usize {
        let mut value: f32 = 0.0f32;
        for stream in 0usize..4 {
            value = fma(
                sigmoid(fma(z[stream], scale[0], base[stream])) + pre_eps,
                f32::cast_from(residual[(token * 4 + stream) * hidden as usize + d]),
                value,
            );
        }
        output_square += value * value;
        layer[token * hidden as usize + d] = F::cast_from(value);
        d += PLANE_DIM as usize;
    }
    if comptime!(normalize_output) {
        let output_inv = 1.0 / (plane_sum(output_square) / hidden as f32 + norm_eps).sqrt();
        d = lane;
        while d < hidden as usize {
            let offset = token * hidden as usize + d;
            layer[offset] =
                F::cast_from(f32::cast_from(layer[offset]) * output_inv * norm_weight[d]);
            d += PLANE_DIM as usize;
        }
    }
    if lane == 0 {
        for j in 0usize..4 {
            post[token * 4 + j] = post_mult * sigmoid(fma(z[4 + j], scale[1], base[4 + j]));
        }
        for j in 0usize..16 {
            combination[token * 16 + j] = matrix[j];
        }
    }
}
#[ruda(launch)]
pub fn post_external<F: Float>(
    x: &Array<F>,
    branch: &Array<F>,
    post: &Array<f32>,
    matrix: &Array<f32>,
    out: &mut Array<F>,
    n: u32,
    hidden: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let token = i / (4 * hidden as usize);
        let stream = i / hidden as usize % 4;
        let d = i % hidden as usize;
        let mut value =
            post[token * 4 + stream] * f32::cast_from(branch[token * hidden as usize + d]);
        for j in 0usize..4 {
            value = fma(
                matrix[token * 16 + j * 4 + stream],
                f32::cast_from(x[(token * 4 + j) * hidden as usize + d]),
                value,
            );
        }
        out[i] = F::cast_from(value);
    }
}
