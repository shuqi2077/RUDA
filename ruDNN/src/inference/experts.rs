//! Native routing and fused expert execution, independent of CUDA vendor MoE.
//! Dense, FP8 and block-scaled FP4/FP8 weights share the same mathematical path.
use super::linear::Matrix;
use super::{InferenceError, alloc, check, float};
use ruda_core::tensor::DType;
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, contiguous::into_contiguous},
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterScore {
    Softmax,
    Sigmoid,
    Identity,
}
#[derive(Clone, Copy, Debug)]
pub struct RoutingOptions {
    pub select_raw_logits: bool,
    pub softmax_selected: bool,
    pub normalization_epsilon: f32,
    pub top_k: usize,
    pub groups: usize,
    pub selected_groups: usize,
    pub group_score_top_k: usize,
    pub scoring: RouterScore,
    pub normalize: bool,
    pub routed_scale: f32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingMethod {
    Default,
    Renormalize,
    DeepSeekV3,
    Llama4,
    RenormalizeNaive,
    TopK,
    SigmoidRenorm,
    MiniMax2,
    Sigmoid,
    TopKSigmoid,
}
impl RoutingOptions {
    pub fn method(method: RoutingMethod, top_k: usize) -> Self {
        let mut o = Self {
            select_raw_logits: false,
            softmax_selected: false,
            normalization_epsilon: 0.0,
            top_k,
            groups: 1,
            selected_groups: 1,
            group_score_top_k: 1,
            scoring: RouterScore::Softmax,
            normalize: false,
            routed_scale: 1.0,
        };
        match method {
            RoutingMethod::Default => {}
            RoutingMethod::Renormalize => {
                o.select_raw_logits = true;
                o.softmax_selected = true;
            }
            RoutingMethod::DeepSeekV3 => {
                o.scoring = RouterScore::Sigmoid;
                o.normalize = true;
            }
            RoutingMethod::Llama4 => {
                o.scoring = RouterScore::Sigmoid;
                o.select_raw_logits = true;
                o.top_k = 1;
            }
            RoutingMethod::RenormalizeNaive => o.normalize = true,
            RoutingMethod::TopK => o.scoring = RouterScore::Identity,
            RoutingMethod::SigmoidRenorm => {
                o.scoring = RouterScore::Sigmoid;
                o.normalize = true;
            }
            RoutingMethod::MiniMax2 => {
                o.scoring = RouterScore::Sigmoid;
                o.normalize = true;
                o.normalization_epsilon = 1e-20;
            }
            RoutingMethod::Sigmoid => o.scoring = RouterScore::Sigmoid,
            RoutingMethod::TopKSigmoid => {
                o.scoring = RouterScore::Sigmoid;
                o.select_raw_logits = true;
            }
        }
        o
    }
}
pub struct Routes<R: Runtime> {
    pub experts: RudaTensor<R>,
    pub weights: RudaTensor<R>,
}
/// Correction bias affects selection only; route weights always come from the
/// uncorrected scores. Group scores sum their highest group_score_top_k scores.
pub fn route<R: Runtime>(
    scores: &RudaTensor<R>,
    correction_bias: Option<&RudaTensor<R>>,
    options: RoutingOptions,
) -> Result<Routes<R>, InferenceError> {
    let scores = super::norm::cast(scores, DType::F32)?;
    if scores.meta.num_dims() != 2 {
        return Err(InferenceError("routing scores must be [tokens,experts]"));
    }
    let tokens = scores.meta.shape()[0];
    let experts = scores.meta.shape()[1];
    if experts == 0
        || options.top_k == 0
        || options.top_k > experts
        || options.groups == 0
        || experts % options.groups != 0
        || options.selected_groups == 0
        || options.selected_groups > options.groups
        || options.group_score_top_k == 0
        || options.group_score_top_k > experts / options.groups
        || options.top_k > options.selected_groups * (experts / options.groups)
        || !options.routed_scale.is_finite()
        || !options.normalization_epsilon.is_finite()
        || options.normalization_epsilon < 0.0
    {
        return Err(InferenceError("invalid grouped routing configuration"));
    }
    let bias = correction_bias
        .map(|t| super::norm::cast(t, DType::F32))
        .transpose()?;
    if let Some(b) = &bias {
        check(b, &scores)?;
        if b.meta.shape()[..] != [experts] {
            return Err(InferenceError("routing bias must be [experts]"));
        }
    }
    let ids = alloc(&scores, [tokens, options.top_k], DType::I32);
    let weights = alloc(&scores, [tokens, options.top_k], DType::F32);
    let group_scores = alloc(&scores, [tokens, options.groups], DType::F32);
    let dim = RudaDim::new(scores.client.properties(), tokens.max(1));
    let count = kernel_dsl::calculate_ruda_count_elemwise(&scores.client, tokens.max(1), dim);
    if tokens > 0 {
        router::launch::<R>(
            &scores.client,
            count,
            dim,
            scores.clone().into_array_arg(),
            bias.as_ref().unwrap_or(&scores).clone().into_array_arg(),
            group_scores.into_array_arg(),
            ids.clone().into_array_arg(),
            weights.clone().into_array_arg(),
            tokens as u32,
            experts as u32,
            options.routed_scale,
            options.normalization_epsilon,
            options.top_k,
            options.groups,
            options.selected_groups,
            options.group_score_top_k,
            match options.scoring {
                RouterScore::Softmax => 0,
                RouterScore::Sigmoid => 1,
                RouterScore::Identity => 2,
            },
            bias.is_some(),
            options.normalize,
            options.select_raw_logits,
            options.softmax_selected,
        );
    }
    Ok(Routes {
        experts: ids,
        weights,
    })
}
#[ruda(launch)]
pub fn router(
    scores: &Array<f32>,
    bias: &Array<f32>,
    group_scores: &mut Array<f32>,
    ids: &mut Array<i32>,
    weights: &mut Array<f32>,
    tokens: u32,
    experts: u32,
    scale: f32,
    epsilon: f32,
    #[comptime] top_k: usize,
    #[comptime] groups: usize,
    #[comptime] selected_groups: usize,
    #[comptime] group_top_k: usize,
    #[comptime] scoring: u32,
    #[comptime] biased: bool,
    #[comptime] normalized: bool,
    #[comptime] raw_selection: bool,
    #[comptime] selected_softmax: bool,
) {
    let row = ABSOLUTE_POS as usize;
    if row < tokens as usize {
        let mut maximum = f32::NEG_INFINITY;
        let mut denominator: f32 = 0.0f32;
        for e in 0..experts as usize {
            maximum = f32::max(maximum, scores[row * experts as usize + e]);
        }
        if comptime!(scoring == 0) {
            for e in 0..experts as usize {
                denominator += (scores[row * experts as usize + e] - maximum).exp();
            }
        }
        let per = experts as usize / groups;
        let mut gs = Array::<f32>::new(groups);
        let mut chosen_groups = Array::<u32>::new(selected_groups);
        let mut chosen = Array::<u32>::new(top_k);
        let mut raw = Array::<f32>::new(top_k);
        for g in 0usize..groups {
            let mut sum: f32 = 0.0f32;
            let mut previous = f32::INFINITY;
            let mut previous_id: usize = 0usize;
            for rank in 0usize..group_top_k {
                let mut best = f32::NEG_INFINITY;
                let mut best_id = experts as usize;
                for j in 0..per {
                    let e = g * per + j;
                    let value = router_score(
                        scores[row * experts as usize + e],
                        maximum,
                        denominator,
                        scoring,
                    );
                    let mut corrected = value;
                    if comptime!(raw_selection) {
                        corrected = scores[row * experts as usize + e];
                    }
                    if comptime!(biased) {
                        corrected += bias[e];
                    }
                    let eligible = rank == 0
                        || corrected < previous
                        || (corrected == previous && e > previous_id);
                    if eligible && (corrected > best || (corrected == best && e < best_id)) {
                        best = corrected;
                        best_id = e;
                    }
                }
                sum += best;
                previous = best;
                previous_id = best_id;
            }
            gs[g] = sum;
            group_scores[row * groups + g] = sum;
        }
        for rank in 0usize..selected_groups {
            let mut best = f32::NEG_INFINITY;
            let mut best_id = groups;
            for g in 0usize..groups {
                let mut used = false;
                for p in 0..rank {
                    used = used || chosen_groups[p] == g as u32;
                }
                if !used && (gs[g] > best || (gs[g] == best && g < best_id)) {
                    best = gs[g];
                    best_id = g;
                }
            }
            chosen_groups[rank] = best_id as u32;
        }
        let mut total: f32 = 0.0f32;
        for rank in 0usize..top_k {
            let mut best = f32::NEG_INFINITY;
            let mut best_id = experts as usize;
            let mut best_raw: f32 = 0.0f32;
            for e in 0..experts as usize {
                let mut live = false;
                for g in 0usize..selected_groups {
                    live = live || chosen_groups[g] == (e / per) as u32;
                }
                for p in 0..rank {
                    if chosen[p] == e as u32 {
                        live = false;
                    }
                }
                let value = router_score(
                    scores[row * experts as usize + e],
                    maximum,
                    denominator,
                    scoring,
                );
                let mut corrected = value;
                if comptime!(raw_selection) {
                    corrected = scores[row * experts as usize + e];
                }
                if comptime!(biased) {
                    corrected += bias[e];
                }
                if live && (corrected > best || (corrected == best && e < best_id)) {
                    best = corrected;
                    best_id = e;
                    best_raw = value;
                    if comptime!(selected_softmax) {
                        best_raw = scores[row * experts as usize + e];
                    }
                }
            }
            chosen[rank] = best_id as u32;
            raw[rank] = best_raw;
            total += best_raw;
        }
        if comptime!(selected_softmax) {
            let mut maximum = f32::NEG_INFINITY;
            for rank in 0usize..top_k {
                maximum = f32::max(maximum, raw[rank]);
            }
            total = 0.0;
            for rank in 0usize..top_k {
                raw[rank] = (raw[rank] - maximum).exp();
                total += raw[rank];
            }
        }
        for rank in 0usize..top_k {
            ids[row * top_k + rank] = chosen[rank] as i32;
            let mut value = raw[rank];
            if comptime!(normalized || selected_softmax) {
                if total + epsilon != 0.0 {
                    value /= total + epsilon;
                }
            }
            weights[row * top_k + rank] = value * scale;
        }
    }
}
#[ruda]
fn router_score(x: f32, maximum: f32, denominator: f32, #[comptime] kind: u32) -> f32 {
    let mut y = x;
    if comptime!(kind == 0) {
        y = (x - maximum).exp() / denominator;
    }
    if comptime!(kind == 1) {
        y = 1.0 / (1.0 + (-x).exp());
    }
    y
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertActivation {
    Standard,
    SwigluBias,
    SwigluStep,
    Situ,
    ClampedReluSquared,
    PowLu,
}
#[derive(Clone, Copy, Debug)]
pub struct ExpertOptions {
    pub mode: ExpertActivation,
    pub activation: super::activation::Activation,
    pub gated: bool,
    pub weight_on_input: bool,
    pub activation_limit: Option<f32>,
    pub output_dtype: DType,
}
impl Default for ExpertOptions {
    fn default() -> Self {
        Self {
            mode: ExpertActivation::Standard,
            activation: super::activation::Activation::Silu,
            gated: true,
            weight_on_input: false,
            activation_limit: None,
            output_dtype: DType::BF16,
        }
    }
}
/// Optional low-precision activation boundaries and device per-expert activation
/// parameters [E,5]: sigmoid alpha / SiTU beta, up bias, clamp,
/// SiTU linear beta, PowLU exponent numerator.
pub struct ExpertNumerics<'a, R: Runtime> {
    pub activation_parameters: Option<&'a RudaTensor<R>>,
    pub input_quantizer: Option<(
        rublas::tensor_float_quant::QuantOptions,
        Option<&'a RudaTensor<R>>,
    )>,
    pub hidden_quantizer: Option<(
        rublas::tensor_float_quant::QuantOptions,
        Option<&'a RudaTensor<R>>,
    )>,
}
impl<R: Runtime> Default for ExpertNumerics<'_, R> {
    fn default() -> Self {
        Self {
            activation_parameters: None,
            input_quantizer: None,
            hidden_quantizer: None,
        }
    }
}
/// W1 [E,2I,K] (gated) or [E,I,K], W2 [E,N,I]. Route order is
/// deterministic, reductions FP32, and shared-expert contributions can be added
/// through `shared_output`. Per-expert biases are optional.
pub fn fused_moe<R: Runtime>(
    input: &RudaTensor<R>,
    w1: Matrix<'_, R>,
    w2: Matrix<'_, R>,
    routes: &Routes<R>,
    bias1: Option<&RudaTensor<R>>,
    bias2: Option<&RudaTensor<R>>,
    shared_output: Option<&RudaTensor<R>>,
    options: ExpertOptions,
) -> Result<RudaTensor<R>, InferenceError> {
    fused_moe_with_numerics(
        input,
        w1,
        w2,
        routes,
        bias1,
        bias2,
        shared_output,
        options,
        ExpertNumerics::default(),
    )
}
pub fn fused_moe_with_numerics<R: Runtime>(
    input: &RudaTensor<R>,
    w1: Matrix<'_, R>,
    w2: Matrix<'_, R>,
    routes: &Routes<R>,
    bias1: Option<&RudaTensor<R>>,
    bias2: Option<&RudaTensor<R>>,
    shared_output: Option<&RudaTensor<R>>,
    options: ExpertOptions,
    numerics: ExpertNumerics<'_, R>,
) -> Result<RudaTensor<R>, InferenceError> {
    let mut x = super::norm::cast(input, DType::F32)?;
    if let Some((recipe, global)) = numerics.input_quantizer {
        x = rublas::tensor_float_quant::quantize(&x, global, None, recipe)
            .and_then(|q| q.dequantize(DType::F32))
            .map_err(|e| InferenceError(e.0))?;
    }
    let a = w1.decode()?;
    let b = w2.decode()?;
    for t in [&a, &b, &routes.experts, &routes.weights] {
        check(t, &x)?;
    }
    if x.meta.num_dims() != 2
        || a.meta.num_dims() != 3
        || b.meta.num_dims() != 3
        || a.meta.shape()[0] != b.meta.shape()[0]
        || a.meta.shape()[2] != x.meta.shape()[1]
        || a.meta.shape()[1] != b.meta.shape()[2] * if options.gated { 2 } else { 1 }
        || routes.experts.dtype != DType::I32
        || routes.weights.dtype != DType::F32
        || routes.experts.meta.num_dims() != 2
        || routes.experts.meta.shape() != routes.weights.meta.shape()
        || routes.experts.meta.shape()[0] != x.meta.shape()[0]
    {
        return Err(InferenceError("fused expert dimensions/routes mismatch"));
    }
    if options.mode == ExpertActivation::ClampedReluSquared
        && (options.gated
            || options.activation_limit.is_none() && numerics.activation_parameters.is_none())
        || matches!(
            options.mode,
            ExpertActivation::SwigluBias
                | ExpertActivation::SwigluStep
                | ExpertActivation::Situ
                | ExpertActivation::PowLu
        ) && !options.gated
    {
        return Err(InferenceError(
            "expert activation mode/gated/clamp contract mismatch",
        ));
    }
    let tokens = x.meta.shape()[0];
    let width = x.meta.shape()[1];
    let experts = a.meta.shape()[0];
    let inner = b.meta.shape()[2];
    let columns = b.meta.shape()[1];
    let top_k = routes.experts.meta.shape()[1];
    if inner == 0
        || width == 0
        || columns == 0
        || top_k == 0
        || options
            .activation_limit
            .is_some_and(|v| !v.is_finite() || v <= 0.0)
    {
        return Err(InferenceError("invalid expert dimensions/activation clamp"));
    }
    let bias1 = bias1
        .map(|t| super::norm::cast(t, DType::F32))
        .transpose()?;
    let bias2 = bias2
        .map(|t| super::norm::cast(t, DType::F32))
        .transpose()?;
    let shared = shared_output
        .map(|t| super::norm::cast(t, DType::F32))
        .transpose()?;
    for (v, shape) in [
        (bias1.as_ref(), vec![experts, a.meta.shape()[1]]),
        (bias2.as_ref(), vec![experts, columns]),
        (shared.as_ref(), vec![tokens, columns]),
    ] {
        if let Some(v) = v {
            check(v, &x)?;
            if v.meta.shape()[..] != shape {
                return Err(InferenceError("expert bias/shared output shape mismatch"));
            }
        }
    }
    let hn = tokens
        .checked_mul(top_k)
        .and_then(|n| n.checked_mul(inner))
        .filter(|n| *n <= u32::MAX as usize)
        .ok_or(InferenceError("expert intermediate exceeds U32"))?;
    let on = tokens
        .checked_mul(columns)
        .filter(|n| *n <= u32::MAX as usize)
        .ok_or(InferenceError("expert output exceeds U32"))?;
    let parameters = if let Some(p) = numerics.activation_parameters {
        check(p, &x)?;
        if p.dtype != DType::F32 || p.meta.shape()[..] != [experts, 5] {
            return Err(InferenceError(
                "expert activation parameters must be F32 [E,5]",
            ));
        }
        p.clone()
    } else {
        let alpha = if options.mode == ExpertActivation::Situ {
            4.0
        } else {
            1.0
        };
        let row = [
            alpha,
            0.0,
            options.activation_limit.unwrap_or(f32::INFINITY),
            25.0,
            2.5,
        ];
        let host: Vec<f32> = (0..experts).flat_map(|_| row).collect();
        RudaTensor::new_contiguous(
            x.client.clone(),
            x.device.clone(),
            ruda_core::tensor::Shape::from([experts, 5]),
            x.client.create_from_slice(bytemuck::cast_slice(&host)),
            DType::F32,
        )
    };
    let hidden = alloc(&x, [tokens, top_k, inner], DType::F32);
    let output = alloc(&x, [tokens, columns], options.output_dtype);
    float(&output)?;
    if hn > 0 {
        let dim = RudaDim::new(x.client.properties(), hn);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, hn, dim);
        expert_first::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            a.into_array_arg(),
            routes.experts.clone().into_array_arg(),
            routes.weights.clone().into_array_arg(),
            bias1.as_ref().unwrap_or(&x).clone().into_array_arg(),
            parameters.into_array_arg(),
            hidden.clone().into_array_arg(),
            tokens as u32,
            width as u32,
            inner as u32,
            experts as u32,
            top_k as u32,
            options.activation_limit.unwrap_or(f32::INFINITY),
            options.gated,
            options.activation.code(),
            match options.mode {
                ExpertActivation::Standard => 0,
                ExpertActivation::SwigluBias => 1,
                ExpertActivation::SwigluStep => 2,
                ExpertActivation::Situ => 3,
                ExpertActivation::ClampedReluSquared => 4,
                ExpertActivation::PowLu => 5,
            },
            bias1.is_some(),
            options.weight_on_input,
        );
    }
    let hidden = if let Some((recipe, global)) = numerics.hidden_quantizer {
        rublas::tensor_float_quant::quantize(&hidden, global, None, recipe)
            .and_then(|q| q.dequantize(DType::F32))
            .map_err(|e| InferenceError(e.0))?
    } else {
        hidden
    };
    if on > 0 {
        let dim = RudaDim::new(x.client.properties(), on);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, on, dim);
        expert_second::launch::<R>(
            &x.client,
            count,
            dim,
            hidden.into_array_arg(),
            b.into_array_arg(),
            routes.experts.clone().into_array_arg(),
            routes.weights.clone().into_array_arg(),
            bias2.as_ref().unwrap_or(&x).clone().into_array_arg(),
            shared.as_ref().unwrap_or(&x).clone().into_array_arg(),
            output.clone().into_array_arg(),
            tokens as u32,
            inner as u32,
            columns as u32,
            experts as u32,
            top_k as u32,
            bias2.is_some(),
            shared.is_some(),
            !options.weight_on_input,
            options.output_dtype.into(),
        );
    }
    Ok(output)
}
#[ruda(launch)]
pub fn expert_first(
    x: &Array<f32>,
    w: &Array<f32>,
    ids: &Array<i32>,
    route_weights: &Array<f32>,
    bias: &Array<f32>,
    parameters: &Array<f32>,
    out: &mut Array<f32>,
    tokens: u32,
    width: u32,
    inner: u32,
    experts: u32,
    top_k: u32,
    limit: f32,
    #[comptime] gated: bool,
    #[comptime] activation: u32,
    #[comptime] mode: u32,
    #[comptime] biased: bool,
    #[comptime] weight_input: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < tokens as usize * top_k as usize * inner as usize {
        let col = i % inner as usize;
        let route = i / inner as usize;
        let token = route / top_k as usize;
        let e = ids[route];
        let mut result: f32 = 0.0f32;
        if e >= 0 && e < experts as i32 {
            let mut stride = inner as usize;
            if comptime!(gated) {
                stride *= 2;
            }
            let mut gate: f32 = 0.0f32;
            let mut value: f32 = 0.0f32;
            if comptime!(biased) {
                gate = bias[e as usize * stride + col];
                if comptime!(gated) {
                    value = bias[e as usize * stride + inner as usize + col];
                }
            }
            for k in 0..width as usize {
                let mut xv = x[token * width as usize + k];
                if comptime!(weight_input) {
                    xv *= route_weights[route];
                }
                gate = fma(
                    xv,
                    w[(e as usize * stride + col) * width as usize + k],
                    gate,
                );
                if comptime!(gated) {
                    value = fma(
                        xv,
                        w[(e as usize * stride + inner as usize + col) * width as usize + k],
                        value,
                    );
                }
            }
            let p = e as usize * 5;
            let bound = f32::min(limit, parameters[p + 2]);
            if comptime!(mode == 0) {
                gate = f32::min(gate, bound);
                result = crate::inference::activation::apply(gate, activation);
                if comptime!(gated) {
                    value = f32::min(f32::max(value, -bound), bound);
                    result *= value;
                }
            }
            if comptime!(mode == 1) {
                gate = f32::min(gate, bound);
                value = f32::min(f32::max(value, -bound), bound) + parameters[p + 1];
                result = gate / (1.0 + (-parameters[p] * gate).exp()) * value;
            }
            if comptime!(mode == 2) {
                result = f32::min(gate / (1.0 + (-gate).exp()), bound)
                    * f32::min(f32::max(value, -bound), bound);
            }
            if comptime!(mode == 3) {
                let beta = parameters[p];
                let linear = parameters[p + 3];
                result = beta * (gate / beta).tanh() / (1.0 + (-gate).exp())
                    * linear
                    * (value / linear).tanh();
            }
            if comptime!(mode == 4) {
                let z = f32::max(bound * (gate / bound).tanh(), 0.0);
                result = z * z;
            }
            if comptime!(mode == 5) {
                let mut activated = gate / (1.0 + (-gate).exp());
                if gate > 0.0 {
                    activated = f32::powf(gate, parameters[p + 4] / (gate.sqrt() + 1.0))
                        / (1.0 + (-gate).exp());
                }
                result = activated * value;
            }
        }
        out[i] = result;
    }
}
#[ruda(launch)]
pub fn expert_second<O: Float>(
    hidden: &Array<f32>,
    w: &Array<f32>,
    ids: &Array<i32>,
    route_weights: &Array<f32>,
    bias: &Array<f32>,
    shared: &Array<f32>,
    out: &mut Array<O>,
    tokens: u32,
    inner: u32,
    columns: u32,
    experts: u32,
    top_k: u32,
    #[comptime] biased: bool,
    #[comptime] has_shared: bool,
    #[comptime] weight_output: bool,
    #[define(O)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < tokens as usize * columns as usize {
        let token = i / columns as usize;
        let col = i % columns as usize;
        let mut sum: f32 = 0.0f32;
        if comptime!(has_shared) {
            sum = shared[i];
        }
        for rank in 0..top_k as usize {
            let route = token * top_k as usize + rank;
            let e = ids[route];
            if e >= 0 && e < experts as i32 {
                let mut value: f32 = 0.0f32;
                if comptime!(biased) {
                    value = bias[e as usize * columns as usize + col];
                }
                for k in 0..inner as usize {
                    value = fma(
                        hidden[route * inner as usize + k],
                        w[(e as usize * columns as usize + col) * inner as usize + k],
                        value,
                    );
                }
                if comptime!(weight_output) {
                    value *= route_weights[route];
                }
                sum += value;
            }
        }
        out[i] = O::cast_from(sum);
    }
}

/// Convert [E,2I,K] between [gate rows, value rows] and interleaved pairs.
pub fn reorder_gated_rows<R: Runtime>(
    input: &RudaTensor<R>,
    interleave: bool,
) -> Result<RudaTensor<R>, InferenceError> {
    let x = into_contiguous(input.clone());
    check(&x, &x)?;
    float(&x)?;
    let rank = x.meta.num_dims();
    if rank < 2
        || x.meta.shape()[rank - 2] % 2 != 0
        || x.meta.shape()[rank - 2] == 0
        || x.meta.shape()[rank - 1] == 0
    {
        return Err(InferenceError("gated row reorder needs [...,2I,K]"));
    }
    let rows = x.meta.shape()[rank - 2];
    let cols = x.meta.shape()[rank - 1];
    let n = x.meta.num_elements();
    let y = alloc(&x, x.meta.shape().clone(), x.dtype);
    if n > 0 {
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        reorder::launch::<R>(
            &x.client,
            count,
            dim,
            x.clone().into_array_arg(),
            y.clone().into_array_arg(),
            n as u32,
            rows as u32,
            cols as u32,
            interleave,
            x.dtype.into(),
        );
    }
    Ok(y)
}
#[ruda(launch)]
pub fn reorder<F: Float>(
    x: &Array<F>,
    y: &mut Array<F>,
    n: u32,
    rows: u32,
    cols: u32,
    #[comptime] interleave: bool,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let col = i % cols as usize;
        let row = i / cols as usize % rows as usize;
        let group = i / (rows as usize * cols as usize);
        let mapped = (row % (rows as usize / 2)) * 2 + row / (rows as usize / 2);
        let dst = (group * rows as usize + mapped) * cols as usize + col;
        if comptime!(interleave) {
            y[dst] = x[i];
        } else {
            y[i] = x[dst];
        }
    }
}

/// Gather per-expert decoder scales into the route-major [T,K] activation rows.
pub fn expert_global_scales<R: Runtime>(
    routes: &Routes<R>,
    per_expert: &RudaTensor<R>,
) -> Result<RudaTensor<R>, InferenceError> {
    check(&routes.experts, per_expert)?;
    check(per_expert, per_expert)?;
    if routes.experts.dtype != DType::I32
        || routes.experts.meta.num_dims() != 2
        || per_expert.dtype != DType::F32
        || per_expert.meta.num_dims() != 1
    {
        return Err(InferenceError(
            "expert scales need I32 routes[T,K] and F32 scales[E]",
        ));
    }
    let out = alloc(per_expert, routes.experts.meta.shape().clone(), DType::F32);
    let n = out.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(out.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
        gather_scales::launch::<R>(
            &out.client,
            count,
            dim,
            routes.experts.clone().into_array_arg(),
            per_expert.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            per_expert.meta.num_elements() as u32,
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn gather_scales(
    ids: &Array<i32>,
    scales: &Array<f32>,
    out: &mut Array<f32>,
    n: u32,
    experts: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let id = ids[i];
        let mut value: f32 = 1.0f32;
        if id >= 0 && id < experts as i32 {
            value = scales[id as usize];
        }
        out[i] = value;
    }
}

/// Retained grouped router. Group-score workspace and result tensors are owned
/// once, rather than reallocated while recording a graph.
pub struct PreparedRoutes<R: Runtime> {
    scores: super::PreparedPack<R>,
    scores_f32: RudaTensor<R>,
    bias: Option<super::PreparedPack<R>>,
    bias_f32: Option<RudaTensor<R>>,
    group_scores: RudaTensor<R>,
    pub output: Routes<R>,
    options: RoutingOptions,
}
impl<R: Runtime> PreparedRoutes<R> {
    pub fn prepare(
        scores: &RudaTensor<R>,
        bias: Option<&RudaTensor<R>>,
        o: RoutingOptions,
    ) -> Result<Self, InferenceError> {
        let scores = super::PreparedPack::prepare(scores)?;
        let x = &scores.output;
        float(x)?;
        if x.meta.num_dims() != 2 {
            return Err(InferenceError("routing scores must be [tokens,experts]"));
        }
        let t = x.meta.shape()[0];
        let e = x.meta.shape()[1];
        if e == 0
            || o.top_k == 0
            || o.top_k > e
            || o.groups == 0
            || e % o.groups != 0
            || o.selected_groups == 0
            || o.selected_groups > o.groups
            || o.group_score_top_k == 0
            || o.group_score_top_k > e / o.groups
            || o.top_k > o.selected_groups * (e / o.groups)
            || !o.routed_scale.is_finite()
            || !o.normalization_epsilon.is_finite()
            || o.normalization_epsilon < 0.0
            || t.checked_mul(o.top_k.max(o.groups))
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
        {
            return Err(InferenceError("invalid prepared routing configuration"));
        }
        let bias = bias.map(super::PreparedPack::prepare).transpose()?;
        let bias_f32 = if let Some(p) = &bias {
            check(&p.output, x)?;
            float(&p.output)?;
            if p.output.meta.shape()[..] != [e] {
                return Err(InferenceError("routing bias must be [experts]"));
            }
            Some(alloc(x, [e], DType::F32))
        } else {
            None
        };
        let scores_f32 = alloc(x, x.meta.shape().clone(), DType::F32);
        let group_scores = alloc(x, [t, o.groups], DType::F32);
        let output = Routes {
            experts: alloc(x, [t, o.top_k], DType::I32),
            weights: alloc(x, [t, o.top_k], DType::F32),
        };
        Ok(Self {
            scores,
            scores_f32,
            bias,
            bias_f32,
            group_scores,
            output,
            options: o,
        })
    }
    pub unsafe fn run(&self) -> Result<&Routes<R>, InferenceError> {
        unsafe {
            self.scores.run();
            super::norm::cast_into(&self.scores.output, &self.scores_f32)?;
            if let (Some(p), Some(out)) = (&self.bias, &self.bias_f32) {
                p.run();
                super::norm::cast_into(&p.output, out)?;
            }
        }
        let x = &self.scores_f32;
        let o = self.options;
        let t = x.meta.shape()[0];
        let e = x.meta.shape()[1];
        if t > 0 {
            let dim = RudaDim::new(x.client.properties(), t);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, t, dim);
            router::launch::<R>(
                &x.client,
                count,
                dim,
                x.clone().into_array_arg(),
                self.bias_f32.as_ref().unwrap_or(x).clone().into_array_arg(),
                self.group_scores.clone().into_array_arg(),
                self.output.experts.clone().into_array_arg(),
                self.output.weights.clone().into_array_arg(),
                t as u32,
                e as u32,
                o.routed_scale,
                o.normalization_epsilon,
                o.top_k,
                o.groups,
                o.selected_groups,
                o.group_score_top_k,
                match o.scoring {
                    RouterScore::Softmax => 0,
                    RouterScore::Sigmoid => 1,
                    RouterScore::Identity => 2,
                },
                self.bias.is_some(),
                o.normalize,
                o.select_raw_logits,
                o.softmax_selected,
            );
        }
        Ok(&self.output)
    }
}
/// A complete two-GEMM MoE replay with native dynamic routing tables, decoded
/// weights, optional biases/shared experts and low-precision activation edges.
pub struct PreparedExperts<'a, R: Runtime> {
    input: super::PreparedPack<R>,
    x: RudaTensor<R>,
    w1: super::linear::PreparedMatrix<'a, R>,
    w2: super::linear::PreparedMatrix<'a, R>,
    ids: super::metadata::PreparedIndices<R>,
    weights: super::PreparedPack<R>,
    weights_f32: RudaTensor<R>,
    parameters: Vec<(super::PreparedPack<R>, RudaTensor<R>)>,
    bias1: Option<RudaTensor<R>>,
    bias2: Option<RudaTensor<R>>,
    shared: Option<RudaTensor<R>>,
    activation_parameters: RudaTensor<R>,
    hidden: RudaTensor<R>,
    input_quantizer: Option<rublas::tensor_float_quant::BlockFloatTensor<R>>,
    hidden_quantizer: Option<rublas::tensor_float_quant::BlockFloatTensor<R>>,
    pub output: RudaTensor<R>,
    options: ExpertOptions,
}
impl<'a, R: Runtime> PreparedExperts<'a, R> {
    pub fn prepare(
        input: &RudaTensor<R>,
        w1: Matrix<'a, R>,
        w2: Matrix<'a, R>,
        routes: &Routes<R>,
        bias1: Option<&RudaTensor<R>>,
        bias2: Option<&RudaTensor<R>>,
        shared: Option<&RudaTensor<R>>,
        options: ExpertOptions,
        numerics: ExpertNumerics<'_, R>,
    ) -> Result<Self, InferenceError> {
        let input = super::PreparedPack::prepare(input)?;
        let source = &input.output;
        float(source)?;
        let x = alloc(source, source.meta.shape().clone(), DType::F32);
        let w1 = super::linear::PreparedMatrix::prepare(w1)?;
        let w2 = super::linear::PreparedMatrix::prepare(w2)?;
        let a = &w1.output;
        let b = &w2.output;
        let ids = super::metadata::PreparedIndices::prepare(&routes.experts, DType::I32)?;
        let weights = super::PreparedPack::prepare(&routes.weights)?;
        for t in [a, b, &ids.output, &weights.output] {
            check(t, &x)?;
        }
        float(&weights.output)?;
        if x.meta.num_dims() != 2
            || a.meta.num_dims() != 3
            || b.meta.num_dims() != 3
            || a.meta.shape()[0] != b.meta.shape()[0]
            || a.meta.shape()[2] != x.meta.shape()[1]
            || a.meta.shape()[1] != b.meta.shape()[2] * if options.gated { 2 } else { 1 }
            || ids.output.meta.num_dims() != 2
            || ids.output.meta.shape() != weights.output.meta.shape()
            || ids.output.meta.shape()[0] != x.meta.shape()[0]
        {
            return Err(InferenceError("prepared expert dimensions/routes mismatch"));
        }
        if options.mode == ExpertActivation::ClampedReluSquared
            && (options.gated
                || options.activation_limit.is_none() && numerics.activation_parameters.is_none())
            || matches!(
                options.mode,
                ExpertActivation::SwigluBias
                    | ExpertActivation::SwigluStep
                    | ExpertActivation::Situ
                    | ExpertActivation::PowLu
            ) && !options.gated
        {
            return Err(InferenceError("expert activation/gating contract mismatch"));
        }
        let (t, k, e, i, n, top) = (
            x.meta.shape()[0],
            x.meta.shape()[1],
            a.meta.shape()[0],
            b.meta.shape()[2],
            b.meta.shape()[1],
            ids.output.meta.shape()[1],
        );
        if k == 0
            || e == 0
            || i == 0
            || n == 0
            || top == 0
            || options
                .activation_limit
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
            || t.checked_mul(top)
                .and_then(|n| n.checked_mul(i))
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
            || t.checked_mul(n)
                .filter(|n| *n <= u32::MAX as usize)
                .is_none()
        {
            return Err(InferenceError("prepared expert workspace/clamp invalid"));
        }
        let mut parameters = Vec::new();
        let mut parameter =
            |p: &RudaTensor<R>, shape: &[usize]| -> Result<RudaTensor<R>, InferenceError> {
                super::check_view(p, &x)?;
                float(p)?;
                if p.meta.shape()[..] != *shape {
                    return Err(InferenceError("expert parameter shape mismatch"));
                }
                let pack = super::PreparedPack::prepare(p)?;
                let out = alloc(&x, p.meta.shape().clone(), DType::F32);
                parameters.push((pack, out.clone()));
                Ok(out)
            };
        let bias1 = bias1
            .map(|p| parameter(p, &[e, a.meta.shape()[1]]))
            .transpose()?;
        let bias2 = bias2.map(|p| parameter(p, &[e, n])).transpose()?;
        let shared = shared.map(|p| parameter(p, &[t, n])).transpose()?;
        let activation_parameters = if let Some(p) = numerics.activation_parameters {
            parameter(p, &[e, 5])?
        } else {
            let alpha = if options.mode == ExpertActivation::Situ {
                4.0
            } else {
                1.0
            };
            let row = [
                alpha,
                0.0,
                options.activation_limit.unwrap_or(f32::INFINITY),
                25.0,
                2.5,
            ];
            let host: Vec<f32> = (0..e).flat_map(|_| row).collect();
            RudaTensor::new_contiguous(
                x.client.clone(),
                x.device.clone(),
                ruda_core::tensor::Shape::from([e, 5]),
                x.client.create_from_slice(bytemuck::cast_slice(&host)),
                DType::F32,
            )
        };
        let hidden = alloc(&x, [t, top, i], DType::F32);
        let output = alloc(&x, [t, n], options.output_dtype);
        float(&output)?;
        let weights_f32 = alloc(&x, [t, top], DType::F32);
        let input_quantizer = numerics
            .input_quantizer
            .map(|(recipe, global)| {
                rublas::tensor_float_quant::allocate_quantized(&x, global, recipe)
                    .map_err(|e| InferenceError(e.0))
            })
            .transpose()?;
        let hidden_quantizer = numerics
            .hidden_quantizer
            .map(|(recipe, global)| {
                rublas::tensor_float_quant::allocate_quantized(&hidden, global, recipe)
                    .map_err(|e| InferenceError(e.0))
            })
            .transpose()?;
        Ok(Self {
            input,
            x,
            w1,
            w2,
            ids,
            weights,
            weights_f32,
            parameters,
            bias1,
            bias2,
            shared,
            activation_parameters,
            hidden,
            input_quantizer,
            hidden_quantizer,
            output,
            options,
        })
    }
    pub unsafe fn run(&self) -> Result<&RudaTensor<R>, InferenceError> {
        unsafe {
            self.input.run();
            super::norm::cast_into(&self.input.output, &self.x)?;
            self.w1.run()?;
            self.w2.run()?;
            self.ids.run();
            self.weights.run();
            super::norm::cast_into(&self.weights.output, &self.weights_f32)?;
            for (p, out) in &self.parameters {
                p.run();
                super::norm::cast_into(&p.output, out)?;
            }
            if let Some(q) = &self.input_quantizer {
                q.quantize_into(&self.x, None)
                    .map_err(|e| InferenceError(e.0))?;
                q.dequantize_into(&self.x)
                    .map_err(|e| InferenceError(e.0))?;
            }
        }
        let x = &self.x;
        let a = &self.w1.output;
        let b = &self.w2.output;
        let o = self.options;
        let (t, k, e, i, n, top) = (
            x.meta.shape()[0],
            x.meta.shape()[1],
            a.meta.shape()[0],
            b.meta.shape()[2],
            b.meta.shape()[1],
            self.ids.output.meta.shape()[1],
        );
        let count = t * top * i;
        if count > 0 {
            let dim = RudaDim::new(x.client.properties(), count);
            let grid = kernel_dsl::calculate_ruda_count_elemwise(&x.client, count, dim);
            expert_first::launch::<R>(
                &x.client,
                grid,
                dim,
                x.clone().into_array_arg(),
                a.clone().into_array_arg(),
                self.ids.output.clone().into_array_arg(),
                self.weights_f32.clone().into_array_arg(),
                self.bias1.as_ref().unwrap_or(x).clone().into_array_arg(),
                self.activation_parameters.clone().into_array_arg(),
                self.hidden.clone().into_array_arg(),
                t as u32,
                k as u32,
                i as u32,
                e as u32,
                top as u32,
                o.activation_limit.unwrap_or(f32::INFINITY),
                o.gated,
                o.activation.code(),
                match o.mode {
                    ExpertActivation::Standard => 0,
                    ExpertActivation::SwigluBias => 1,
                    ExpertActivation::SwigluStep => 2,
                    ExpertActivation::Situ => 3,
                    ExpertActivation::ClampedReluSquared => 4,
                    ExpertActivation::PowLu => 5,
                },
                self.bias1.is_some(),
                o.weight_on_input,
            );
        }
        if let Some(q) = &self.hidden_quantizer {
            unsafe {
                q.quantize_into(&self.hidden, None)
                    .map_err(|e| InferenceError(e.0))?;
                q.dequantize_into(&self.hidden)
                    .map_err(|e| InferenceError(e.0))?;
            }
        }
        let count = t * n;
        if count > 0 {
            let dim = RudaDim::new(x.client.properties(), count);
            let grid = kernel_dsl::calculate_ruda_count_elemwise(&x.client, count, dim);
            expert_second::launch::<R>(
                &x.client,
                grid,
                dim,
                self.hidden.clone().into_array_arg(),
                b.clone().into_array_arg(),
                self.ids.output.clone().into_array_arg(),
                self.weights_f32.clone().into_array_arg(),
                self.bias2.as_ref().unwrap_or(x).clone().into_array_arg(),
                self.shared.as_ref().unwrap_or(x).clone().into_array_arg(),
                self.output.clone().into_array_arg(),
                t as u32,
                i as u32,
                n as u32,
                e as u32,
                top as u32,
                self.bias2.is_some(),
                self.shared.is_some(),
                !o.weight_on_input,
                o.output_dtype.into(),
            );
        }
        Ok(&self.output)
    }
}
/// Packed precomputed routing: signed expert ID in the high 16 bits and
/// BF16/F16 route weight in the low 16 bits. No CPU unpacking is involved.
pub fn unpack_routes<R: Runtime>(
    packed: &RudaTensor<R>,
    bf16: bool,
) -> Result<Routes<R>, InferenceError> {
    check(packed, packed)?;
    if !matches!(packed.dtype, DType::I32 | DType::U32) || packed.meta.num_dims() != 2 {
        return Err(InferenceError("packed routes need I32/U32 [tokens,topK]"));
    }
    let output = Routes {
        experts: alloc(packed, packed.meta.shape().clone(), DType::I32),
        weights: alloc(packed, packed.meta.shape().clone(), DType::F32),
    };
    unsafe {
        unpack_routes_into(packed, bf16, &output)?;
    }
    Ok(output)
}
pub unsafe fn unpack_routes_into<R: Runtime>(
    packed: &RudaTensor<R>,
    bf16: bool,
    out: &Routes<R>,
) -> Result<(), InferenceError> {
    for t in [packed, &out.experts, &out.weights] {
        check(t, packed)?;
    }
    if !matches!(packed.dtype, DType::I32 | DType::U32)
        || packed.meta.num_dims() != 2
        || out.experts.dtype != DType::I32
        || out.weights.dtype != DType::F32
        || out.experts.meta.shape() != packed.meta.shape()
        || out.weights.meta.shape() != packed.meta.shape()
    {
        return Err(InferenceError("preallocated packed routes mismatch"));
    }
    let n = packed.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(packed.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&packed.client, n, dim);
        unpack_route_words::launch::<R>(
            &packed.client,
            count,
            dim,
            packed.clone().into_array_arg(),
            out.experts.clone().into_array_arg(),
            out.weights.clone().into_array_arg(),
            n as u32,
            bf16,
            packed.dtype.into(),
        );
    }
    Ok(())
}
#[ruda(launch)]
pub fn unpack_route_words<I: Int>(
    packed: &Array<I>,
    ids: &mut Array<i32>,
    weights: &mut Array<f32>,
    n: u32,
    #[comptime] bf16: bool,
    #[define(I)] _i: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let bits = u32::cast_from(packed[i]);
        ids[i] = (bits as i32) >> 16;
        let raw = bits & 65535u32;
        let mut value: f32 = 0.0f32;
        if comptime!(bf16) {
            value = f32::reinterpret(raw << 16);
        } else {
            let sign = (raw & 32768u32) << 16;
            let exp = (raw >> 10) & 31u32;
            let mantissa = raw & 1023u32;
            if exp == 0 {
                value = mantissa as f32 * 0.000000059604644775390625;
                if sign != 0 {
                    value = -value;
                }
            } else if exp == 31 {
                value = f32::reinterpret(sign | 0x7f800000u32 | (mantissa << 13));
            } else {
                value = f32::reinterpret(sign | ((exp + 112u32) << 23) | (mantissa << 13));
            }
        }
        weights[i] = value;
    }
}
