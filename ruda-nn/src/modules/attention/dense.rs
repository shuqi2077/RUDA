//! Explicit visibility and grouped-query attention on native backend tensors.
use super::{PackedAttentionOptions, PackedCausalAlignment};
use crate::{Dropout, DropoutConfig, Linear, LinearConfig};
use ruda_model::{
    config::Config, module::Module,
    tensor::{Bool, DType, FloatDType, Int, Tensor, backend::Backend},
};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Scale, causal alignment and inclusive window rules shared with packed attention.
pub type DenseAttentionOptions = PackedAttentionOptions;

/// Caller-selected visibility and additive score bias.
#[derive(Clone, Debug)]
pub struct DenseAttentionMask<B: Backend> {
    /// [batch, queries], True exactly at real query tokens.
    pub query_valid: Option<Tensor<B, 2, Bool>>,
    /// [batch, keys], True exactly at real key/value tokens.
    pub key_valid: Option<Tensor<B, 2, Bool>>,
    /// [batch, heads, queries, keys], with explicit singleton-axis broadcasting.
    /// True allows an edge, unlike an exclusion mask.
    pub allowed: Option<Tensor<B, 4, Bool>>,
    /// Same broadcast geometry as allowed; differentiable additive score values.
    /// Storage matches the input dtype, or FP32 for low-precision inputs.
    pub bias: Option<Tensor<B, 4>>,
}

impl<B: Backend> Default for DenseAttentionMask<B> {
    fn default() -> Self {
        Self { query_valid: None, key_valid: None, allowed: None, bias: None }
    }
}

fn check_broadcast(shape: [usize; 4], target: [usize; 4]) {
    assert!(shape.iter().zip(target).all(|(&size, expected)| size == 1 || size == expected),
        "attention mask/bias does not broadcast to the actual score geometry");
}

fn connected_zero<B: Backend>(tensor: Tensor<B, 4>, dtype: DType) -> Tensor<B, 1> {
    let excluded = Tensor::<B, 4, Bool>::zeros(tensor.dims(), &tensor.device()).bool_not();
    tensor.cast(dtype).mask_fill(excluded, 0).sum()
}

/// Native dense MHA/GQA/MQA with explicit masks, bias and backend autodiff.
///
/// Q/K/V are [batch, heads, tokens, features]. Query heads must be a multiple
/// of KV heads, and value width may differ from QK width. FP16/BF16 scores and
/// softmax use FP32; explicit FP64 inputs retain FP64. Output retains Q storage.
/// Fully excluded/all-negative-infinity score rows return connected zeros.
/// This uses ordinary dense backend tensor operations, not a fused Flash kernel.
pub fn dense_scaled_dot_product_attention<B: Backend>(
    query: Tensor<B, 4>, key: Tensor<B, 4>, value: Tensor<B, 4>,
    masks: DenseAttentionMask<B>, options: DenseAttentionOptions, dropout: Option<&Dropout>,
) -> Tensor<B, 4> {
    let [batch, heads, queries, features] = query.dims();
    let [key_batch, kv_heads, keys, key_features] = key.dims();
    let [value_batch, value_heads, values, value_features] = value.dims();
    assert!(heads > 0 && kv_heads > 0 && heads.is_multiple_of(kv_heads) && features > 0 && value_features > 0,
        "invalid grouped attention head/feature geometry");
    assert_eq!((batch, keys, kv_heads, features), (key_batch, values, value_heads, key_features),
        "key/value payloads or QK feature geometry differ");
    assert_eq!(batch, value_batch, "query/value batches differ");
    let device = query.device();
    let storage = query.dtype();
    assert!(matches!(storage, DType::F32 | DType::F16 | DType::BF16 | DType::F64)
        && key.dtype() == storage && value.dtype() == storage, "attention requires matching floating input storage");
    assert!(device == key.device() && device == value.device(), "attention inputs must share a device");
    let compute = if storage == DType::F64 { DType::F64 } else { DType::F32 };
    let score_shape = [batch, heads, queries, keys];
    if let Some(mask) = &masks.query_valid {
        assert_eq!(mask.dims(), [batch, queries], "query visibility differs from actual query tokens");
        assert_eq!(mask.device(), device, "query visibility must share the device");
    }
    if let Some(mask) = &masks.key_valid {
        assert_eq!(mask.dims(), [batch, keys], "key visibility differs from actual key tokens");
        assert_eq!(mask.device(), device, "key visibility must share the device");
    }
    if let Some(mask) = &masks.allowed {
        check_broadcast(mask.dims(), score_shape);
        assert_eq!(mask.device(), device, "allowed edges must share the device");
    }
    if let Some(bias) = &masks.bias {
        check_broadcast(bias.dims(), score_shape);
        assert_eq!(bias.device(), device, "attention bias must share the device");
        assert!(bias.dtype() == storage || bias.dtype() == compute, "attention bias storage cannot be silently narrowed");
    }
    let scale = options.scale.unwrap_or(1.0 / (features as f64).sqrt());
    assert!(scale.is_finite() && (compute == DType::F64 || (scale as f32).is_finite()),
        "attention scale must be finite in the selected compute precision");
    if let Some((left, right)) = options.window { assert!(left >= -1 && right >= -1, "window distances must be nonnegative or -1"); }
    if let Some(dropout) = dropout { assert!(dropout.prob.is_finite() && (0.0..=1.0).contains(&dropout.prob), "invalid attention dropout probability"); }
    if batch == 0 || queries == 0 || keys == 0
        || dropout.is_some_and(|dropout|dropout.prob == 1.0 && B::ad_enabled(&device)) {
        let mut zero = connected_zero(query, compute) + connected_zero(key, compute) + connected_zero(value, compute);
        if let Some(bias) = masks.bias { zero = zero + connected_zero(bias, compute); }
        return (Tensor::<B, 4>::zeros([batch, heads, queries, value_features], (&device, compute))
            + zero.reshape([1, 1, 1, 1])).cast(storage);
    }
    let groups = heads / kv_heads;
    let mut query = query.cast(compute);
    let mut key = key.cast(compute);
    let mut value = value.cast(compute);
    if let Some(mask) = &masks.query_valid {
        query = query.mask_fill(mask.clone().bool_not().reshape([batch,1,queries,1])
            .expand([batch,heads,queries,features]),0);
    }
    if let Some(mask) = &masks.key_valid {
        let excluded = mask.clone().bool_not().reshape([batch,1,keys,1]);
        key = key.mask_fill(excluded.clone().expand([batch,kv_heads,keys,features]),0);
        value = value.mask_fill(excluded.expand([batch,kv_heads,keys,value_features]),0);
    }
    let key = key.reshape([batch, kv_heads, 1, keys, features])
        .repeat_dim(2, groups).reshape([batch, heads, keys, features]);
    let value = value.reshape([batch, kv_heads, 1, keys, value_features])
        .repeat_dim(2, groups).reshape([batch, heads, keys, value_features]);
    let mut scores = query.matmul(key.swap_dims(2, 3)).mul_scalar(scale);
    if let Some(bias) = masks.bias { scores = scores + bias.cast(compute).expand(score_shape); }
    let mut excluded = Tensor::<B, 4, Bool>::zeros(score_shape, &device);
    if let Some(mask) = masks.allowed { excluded = excluded.bool_or(mask.expand(score_shape).bool_not()); }
    if let Some(mask) = masks.query_valid {
        excluded = excluded.bool_or(mask.reshape([batch, 1, queries, 1]).expand(score_shape).bool_not());
    }
    if let Some(mask) = masks.key_valid {
        excluded = excluded.bool_or(mask.reshape([batch, 1, 1, keys]).expand(score_shape).bool_not());
    }
    if options.causal || options.window.is_some() {
        let query_length = i64::try_from(queries).expect("query positions exceed integer range");
        let key_length = i64::try_from(keys).expect("key positions exceed integer range");
        let offset = match options.alignment {
            PackedCausalAlignment::UpperLeft => 0,
            PackedCausalAlignment::LowerRight => key_length - query_length,
        };
        let rows = Tensor::<B, 1, Int>::arange(0..query_length, (&device, DType::I64))
            .add_scalar(offset).reshape([1, 1, queries, 1]).expand(score_shape);
        let columns = Tensor::<B, 1, Int>::arange(0..key_length, (&device, DType::I64))
            .reshape([1, 1, 1, keys]).expand(score_shape);
        if options.causal { excluded = excluded.bool_or(columns.clone().greater(rows.clone())); }
        if let Some((left, right)) = options.window {
            if left >= 0 && (left as usize) < queries.max(keys) {
                excluded = excluded.bool_or(columns.clone().lower(rows.clone().sub_scalar(left as i64)));
            }
            if right >= 0 && (right as usize) < queries.max(keys) {
                excluded = excluded.bool_or(columns.greater(rows.add_scalar(right as i64)));
            }
        }
    }
    scores = scores.mask_fill(excluded, f64::NEG_INFINITY);
    let fully_excluded = scores.clone().equal_elem(f64::NEG_INFINITY).all_dim(3);
    let maximum = scores.clone().max_dim(3).mask_fill(fully_excluded.clone(), 0);
    let exponentials = (scores - maximum).exp();
    let denominator = exponentials.clone().sum_dim(3).clamp_min(1);
    let mut weights = exponentials / denominator;
    if let Some(dropout) = dropout {
        weights = dropout.forward(weights);
    }
    weights.matmul(value).mask_fill(fully_excluded.expand([batch,heads,queries,value_features]),0).cast(storage)
}

/// Grouped-query attention projection geometry, independent of model family.
#[derive(Config, Debug)]
pub struct GroupedQueryAttentionConfig {
    /// Input/output residual width.
    pub d_model: usize,
    /// Number of independently projected query heads.
    pub query_heads: usize,
    /// Number of shared key/value heads; one selects MQA.
    pub kv_heads: usize,
    /// Q/K/V feature width per head.
    pub head_dimension: usize,
    /// Bias in each actual Q/K/V/output projection.
    #[config(default = false)]
    pub bias: bool,
    /// Native Dropout probability applied to normalized attention weights.
    #[config(default = 0.0)]
    pub dropout: f64,
}

/// Actual trainable GQA/MQA projections and explicit-mask dense attention.
#[derive(Module, Debug)]
pub struct GroupedQueryAttention<B: Backend> {
    /// Query projection from residual width to query_heads * head_dimension.
    pub query: Linear<B>,
    /// Shared key projection from residual width to kv_heads * head_dimension.
    pub key: Linear<B>,
    /// Shared value projection using the same actual KV head geometry.
    pub value: Linear<B>,
    /// Context projection back to residual width.
    pub output: Linear<B>,
    /// Dropout on normalized weights; native backend training mode applies.
    pub dropout: Dropout,
    /// Actual query head count.
    pub query_heads: usize,
    /// Actual shared KV head count.
    pub kv_heads: usize,
    /// Actual per-head projection width.
    pub head_dimension: usize,
}

impl GroupedQueryAttentionConfig {
    /// Initialize only the declared projections on the explicit backend device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GroupedQueryAttention<B> {
        assert!(self.d_model > 0 && self.query_heads > 0 && self.kv_heads > 0
            && self.head_dimension > 0 && self.query_heads.is_multiple_of(self.kv_heads), "invalid grouped projection geometry");
        let queries = self.query_heads.checked_mul(self.head_dimension).expect("query projection width overflow");
        let keys = self.kv_heads.checked_mul(self.head_dimension).expect("KV projection width overflow");
        GroupedQueryAttention {
            query: LinearConfig::new(self.d_model, queries).with_bias(self.bias).init(device),
            key: LinearConfig::new(self.d_model, keys).with_bias(self.bias).init(device),
            value: LinearConfig::new(self.d_model, keys).with_bias(self.bias).init(device),
            output: LinearConfig::new(queries, self.d_model).with_bias(self.bias).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            query_heads: self.query_heads, kv_heads: self.kv_heads, head_dimension: self.head_dimension,
        }
    }
}

impl<B: Backend> GroupedQueryAttention<B> {
    /// Connect actual loaded projections, including cross-attention with a
    /// different query/memory input width. No parameters are rebuilt or copied.
    pub fn from_projections(query: Linear<B>,key: Linear<B>,value: Linear<B>,output: Linear<B>,
        query_heads: usize,kv_heads: usize,head_dimension: usize,dropout: Dropout) -> Self {
        assert!(query_heads > 0 && kv_heads > 0 && head_dimension > 0
            && query_heads.is_multiple_of(kv_heads),"invalid grouped projection head geometry");
        let query_width = query_heads.checked_mul(head_dimension).expect("query width overflow");
        let key_width = kv_heads.checked_mul(head_dimension).expect("KV width overflow");
        let query_shape = query.weight.val().dims();
        let key_shape = key.weight.val().dims();
        let value_shape = value.weight.val().dims();
        assert_eq!(query_shape[1],query_width,"actual query weight differs from head geometry");
        assert_eq!(key_shape[1],key_width,"actual key weight differs from head geometry");
        assert_eq!(value_shape,key_shape,"actual key/value input and head widths differ");
        assert_eq!(output.weight.val().dims(),[query_width,query_shape[0]],"actual attention output/residual widths differ");
        assert!(dropout.prob.is_finite() && (0.0..=1.0).contains(&dropout.prob),"invalid attention dropout");
        Self {query,key,value,output,dropout,query_heads,kv_heads,head_dimension}
    }

    /// Apply the actual Q/K/V parameters, exposing [batch, heads, tokens, features].
    /// The caller may transform Q/K positions before forward_projected.
    pub fn project(&self, query: Tensor<B, 3>, key: Tensor<B, 3>, value: Tensor<B, 3>)
        -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, queries, _] = query.dims();
        let [key_batch, keys, _] = key.dims();
        let [value_batch, values, _] = value.dims();
        assert_eq!((batch, keys), (key_batch, values), "grouped projection batches/key lengths differ");
        assert_eq!(batch, value_batch, "grouped value batch differs");
        let query = self.query.forward(query).reshape([batch, queries, self.query_heads, self.head_dimension]).swap_dims(1, 2);
        let key = self.key.forward(key).reshape([batch, keys, self.kv_heads, self.head_dimension]).swap_dims(1, 2);
        let value = self.value.forward(value).reshape([batch, keys, self.kv_heads, self.head_dimension]).swap_dims(1, 2);
        (query, key, value)
    }

    /// Explicit projection arithmetic dtype, retaining derivatives to each
    /// parameter's original storage. Outputs use the selected arithmetic dtype.
    /// This casts parameter VALUES, not modules or newly detached parameter leaves.
    pub fn project_with_compute_dtype(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
        dtype: FloatDType) -> (Tensor<B,4>,Tensor<B,4>,Tensor<B,4>) {
        let [batch,queries,_] = query.dims();
        let [key_batch,keys,_] = key.dims();
        let [value_batch,values,_] = value.dims();
        assert_eq!((batch,keys),(key_batch,values),"grouped projection batches/key lengths differ");
        assert_eq!(batch,value_batch,"grouped value batch differs");
        let project = |layer: &Linear<B>,input: Tensor<B,3>|ruda_model::tensor::module::linear(
            input.cast(dtype),layer.weight.val().cast(dtype),layer.bias.as_ref().map(|bias|bias.val().cast(dtype)));
        let query = project(&self.query,query).reshape([batch,queries,self.query_heads,self.head_dimension]).swap_dims(1,2);
        let key = project(&self.key,key).reshape([batch,keys,self.kv_heads,self.head_dimension]).swap_dims(1,2);
        let value = project(&self.value,value).reshape([batch,keys,self.kv_heads,self.head_dimension]).swap_dims(1,2);
        (query,key,value)
    }

    /// Attend actual projected/transformed heads and apply the existing output weight.
    /// RoPE, cache concatenation or other positional transforms remain explicit.
    pub fn forward_projected(&self, query: Tensor<B, 4>, key: Tensor<B, 4>, value: Tensor<B, 4>,
        masks: DenseAttentionMask<B>, options: DenseAttentionOptions) -> Tensor<B, 3> {
        let [batch, heads, queries, width] = query.dims();
        assert_eq!((heads, width), (self.query_heads, self.head_dimension), "projected query heads differ from the output weight");
        assert_eq!((key.dims()[1], key.dims()[3]), (self.kv_heads, self.head_dimension), "projected key geometry differs");
        assert_eq!((value.dims()[1], value.dims()[3]), (self.kv_heads, self.head_dimension), "projected value geometry differs");
        let context = dense_scaled_dot_product_attention(query, key, value, masks, options, Some(&self.dropout));
        self.output.forward(context.swap_dims(1, 2).reshape([batch, queries, self.query_heads * self.head_dimension]))
    }

    /// Project, attend and restore residual width without adding a residual connection.
    pub fn forward(&self, query: Tensor<B, 3>, key: Tensor<B, 3>, value: Tensor<B, 3>,
        masks: DenseAttentionMask<B>, options: DenseAttentionOptions) -> Tensor<B, 3> {
        let (query, key, value) = self.project(query, key, value);
        self.forward_projected(query, key, value, masks, options)
    }

    /// Whole attention projection/output arithmetic precision chosen explicitly.
    /// Input/bias storage may differ; only the final output returns to query storage.
    pub fn forward_with_compute_dtype(&self,query: Tensor<B,3>,key: Tensor<B,3>,value: Tensor<B,3>,
        mut masks: DenseAttentionMask<B>,options: DenseAttentionOptions,dtype: FloatDType) -> Tensor<B,3> {
        let storage = query.dtype();
        let (query,key,value) = self.project_with_compute_dtype(query,key,value,dtype);
        if let Some(bias) = masks.bias.take() { masks.bias = Some(bias.cast(dtype)); }
        let [batch,heads,queries,width] = query.dims();
        let context = dense_scaled_dot_product_attention(query,key,value,masks,options,Some(&self.dropout))
            .swap_dims(1,2).reshape([batch,queries,heads*width]);
        ruda_model::tensor::module::linear(context,self.output.weight.val().cast(dtype),
            self.output.bias.as_ref().map(|bias|bias.val().cast(dtype))).cast(storage)
    }
}
