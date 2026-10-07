use ruda_model::tensor::{Bool,DType,Int,IntDType,Tensor,activation::log_softmax,backend::Backend};
use super::CrossEntropyLoss;

/// Unreduced native classification loss and its actual selected normalization weights.
/// Hard targets normalize by selected class weight; soft targets normalize by selected rows.
#[derive(Clone,Debug)]
pub struct CategoricalLossTerms<B: Backend,const D: usize = 1> {
    /// Actual per-row/token/pixel loss in FP32, or FP64 when an operand uses FP64.
    pub values: Tensor<B,D>,
    /// Actual selected effective weights; excluded targets have exactly zero weight.
    pub normalizers: Tensor<B,D>,
    /// True exactly at caller-selected, nonignored targets, independently of class weight.
    pub valid: Tensor<B,D,Bool>,
}

impl<B: Backend,const D: usize> CategoricalLossTerms<B,D> {
    /// Actual unnormalized sum; no microbatch or distributed normalization occurs here.
    pub fn loss_sum(&self) -> Tensor<B,1> { self.values.clone().sum() }

    /// Actual effective denominator for weighted accumulation or collective reduction.
    pub fn effective_weight(&self) -> Tensor<B,1> { self.normalizers.clone().sum() }

    /// Exact I64 selected-target count, not the physical token/pixel storage size.
    pub fn valid_count(&self) -> Tensor<B,1,Int> { self.valid.clone().int().cast(IntDType::I64).sum() }

    /// Effective weighted mean; no selected effective weight yields a connected zero.
    /// Fractional nonzero denominators are not clamped to one.
    pub fn mean(&self) -> Tensor<B,1> {
        let denominator = self.effective_weight();
        let empty = denominator.clone().equal_elem(0);
        self.loss_sum().mask_fill(empty.clone(),0)/denominator.mask_fill(empty,1)
    }

    /// Reweight actual samples/tokens/pixels without losing their existing selection mask.
    /// Caller weights must be finite and nonnegative; FP64 weight storage retains FP64 work.
    pub fn reweighted(self,weights: Tensor<B,D>) -> Self {
        assert_eq!(weights.dims(),self.values.dims(),"classification sample weights differ from actual target geometry");
        assert_eq!(weights.device(),self.values.device(),"classification sample weights must share the device");
        check_float(weights.dtype());
        let dtype = if weights.dtype() == DType::F64 || self.values.dtype() == DType::F64 {DType::F64} else {DType::F32};
        let weights = weights.cast(dtype).mask_fill(self.valid.clone().bool_not(),0);
        Self {values:product(self.values.cast(dtype),weights.clone()),
            normalizers:self.normalizers.cast(dtype)*weights,valid:self.valid}
    }

    /// Restore actual token/pixel axes after a classification-row computation.
    pub fn reshape<const U: usize>(self,shape: [usize;U]) -> CategoricalLossTerms<B,U> {
        CategoricalLossTerms {values:self.values.reshape(shape),normalizers:self.normalizers.reshape(shape),valid:self.valid.reshape(shape)}
    }
}

pub(super) fn check_float(dtype: DType) {
    assert!(matches!(dtype,DType::F16|DType::BF16|DType::F32|DType::Flex32|DType::F64),"classification operands must use floating storage");
}

fn compute<B: Backend>(criterion: &CrossEntropyLoss<B>,inputs: &Tensor<B,2>,target_dtype: Option<DType>) -> DType {
    check_float(inputs.dtype());
    let mut double = inputs.dtype() == DType::F64 || target_dtype == Some(DType::F64);
    if let Some(weights) = &criterion.weights {
        check_float(weights.dtype());
        assert_eq!(weights.dims(),[inputs.dims()[1]],"classification class weight count differs from actual classes");
        assert_eq!(weights.device(),inputs.device(),"classification weights must share the input device");
        double |= weights.dtype() == DType::F64;
    }
    let alpha = criterion.smoothing.unwrap_or(0.0);
    assert!(alpha.is_finite() && (0.0..=1.0).contains(&alpha),"classification smoothing must be in [0,1]");
    assert!(inputs.dims()[1] > 0,"classification class dimension must be nonempty");
    if double {DType::F64} else {DType::F32}
}

fn row_mask<B: Backend>(inputs: &Tensor<B,2>,visible: Option<Tensor<B,1,Bool>>) -> Tensor<B,1,Bool> {
    if let Some(visible) = visible {
        assert_eq!(visible.dims(),[inputs.dims()[0]],"classification visibility differs from actual input rows");
        assert_eq!(visible.device(),inputs.device(),"classification visibility must share the input device");
        visible.bool_not()
    } else {Tensor::<B,1,Bool>::zeros([inputs.dims()[0]],&inputs.device())}
}

fn probabilities<B: Backend>(criterion: &CrossEntropyLoss<B>,inputs: Tensor<B,2>,excluded: Tensor<B,1,Bool>,dtype: DType)
    -> Tensor<B,2> {
    let shape = inputs.dims();
    let inputs = inputs.cast(dtype).mask_fill(excluded.reshape([shape[0],1]).expand(shape),if criterion.logits {0} else {1});
    if criterion.logits {log_softmax(inputs,1)} else {inputs.log()}
}

pub(super) fn product<B: Backend,const D: usize>(value: Tensor<B,D>,coefficient: Tensor<B,D>) -> Tensor<B,D> {
    let zero_infinity = coefficient.clone().equal_elem(0).bool_and(value.clone().is_inf());
    value.mask_fill(zero_infinity,0)*coefficient
}

impl<B: Backend> CrossEntropyLoss<B> {
    /// Per-row hard-label loss with explicit ignore sentinel and target visibility.
    /// Existing pad_tokens are excluded as well. Nonignored indices must be actual class IDs.
    /// This opt-in API normalizes only selected weights, leaving legacy forward unchanged.
    pub fn forward_terms(&self,inputs: Tensor<B,2>,targets: Tensor<B,1,Int>,ignore_index: Option<i64>,
        visible: Option<Tensor<B,1,Bool>>) -> CategoricalLossTerms<B> {
        let [rows,classes] = inputs.dims();
        assert_eq!(targets.dims(),[rows],"classification input and target rows differ");
        assert_eq!(targets.device(),inputs.device(),"classification inputs and targets must share a device");
        let dtype = compute(self,&inputs,None);
        let targets = targets.cast(IntDType::I64);
        let mut excluded = row_mask(&inputs,visible);
        if let Some(ignore_index) = ignore_index { excluded = excluded.bool_or(targets.clone().equal_elem(ignore_index)); }
        if let Some(pads) = &self.pad_tokens {
            for &pad in pads {
                let pad = i64::try_from(pad).expect("classification pad target exceeds signed index range");
                excluded = excluded.bool_or(targets.clone().equal_elem(pad));
            }
        }
        let valid = excluded.clone().bool_not();
        if rows == 0 {
            return CategoricalLossTerms {values:inputs.cast(dtype).reshape([0]),normalizers:Tensor::zeros([0],(&targets.device(),dtype)),valid};
        }
        let targets = targets.mask_fill(excluded.clone(),0);
        let normalizers = if let Some(weights) = &self.weights {weights.clone().cast(dtype).gather(0,targets.clone())}
            else {Tensor::ones([rows],(&inputs.device(),dtype))};
        let log_probabilities = probabilities(self,inputs,excluded.clone(),dtype);
        let alpha = f64::from(self.smoothing.unwrap_or(0.0));
        let mut values = if alpha == 1.0 {Tensor::zeros([rows],(&targets.device(),dtype))} else {
            let selected = log_probabilities.clone().gather(1,targets.reshape([rows,1])).reshape([rows]);
            product(selected,normalizers.clone()).neg().mul_scalar(1.0-alpha)
        };
        if alpha > 0.0 {
            let smoothed = if let Some(weights) = &self.weights {
                let weights = weights.clone().cast(dtype).reshape([1,classes]).expand([rows,classes]);
                product(log_probabilities,weights).sum_dim(1).reshape([rows]).div_scalar(classes as f64)
            } else {log_probabilities.mean_dim(1).reshape([rows])};
            values = values-smoothed.mul_scalar(alpha);
        }
        CategoricalLossTerms {values:values.mask_fill(excluded.clone(),0),normalizers:normalizers.mask_fill(excluded,0),valid}
    }

    /// Actual soft-label distributions and class weights, with per-row loss and row counts.
    /// Targets are not rounded, detached or renormalized. Smoothing mixes them with the
    /// full-class uniform distribution; selected-row mean is not selected-class-weight mean.
    /// When logits=false, actual probabilities are logged without epsilon clipping.
    pub fn forward_soft_terms(&self,inputs: Tensor<B,2>,targets: Tensor<B,2>,visible: Option<Tensor<B,1,Bool>>)
        -> CategoricalLossTerms<B> {
        assert_eq!(inputs.dims(),targets.dims(),"soft classification input/target class geometry differs");
        assert_eq!(inputs.device(),targets.device(),"soft classification inputs and targets must share a device");
        check_float(targets.dtype());
        let [rows,classes] = inputs.dims();
        let dtype = compute(self,&inputs,Some(targets.dtype()));
        let excluded = row_mask(&inputs,visible);
        let valid = excluded.clone().bool_not();
        if rows == 0 {
            return CategoricalLossTerms {values:inputs.cast(dtype).reshape([0])+targets.cast(dtype).reshape([0]),
                normalizers:Tensor::zeros([0],(&valid.device(),dtype)),valid};
        }
        let excluded_classes = excluded.clone().reshape([rows,1]).expand([rows,classes]);
        let targets = targets.cast(dtype).mask_fill(excluded_classes.clone(),0);
        let alpha = f64::from(self.smoothing.unwrap_or(0.0));
        let targets = if alpha == 0.0 {targets} else {targets.mul_scalar(1.0-alpha).add_scalar(alpha/classes as f64)};
        let targets = targets.mask_fill(excluded_classes,0);
        let coefficients = if let Some(weights) = &self.weights {targets*weights.clone().cast(dtype).reshape([1,classes])} else {targets};
        let log_probabilities = probabilities(self,inputs,excluded.clone(),dtype);
        let values = product(log_probabilities,coefficients).sum_dim(1).reshape([rows]).neg().mask_fill(excluded,0);
        let normalizers = valid.clone().float().cast(dtype);
        CategoricalLossTerms {values,normalizers,valid}
    }

    /// Per-token classification of [batch,tokens,classes], with no causal target shift.
    pub fn forward_token_terms(&self,inputs: Tensor<B,3>,targets: Tensor<B,2,Int>,ignore_index: Option<i64>,
        visible: Option<Tensor<B,2,Bool>>) -> CategoricalLossTerms<B,2> {
        let [batch,tokens,classes] = inputs.dims();
        assert_eq!(targets.dims(),[batch,tokens],"token classification targets differ from actual token rows");
        let rows = batch.checked_mul(tokens).expect("token classification row count overflow");
        let visible = visible.map(|visible| {
            assert_eq!(visible.dims(),[batch,tokens],"token classification visibility differs from actual rows"); visible.reshape([rows])
        });
        self.forward_terms(inputs.reshape([rows,classes]),targets.reshape([rows]),ignore_index,visible).reshape([batch,tokens])
    }

    /// Per-token soft labels with actual full-class distributions and unchanged token axes.
    pub fn forward_soft_token_terms(&self,inputs: Tensor<B,3>,targets: Tensor<B,3>,visible: Option<Tensor<B,2,Bool>>)
        -> CategoricalLossTerms<B,2> {
        let [batch,tokens,classes] = inputs.dims();
        assert_eq!(targets.dims(),[batch,tokens,classes],"soft token label geometry differs");
        let rows = batch.checked_mul(tokens).expect("soft token classification row count overflow");
        let visible = visible.map(|visible| {
            assert_eq!(visible.dims(),[batch,tokens],"soft token visibility differs from actual rows"); visible.reshape([rows])
        });
        self.forward_soft_terms(inputs.reshape([rows,classes]),targets.reshape([rows,classes]),visible).reshape([batch,tokens])
    }

    /// Native per-pixel classification of NCHW logits against actual [N,H,W] class indices.
    pub fn forward_pixel_terms(&self,inputs: Tensor<B,4>,targets: Tensor<B,3,Int>,ignore_index: Option<i64>,
        visible: Option<Tensor<B,3,Bool>>) -> CategoricalLossTerms<B,3> {
        let [batch,classes,height,width] = inputs.dims();
        assert_eq!(targets.dims(),[batch,height,width],"pixel classification target geometry differs");
        let rows = batch.checked_mul(height).and_then(|rows|rows.checked_mul(width)).expect("pixel classification row count overflow");
        let visible = visible.map(|visible| {
            assert_eq!(visible.dims(),[batch,height,width],"pixel classification visibility differs"); visible.reshape([rows])
        });
        self.forward_terms(inputs.permute([0,2,3,1]).reshape([rows,classes]),targets.reshape([rows]),ignore_index,visible)
            .reshape([batch,height,width])
    }

    /// Native NCHW soft-label pixel loss, preserving every actual class probability.
    pub fn forward_soft_pixel_terms(&self,inputs: Tensor<B,4>,targets: Tensor<B,4>,visible: Option<Tensor<B,3,Bool>>)
        -> CategoricalLossTerms<B,3> {
        let [batch,classes,height,width] = inputs.dims();
        assert_eq!(targets.dims(),[batch,classes,height,width],"soft pixel target geometry differs");
        let rows = batch.checked_mul(height).and_then(|rows|rows.checked_mul(width)).expect("soft pixel classification row count overflow");
        let visible = visible.map(|visible| {
            assert_eq!(visible.dims(),[batch,height,width],"soft pixel visibility differs"); visible.reshape([rows])
        });
        self.forward_soft_terms(inputs.permute([0,2,3,1]).reshape([rows,classes]),targets.permute([0,2,3,1]).reshape([rows,classes]),visible)
            .reshape([batch,height,width])
    }
}
