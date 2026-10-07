use ruda_model::tensor::{Bool,DType,Tensor,backend::Backend};
use super::{CategoricalLossTerms,categorical::{check_float,product}};

fn check_weight<B: Backend,const D: usize>(weight: &Tensor<B,D>,input: &Tensor<B,D>) {
    check_float(weight.dtype());
    assert_eq!(weight.device(),input.device(),"binary loss weights must share the input device");
    assert!(weight.dims().iter().zip(input.dims()).all(|(&size,expected)|size == 1 || size == expected),
        "binary loss weights do not broadcast to actual input geometry");
}

fn prepare<B: Backend,const D: usize>(input: &Tensor<B,D>,targets: &Tensor<B,D>,visible: Option<Tensor<B,D,Bool>>,
    weight: Option<&Tensor<B,D>>,positive_weight: Option<&Tensor<B,D>>) -> (DType,Tensor<B,D,Bool>) {
    check_float(input.dtype()); check_float(targets.dtype());
    assert_eq!(input.dims(),targets.dims(),"binary loss input and actual soft-label geometry differs");
    assert_eq!(input.device(),targets.device(),"binary loss inputs and targets must share a device");
    let mut double = input.dtype() == DType::F64 || targets.dtype() == DType::F64;
    for weight in [weight,positive_weight].into_iter().flatten() {
        check_weight(weight,input);
        double |= weight.dtype() == DType::F64;
    }
    let excluded = if let Some(visible) = visible {
        assert_eq!(visible.dims(),input.dims(),"binary loss selection mask differs from actual element geometry");
        assert_eq!(visible.device(),input.device(),"binary loss selection must share the input device");
        visible.bool_not()
    } else {Tensor::<B,D,Bool>::zeros(input.dims(),&input.device())};
    (if double {DType::F64} else {DType::F32},excluded)
}

fn log_probability<B: Backend,const D: usize>(input: Tensor<B,D>) -> Tensor<B,D> {
    let negative = input.clone().lower_elem(0);
    let positive_input = input.clone().mask_fill(negative.clone(),0);
    let negative_input = input.mask_fill(negative.clone().bool_not(),0);
    let positive_branch = positive_input.neg().exp().log1p().neg();
    let negative_branch = negative_input.clone()-negative_input.exp().log1p();
    positive_branch.mask_where(negative,negative_branch)
}

/// Native unreduced BCE-with-logits for actual soft/hard Bernoulli targets in [0,1].
/// Weights broadcast only along explicit singleton axes; positive_weight scales only
/// the positive target term. Selected-element mean divides by element count, not weight sum.
/// No sigmoid materialization, epsilon clipping or implicit target detachment occurs.
pub fn binary_cross_entropy_with_logits_terms<B: Backend,const D: usize>(input: Tensor<B,D>,targets: Tensor<B,D>,
    visible: Option<Tensor<B,D,Bool>>,weight: Option<Tensor<B,D>>,positive_weight: Option<Tensor<B,D>>) -> CategoricalLossTerms<B,D> {
    let (dtype,excluded) = prepare(&input,&targets,visible,weight.as_ref(),positive_weight.as_ref());
    let shape = input.dims();
    let input = input.cast(dtype).mask_fill(excluded.clone(),0);
    let targets = targets.cast(dtype).mask_fill(excluded.clone(),0);
    let negative_target = targets.clone().neg().add_scalar(1);
    let positive_target = if let Some(weight) = positive_weight {
        targets*weight.cast(dtype).expand(shape).mask_fill(excluded.clone(),0)
    } else {targets};
    let log_positive = log_probability(input.clone());
    let log_negative = log_probability(input.neg());
    let mut values = (product(log_positive,positive_target)+product(log_negative,negative_target)).neg();
    if let Some(weight) = weight {
        let weight = weight.cast(dtype).expand(shape).mask_fill(excluded.clone(),0);
        values = product(values,weight);
    }
    let valid = excluded.clone().bool_not();
    CategoricalLossTerms {values:values.mask_fill(excluded,0),normalizers:valid.clone().float().cast(dtype),valid}
}

/// Native unreduced binary CE of actual probabilities and soft Bernoulli targets.
/// Logs are unclipped: a positive target at zero probability has infinite loss.
/// Zero target coefficients contribute zero even at probability boundaries.
/// Selection masks are explicit and remove operands before logarithms.
pub fn binary_cross_entropy_probability_terms<B: Backend,const D: usize>(input: Tensor<B,D>,targets: Tensor<B,D>,
    visible: Option<Tensor<B,D,Bool>>,weight: Option<Tensor<B,D>>) -> CategoricalLossTerms<B,D> {
    let (dtype,excluded) = prepare(&input,&targets,visible,weight.as_ref(),None);
    let shape = input.dims();
    let input = input.cast(dtype).mask_fill(excluded.clone(),0.5);
    let targets = targets.cast(dtype).mask_fill(excluded.clone(),0);
    let negative_target = targets.clone().neg().add_scalar(1);
    let log_positive = input.clone().log();
    let log_negative = input.neg().log1p();
    let mut values = (product(log_positive,targets)+product(log_negative,negative_target)).neg();
    if let Some(weight) = weight {
        let weight = weight.cast(dtype).expand(shape).mask_fill(excluded.clone(),0);
        values = product(values,weight);
    }
    let valid = excluded.clone().bool_not();
    CategoricalLossTerms {values:values.mask_fill(excluded,0),normalizers:valid.clone().float().cast(dtype),valid}
}
