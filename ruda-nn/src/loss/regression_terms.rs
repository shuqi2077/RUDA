use ruda_model::tensor::{Bool,DType,Tensor,backend::Backend};
use super::{CategoricalLossTerms,HuberLoss,LpLoss,MseLoss,PoissonNllLoss,SmoothL1Loss,
    categorical::{check_float,product}};

/// Native unreduced loss, actual normalization weights and explicit selection.
/// The existing categorical type remains the same type for API compatibility.
pub type LossTerms<B,const D: usize = 1> = CategoricalLossTerms<B,D>;

fn prepare<B: Backend,const D: usize>(predictions: Tensor<B,D>,targets: Tensor<B,D>,
    visible: Option<Tensor<B,D,Bool>>) -> (Tensor<B,D>,Tensor<B,D>,Tensor<B,D,Bool>) {
    assert_eq!(predictions.dims(),targets.dims(),"regression prediction/target geometry differs");
    assert_eq!(predictions.device(),targets.device(),"regression operands must share the device");
    check_float(predictions.dtype()); check_float(targets.dtype());
    let dtype = if predictions.dtype() == DType::F64 || targets.dtype() == DType::F64 {DType::F64} else {DType::F32};
    let valid = if let Some(visible) = visible {
        assert_eq!(visible.dims(),predictions.dims(),"regression selection differs from actual element geometry");
        assert_eq!(visible.device(),predictions.device(),"regression selection must share the operand device");
        visible
    } else {Tensor::<B,D,Bool>::zeros(predictions.dims(),&predictions.device()).bool_not()};
    let excluded = valid.clone().bool_not();
    (predictions.cast(dtype).mask_fill(excluded.clone(),0),targets.cast(dtype).mask_fill(excluded,0),valid)
}

fn finish<B: Backend,const D: usize>(values: Tensor<B,D>,valid: Tensor<B,D,Bool>) -> LossTerms<B,D> {
    let dtype = values.dtype();
    LossTerms {values:values.mask_fill(valid.clone().bool_not(),0),normalizers:valid.clone().float().cast(dtype),valid}
}

impl MseLoss {
    /// Native per-element squared error with explicit supervision selection.
    /// Excluded operands are removed before subtraction; mean uses selected elements.
    /// FP16/BF16 compute in FP32; either FP64 operand retains FP64 computation.
    pub fn forward_terms<B: Backend,const D: usize>(&self,predictions: Tensor<B,D>,targets: Tensor<B,D>,
        visible: Option<Tensor<B,D,Bool>>) -> LossTerms<B,D> {
        let (predictions,targets,valid) = prepare(predictions,targets,visible);
        finish(self.forward_no_reduction(predictions,targets),valid)
    }
}

impl LpLoss {
    /// Native selected |prediction-target|^p, not the rooted Lp norm.
    /// Uses the module's actual exponent; active zero-residual singularities for
    /// p<1 are not clipped. Excluded entries never evaluate that singular power.
    pub fn forward_terms<B: Backend,const D: usize>(&self,predictions: Tensor<B,D>,targets: Tensor<B,D>,
        visible: Option<Tensor<B,D,Bool>>) -> LossTerms<B,D> {
        assert!(self.p.is_finite() && self.p > 0.,"regression Lp exponent must be finite and positive");
        let (predictions,targets,valid) = prepare(predictions,targets,visible);
        let residual = (predictions-targets).mask_fill(valid.clone().bool_not(),1);
        let values = if self.p == 1. {residual.abs()} else if self.p == 2. {residual.square()}
            else {residual.abs().powf_scalar(self.p)};
        finish(values,valid)
    }
}

impl HuberLoss {
    /// Native selected Huber error, honoring this module's delta and linear bias.
    /// The inactive quadratic branch does not square large residuals. Sample
    /// weights can subsequently be applied with LossTerms::reweighted.
    pub fn forward_terms<B: Backend,const D: usize>(&self,predictions: Tensor<B,D>,targets: Tensor<B,D>,
        visible: Option<Tensor<B,D,Bool>>) -> LossTerms<B,D> {
        assert!(self.delta.is_finite() && self.delta >= 0. && self.lin_bias.is_finite(),
            "selected Huber loss requires finite delta and linear bias");
        let (predictions,targets,valid) = prepare(predictions,targets,visible);
        let residual = predictions-targets;
        let absolute = residual.clone().abs();
        let large = absolute.clone().greater_elem(f64::from(self.delta));
        let inside = residual.mask_fill(large.clone(),0).square().mul_scalar(0.5);
        let coefficient = absolute.full_like(f64::from(self.delta));
        let outside = product(absolute.mask_fill(large.clone().bool_not(),0),coefficient)
            .sub_scalar(f64::from(self.lin_bias));
        finish(inside.mask_where(large,outside),valid)
    }
}

impl SmoothL1Loss {
    /// Native selected smooth-L1 error using the actual configured beta.
    /// Mean counts actual selected elements, including selected zero residuals.
    pub fn forward_terms<B: Backend,const D: usize>(&self,predictions: Tensor<B,D>,targets: Tensor<B,D>,
        visible: Option<Tensor<B,D,Bool>>) -> LossTerms<B,D> {
        assert!(self.beta.is_finite() && self.beta > 0.,"selected smooth-L1 beta must be finite and positive");
        let (predictions,targets,valid) = prepare(predictions,targets,visible);
        let residual = predictions-targets;
        let absolute = residual.clone().abs();
        let quadratic = absolute.clone().lower_elem(f64::from(self.beta));
        let inside = residual.mask_fill(quadratic.clone().bool_not(),0).square()
            .mul_scalar(0.5).div_scalar(f64::from(self.beta));
        let outside = absolute.mask_fill(quadratic.clone(),0).sub_scalar(0.5*f64::from(self.beta));
        finish(outside.mask_where(quadratic,inside),valid)
    }
}

impl PoissonNllLoss {
    /// Native selected Poisson NLL with this module's log_input/full/eps options.
    /// Targets must be nonnegative; non-log inputs must be nonnegative. No host
    /// scan, rounding, target detachment or inferred missing-value mask occurs.
    /// The full term is the configured Stirling approximation for targets>1,
    /// not log-gamma. Excluded entries are removed before logarithms/exp.
    pub fn forward_terms<B: Backend,const D: usize>(&self,predictions: Tensor<B,D>,targets: Tensor<B,D>,
        visible: Option<Tensor<B,D,Bool>>) -> LossTerms<B,D> {
        assert!(self.eps.is_finite() && self.eps > 0.,"selected Poisson epsilon must be finite and positive");
        let (predictions,targets,valid) = prepare(predictions,targets,visible);
        let mut values = if self.log_input {
            predictions.clone().exp()-product(predictions,targets.clone())
        } else {
            let predictions = predictions.mask_fill(valid.clone().bool_not(),1);
            predictions.clone()-product(predictions.add_scalar(self.eps).log(),targets.clone())
        };
        if self.full {
            let approximate = targets.clone().greater_elem(1);
            let mass = targets.mask_fill(approximate.clone().bool_not(),1);
            let stirling = mass.clone()*mass.clone().log()-mass.clone()
                +(mass.log().add_scalar(num_traits::Float::ln(core::f64::consts::TAU))).mul_scalar(0.5);
            values = values+stirling.mask_fill(approximate.bool_not(),0);
        }
        finish(values,valid)
    }
}
