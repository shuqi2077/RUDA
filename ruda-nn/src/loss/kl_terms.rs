use ruda_model::tensor::{Bool,DType,Tensor,backend::Backend};
use super::{KLDivLoss,CategoricalLossTerms,categorical::check_float};

impl KLDivLoss {
    /// Actual per-row KL over all classes, with explicit selection and selected-row mean.
    /// Predictions are log probabilities; target space follows this module's log_target.
    /// Zero target mass contributes exactly zero without epsilon clipping or renormalization.
    pub fn forward_terms<B: Backend>(&self,predictions: Tensor<B,2>,targets: Tensor<B,2>,
        visible: Option<Tensor<B,1,Bool>>) -> CategoricalLossTerms<B> {
        assert_eq!(predictions.dims(),targets.dims(),"KL input and actual target class geometry differs");
        assert_eq!(predictions.device(),targets.device(),"KL input and target devices differ");
        check_float(predictions.dtype()); check_float(targets.dtype());
        let [rows,classes] = predictions.dims();
        assert!(classes > 0,"KL class dimension must be nonempty");
        let dtype = if predictions.dtype() == DType::F64 || targets.dtype() == DType::F64 {DType::F64} else {DType::F32};
        let excluded = if let Some(visible) = visible {
            assert_eq!(visible.dims(),[rows],"KL row selection geometry differs");
            assert_eq!(visible.device(),predictions.device(),"KL row selection device differs");
            visible.bool_not()
        } else {Tensor::<B,1,Bool>::zeros([rows],&predictions.device())};
        let valid = excluded.clone().bool_not();
        if rows == 0 {
            return CategoricalLossTerms {values:predictions.cast(dtype).reshape([0])+targets.cast(dtype).reshape([0]),
                normalizers:Tensor::zeros([0],(&valid.device(),dtype)),valid};
        }
        let excluded_classes = excluded.clone().reshape([rows,1]).expand([rows,classes]);
        let predictions = predictions.cast(dtype).mask_fill(excluded_classes.clone(),0);
        let targets = targets.cast(dtype).mask_fill(excluded_classes,0);
        let (mass,log_target) = if self.log_target { (targets.clone().exp(),targets) } else {
            let zero = targets.clone().equal_elem(0);
            (targets.clone(),targets.mask_fill(zero,1).log())
        };
        let zero = mass.clone().equal_elem(0);
        let difference = log_target.mask_fill(zero.clone(),0)-predictions.mask_fill(zero,0);
        let values = (mass*difference).sum_dim(1).reshape([rows]).mask_fill(excluded,0);
        let normalizers = valid.clone().float().cast(dtype);
        CategoricalLossTerms {values,normalizers,valid}
    }

    /// Actual per-token KL of [batch,tokens,classes] distributions without token averaging.
    pub fn forward_token_terms<B: Backend>(&self,predictions: Tensor<B,3>,targets: Tensor<B,3>,
        visible: Option<Tensor<B,2,Bool>>) -> CategoricalLossTerms<B,2> {
        let [batch,tokens,classes] = predictions.dims();
        assert_eq!(targets.dims(),[batch,tokens,classes],"token KL target geometry differs");
        let rows = batch.checked_mul(tokens).expect("token KL row count overflow");
        let visible = visible.map(|visible| {
            assert_eq!(visible.dims(),[batch,tokens],"token KL selection geometry differs"); visible.reshape([rows])
        });
        self.forward_terms(predictions.reshape([rows,classes]),targets.reshape([rows,classes]),visible).reshape([batch,tokens])
    }

    /// Native NCHW per-pixel KL over actual classes, with original [N,H,W] output axes.
    pub fn forward_pixel_terms<B: Backend>(&self,predictions: Tensor<B,4>,targets: Tensor<B,4>,
        visible: Option<Tensor<B,3,Bool>>) -> CategoricalLossTerms<B,3> {
        let [batch,classes,height,width] = predictions.dims();
        assert_eq!(targets.dims(),[batch,classes,height,width],"pixel KL target geometry differs");
        let rows = batch.checked_mul(height).and_then(|rows|rows.checked_mul(width)).expect("pixel KL row count overflow");
        let visible = visible.map(|visible| {
            assert_eq!(visible.dims(),[batch,height,width],"pixel KL selection geometry differs"); visible.reshape([rows])
        });
        self.forward_terms(predictions.permute([0,2,3,1]).reshape([rows,classes]),targets.permute([0,2,3,1]).reshape([rows,classes]),visible)
            .reshape([batch,height,width])
    }
}
