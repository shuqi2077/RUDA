use alloc::vec::Vec;
use core::ops::Range;
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,tensor_parallel as region};
use ruda_model::tensor::{Bool,DType,Int,IntDType,Tensor,TensorPrimitive,ElementConversion,backend::Backend};
use region::BroadcastTensorCollective;
use crate::loss::LossTerms;

/// Explicit rank-ordered vocabulary storage intervals, including trailing padding.
/// All ranks supply the same layout. Intervals may have different positive widths;
/// logical classes are exactly [0,vocabulary_size), not all allocated columns.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct VocabParallelLossLayout {
    offsets: Vec<usize>,
    vocabulary_size: usize,
}

impl VocabParallelLossLayout {
    /// Build contiguous storage intervals in communicator rank order.
    /// At least one real class and one storage column per rank are required.
    /// A rank can own only padding; its columns still receive zero loss gradients.
    pub fn new(widths: Vec<usize>,vocabulary_size: usize) -> Self {
        assert!(!widths.is_empty() && vocabulary_size > 0,"parallel loss needs ranks and real classes");
        let mut offsets = Vec::with_capacity(widths.len()+1);
        offsets.push(0usize);
        for width in widths {
            assert!(width > 0,"parallel loss storage intervals must be nonempty");
            let next = offsets.last().copied().unwrap().checked_add(width).expect("parallel vocabulary storage overflow");
            assert!(next <= i64::MAX as usize,"parallel vocabulary storage exceeds signed index range");
            offsets.push(next);
        }
        assert!(vocabulary_size <= *offsets.last().unwrap(),"logical vocabulary exceeds actual shard storage");
        Self {offsets,vocabulary_size}
    }

    /// Actual communicator rank count, not the number of logical classes.
    pub fn world_size(&self) -> usize {self.offsets.len()-1}
    /// Number of actual classes; all larger stored columns are padding.
    pub fn vocabulary_size(&self) -> usize {self.vocabulary_size}
    /// Actual allocated vocabulary columns across all rank-local shards.
    pub fn storage_size(&self) -> usize {*self.offsets.last().unwrap()}
    /// This rank's contiguous stored global column interval.
    pub fn interval(&self,rank: usize) -> Range<usize> {
        assert!(rank < self.world_size(),"parallel vocabulary rank is out of range");
        self.offsets[rank]..self.offsets[rank+1]
    }
}

/// One replicated logical categorical objective over actual rank-local logits.
/// All ranks must use identical rows, visibility, labels/layout and criterion options.
/// Class weights and soft labels are the actual local vocabulary slices. This is
/// tensor parallelism: do not sum the replicated losses as independent DP losses.
#[derive(Clone,Debug)]
pub struct VocabParallelCrossEntropy {
    layout: VocabParallelLossLayout,
    label_smoothing: f64,
    ignore_index: Option<i64>,
}

struct Statistics<B: Backend,S: CheckpointStrategy> {
    shifted: Tensor<Autodiff<B,S>,2>,
    log_sum: Tensor<Autodiff<B,S>,1>,
    valid: Tensor<Autodiff<B,S>,1,Bool>,
    weights: Option<Tensor<Autodiff<B,S>,2>>,
}

pub(super) fn floating(dtype: DType) {
    assert!(matches!(dtype,DType::F16|DType::BF16|DType::F32|DType::Flex32|DType::F64),
        "vocabulary loss requires native floating operands");
}

pub(super) fn work_dtype<B:Backend,C:BroadcastTensorCollective<B>>(double:bool,device:&B::Device,communicator:&C) -> Result<DType,C::Error> {
    let flag = Tensor::<B,1>::full([1],if double {1.} else {0.},(device,DType::F32));
    let flag = communicator.all_reduce_sum(flag.into_primitive().tensor())?;
    let double = Tensor::<B,1>::from_primitive(TensorPrimitive::Float(flag)).greater_elem(0).any().into_scalar().elem::<bool>();
    Ok(if double {DType::F64} else {DType::F32})
}

fn product<B: Backend,S: CheckpointStrategy>(value: Tensor<Autodiff<B,S>,2>,coefficient: Tensor<Autodiff<B,S>,2>)
    -> Tensor<Autodiff<B,S>,2> {
    let zero_infinity = coefficient.clone().equal_elem(0).bool_and(value.clone().is_inf());
    value.mask_fill(zero_infinity,0)*coefficient
}

impl VocabParallelCrossEntropy {
    /// Use explicit full-class smoothing and an optional hard-label ignore sentinel.
    /// Soft-label selection uses only explicit visibility, never a hard-label sentinel.
    pub fn new(layout: VocabParallelLossLayout,label_smoothing: f64,ignore_index: Option<i64>) -> Self {
        assert!(label_smoothing.is_finite() && (0.0..=1.0).contains(&label_smoothing),"invalid full-vocabulary smoothing");
        Self {layout,label_smoothing,ignore_index}
    }

    /// Actual partition retained by this criterion.
    pub fn layout(&self) -> &VocabParallelLossLayout {&self.layout}

    fn visibility<B: Backend,S: CheckpointStrategy>(&self,logits: &Tensor<Autodiff<B,S>,2>,
        visible: Option<Tensor<Autodiff<B,S>,1,Bool>>) -> Tensor<Autodiff<B,S>,1,Bool> {
        if let Some(visible) = visible {
            assert_eq!(visible.dims(),[logits.dims()[0]],"parallel loss visibility and actual rows differ");
            assert_eq!(visible.device(),logits.device(),"parallel loss visibility must share the device");
            visible
        } else {Tensor::<Autodiff<B,S>,1,Bool>::zeros([logits.dims()[0]],&logits.device()).bool_not()}
    }

    fn statistics<B,S,C>(&self,logits: Tensor<Autodiff<B,S>,2>,valid: Tensor<Autodiff<B,S>,1,Bool>,
        class_weights: Option<Tensor<Autodiff<B,S>,1>>,target_dtype: Option<DType>,communicator: &C)
        -> Result<Statistics<B,S>,C::Error>
        where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B> {
        floating(logits.dtype());
        let [rows,width] = logits.dims();
        assert_eq!(communicator.world_size() as usize,self.layout.world_size(),"vocabulary layout and communicator size differ");
        let interval = self.layout.interval(communicator.rank() as usize);
        assert_eq!(interval.len(),width,"local logits do not match this rank's actual vocabulary storage");
        let mut double = logits.dtype() == DType::F64 || target_dtype == Some(DType::F64);
        if let Some(weights) = &class_weights {
            floating(weights.dtype());
            assert_eq!(weights.dims(),[width],"local class weights and actual vocabulary columns differ");
            assert_eq!(weights.device(),logits.device(),"parallel class weights must share the logits device");
            double |= weights.dtype() == DType::F64;
        }
        // One identical scalar collective selects a work dtype across uneven shards.
        let dtype = work_dtype::<B,C>(double,&logits.device(),communicator)?;
        let logits = logits.cast(dtype);
        let real = Tensor::<Autodiff<B,S>,1,Int>::arange(interval.start as i64..interval.end as i64,(&logits.device(),DType::I64))
            .lower_elem(self.layout.vocabulary_size as i64).reshape([1,width]).expand([rows,width]);
        let selected = valid.clone().reshape([rows,1]).expand([rows,width]).bool_and(real.clone());
        let weights = class_weights.map(|weights|weights.cast(dtype).reshape([1,width]).expand([rows,width]).mask_fill(selected.clone().bool_not(),0));
        if rows == 0 {
            let log_sum = logits.clone().sum_dim(1).reshape([0]);
            return Ok(Statistics {shifted:logits,log_sum,valid,weights});
        }
        let logits = logits.mask_fill(valid.clone().bool_not().reshape([rows,1]).expand([rows,width]),0)
            .mask_fill(real.clone().bool_not(),f64::NEG_INFINITY);
        let local_max = logits.clone().detach().max_dim(1).inner();
        let maxima = communicator.all_gather_float(local_max.into_primitive().tensor())?;
        let maximum = Tensor::<B,2>::from_primitive(TensorPrimitive::Float(maxima))
            .reshape([self.layout.world_size(),rows]).max_dim(0).reshape([rows,1]);
        let shifted = logits-Tensor::<Autodiff<B,S>,2>::from_inner(maximum);
        let sums = region::reduce_from_region(shifted.clone().exp().sum_dim(1),communicator.clone())?;
        let log_sum = sums.log().reshape([rows]);
        Ok(Statistics {shifted:shifted.mask_fill(selected.bool_not(),0),log_sum,valid,weights})
    }

    /// Native hard-label terms without collecting full vocabulary logits.
    /// Mean normalizes by selected target-class weights. Smoothing uses every real
    /// class, excludes stored padding, and leaves the hard-target denominator intact.
    /// FP64 on any rank selects FP64 work on all ranks; other float storage uses FP32.
    /// Selected labels must be real global IDs; ignored/excluded labels are not gathered.
    pub fn forward_terms<B,S,C>(&self,logits: Tensor<Autodiff<B,S>,2>,labels: Tensor<Autodiff<B,S>,1,Int>,communicator: C,
        visible: Option<Tensor<Autodiff<B,S>,1,Bool>>,class_weights: Option<Tensor<Autodiff<B,S>,1>>)
        -> Result<LossTerms<Autodiff<B,S>>,C::Error>
        where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B> {
        assert_eq!(labels.dims(),[logits.dims()[0]],"parallel loss labels and actual rows differ");
        assert_eq!(labels.device(),logits.device(),"parallel loss labels must share the device");
        let [rows,width] = logits.dims();
        let labels = labels.cast(IntDType::I64);
        let mut valid = self.visibility(&logits,visible);
        if let Some(ignore) = self.ignore_index {valid = valid.bool_and(labels.clone().equal_elem(ignore).bool_not());}
        let statistics = self.statistics(logits,valid,class_weights,None,&communicator)?;
        let Statistics {shifted,log_sum,valid,weights} = statistics;
        let dtype = log_sum.dtype();
        if rows == 0 {return Ok(LossTerms {values:log_sum,normalizers:Tensor::zeros([0],(&labels.device(),dtype)),valid});}
        let invalid = labels.clone().lower_elem(0).bool_or(labels.clone().greater_equal_elem(self.layout.vocabulary_size as i64)).bool_and(valid.clone());
        let invalid = invalid.any().float().cast(DType::F32).inner();
        let invalid = communicator.all_reduce_sum(invalid.into_primitive().tensor())?;
        assert!(!Tensor::<B,1>::from_primitive(TensorPrimitive::Float(invalid)).greater_elem(0).any().into_scalar().elem::<bool>(),
            "selected target lies outside the logical vocabulary");
        let interval = self.layout.interval(communicator.rank() as usize);
        let owned = labels.clone().greater_equal_elem(interval.start as i64)
            .bool_and(labels.clone().lower_elem(interval.end.min(self.layout.vocabulary_size) as i64)).bool_and(valid.clone());
        let indices = labels.sub_scalar(interval.start as i64).mask_fill(owned.clone().bool_not(),0).reshape([rows,1]);
        let normalizers = if let Some(weights) = &weights {
            let selected = weights.clone().gather(1,indices.clone()).reshape([rows]).mask_fill(owned.clone().bool_not(),0);
            region::reduce_from_region(selected,communicator.clone())?
        } else {valid.clone().float().cast(dtype)};
        let selected = shifted.clone().gather(1,indices).mask_fill(owned.bool_not().reshape([rows,1]),0);
        let selected = region::reduce_from_region(selected,communicator.clone())?.reshape([rows]);
        let mut values = if self.label_smoothing == 1. {log_sum.zeros_like()} else {
            let loss = (log_sum.clone()-selected).reshape([rows,1]);
            product(loss,normalizers.clone().reshape([rows,1])).reshape([rows]).mul_scalar(1.-self.label_smoothing)
        };
        if self.label_smoothing > 0. {
            let (mass,weighted) = if let Some(weights) = weights {
                let mass = region::reduce_from_region(weights.clone().sum_dim(1),communicator.clone())?.reshape([rows]);
                let weighted = region::reduce_from_region(product(shifted,weights).sum_dim(1),communicator)?.reshape([rows]);
                (mass,weighted)
            } else {
                let mass = log_sum.full_like(self.layout.vocabulary_size as f64);
                let weighted = region::reduce_from_region(shifted.sum_dim(1),communicator)?.reshape([rows]);
                (mass,weighted)
            };
            let uniform = log_sum*mass-weighted;
            values = values+uniform.mul_scalar(self.label_smoothing/self.layout.vocabulary_size as f64);
        }
        Ok(LossTerms {values:values.mask_fill(valid.clone().bool_not(),0),normalizers,valid})
    }

    /// Actual rank-local slices of a full soft-label distribution, without rounding,
    /// detachment or renormalization. Mean uses selected rows, not class-weight mass.
    /// Global normalization mass is reduced before applying log-sum-exp so local
    /// derivatives include every shard's target mass, not only this rank's mass.
    pub fn forward_soft_terms<B,S,C>(&self,logits: Tensor<Autodiff<B,S>,2>,targets: Tensor<Autodiff<B,S>,2>,communicator: C,
        visible: Option<Tensor<Autodiff<B,S>,1,Bool>>,class_weights: Option<Tensor<Autodiff<B,S>,1>>)
        -> Result<LossTerms<Autodiff<B,S>>,C::Error>
        where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B> {
        assert_eq!(logits.dims(),targets.dims(),"parallel soft labels and actual local logits differ");
        assert_eq!(logits.device(),targets.device(),"parallel soft labels must share the device");
        floating(targets.dtype());
        let [rows,width] = logits.dims();
        let valid = self.visibility(&logits,visible);
        let Statistics {shifted,log_sum,valid,weights} = self.statistics(logits,valid,class_weights,Some(targets.dtype()),&communicator)?;
        let dtype = log_sum.dtype();
        if rows == 0 {return Ok(LossTerms {values:log_sum+targets.cast(dtype).sum_dim(1).reshape([0]),
            normalizers:Tensor::zeros([0],(&valid.device(),dtype)),valid});}
        let interval = self.layout.interval(communicator.rank() as usize);
        let real = Tensor::<Autodiff<B,S>,1,Int>::arange(interval.start as i64..interval.end as i64,(&targets.device(),DType::I64))
            .lower_elem(self.layout.vocabulary_size as i64).reshape([1,width]).expand([rows,width]);
        let selected = valid.clone().reshape([rows,1]).expand([rows,width]).bool_and(real);
        let targets = targets.cast(dtype).mask_fill(selected.clone().bool_not(),0);
        let targets = targets.mul_scalar(1.-self.label_smoothing).add_scalar(self.label_smoothing/self.layout.vocabulary_size as f64)
            .mask_fill(selected.bool_not(),0);
        let coefficients = if let Some(weights) = weights {targets*weights} else {targets};
        let mass = region::reduce_from_region(coefficients.clone().sum_dim(1),communicator.clone())?.reshape([rows]);
        let weighted = region::reduce_from_region(product(shifted,coefficients).sum_dim(1),communicator)?.reshape([rows]);
        let values = (product(log_sum.reshape([rows,1]),mass.reshape([rows,1])).reshape([rows])-weighted).mask_fill(valid.clone().bool_not(),0);
        Ok(LossTerms {values,normalizers:valid.clone().float().cast(dtype),valid})
    }

    /// Actual [batch,tokens,local_vocab] hard-label loss, with no causal shifting.
    pub fn forward_token_terms<B,S,C>(&self,logits: Tensor<Autodiff<B,S>,3>,labels: Tensor<Autodiff<B,S>,2,Int>,communicator: C,
        visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,class_weights: Option<Tensor<Autodiff<B,S>,1>>)
        -> Result<LossTerms<Autodiff<B,S>,2>,C::Error>
        where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B> {
        let [batch,tokens,width] = logits.dims();
        assert_eq!(labels.dims(),[batch,tokens],"parallel token targets differ from actual axes");
        let rows = batch.checked_mul(tokens).expect("parallel token row count overflow");
        let visible = visible.map(|mask| {assert_eq!(mask.dims(),[batch,tokens],"parallel token visibility differs");mask.reshape([rows])});
        self.forward_terms(logits.reshape([rows,width]),labels.reshape([rows]),communicator,visible,class_weights)
            .map(|terms|terms.reshape([batch,tokens]))
    }

    /// Actual full soft distributions supplied as [batch,tokens,local_vocab] slices.
    pub fn forward_soft_token_terms<B,S,C>(&self,logits: Tensor<Autodiff<B,S>,3>,targets: Tensor<Autodiff<B,S>,3>,communicator: C,
        visible: Option<Tensor<Autodiff<B,S>,2,Bool>>,class_weights: Option<Tensor<Autodiff<B,S>,1>>)
        -> Result<LossTerms<Autodiff<B,S>,2>,C::Error>
        where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B> {
        let [batch,tokens,width] = logits.dims();
        assert_eq!(targets.dims(),[batch,tokens,width],"parallel soft token targets differ from actual axes");
        let rows = batch.checked_mul(tokens).expect("parallel soft token row count overflow");
        let visible = visible.map(|mask| {assert_eq!(mask.dims(),[batch,tokens],"parallel soft token visibility differs");mask.reshape([rows])});
        self.forward_soft_terms(logits.reshape([rows,width]),targets.reshape([rows,width]),communicator,visible,class_weights)
            .map(|terms|terms.reshape([batch,tokens]))
    }
}

impl VocabParallelLossLayout {
    /// Normalize actual local logit shards over every real global class without gathering logits.
    /// Returns native FP32 or globally selected FP64 log probabilities. Stored padding is -Inf;
    /// explicitly excluded rows are zero and must keep that selection in downstream objectives.
    /// Backward SUMs class-shard contributions to the shared normalizer for one logical TP loss.
    pub fn log_softmax<B,S,C>(&self,logits:Tensor<Autodiff<B,S>,2>,communicator:C,visible:Option<Tensor<Autodiff<B,S>,1,Bool>>)
        -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B> {
        self.log_softmax_for_dtype(logits,communicator,visible,None)
    }

    pub(super) fn log_softmax_for_dtype<B,S,C>(&self,logits:Tensor<Autodiff<B,S>,2>,communicator:C,
        visible:Option<Tensor<Autodiff<B,S>,1,Bool>>,target_dtype:Option<DType>) -> Result<Tensor<Autodiff<B,S>,2>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B> {
        let criterion = VocabParallelCrossEntropy::new(self.clone(),0.,None);
        let valid = criterion.visibility(&logits,visible);let [rows,width] = logits.dims();
        let Statistics {shifted,log_sum,valid,..} = criterion.statistics(logits,valid,None,target_dtype,&communicator)?;
        if rows == 0 {return Ok(shifted);}
        let log_sum = region::copy_to_region(log_sum,communicator.clone())?;
        let output = shifted-log_sum.reshape([rows,1]);
        let interval = self.interval(communicator.rank() as usize);
        let padding = Tensor::<Autodiff<B,S>,1,Int>::arange(interval.start as i64..interval.end as i64,(&output.device(),DType::I64))
            .greater_equal_elem(self.vocabulary_size() as i64).reshape([1,width]).expand([rows,width]);
        Ok(output.mask_fill(padding,f64::NEG_INFINITY).mask_fill(valid.bool_not().reshape([rows,1]).expand([rows,width]),0))
    }

    /// Native [batch,tokens,local_classes] log probabilities, retaining actual token axes.
    pub fn log_softmax_tokens<B,S,C>(&self,logits:Tensor<Autodiff<B,S>,3>,communicator:C,visible:Option<Tensor<Autodiff<B,S>,2,Bool>>)
        -> Result<Tensor<Autodiff<B,S>,3>,C::Error>
        where B:Backend,S:CheckpointStrategy,C:BroadcastTensorCollective<B> {
        let [batch,tokens,width] = logits.dims();let rows = batch.checked_mul(tokens).expect("parallel normalization token row count overflow");
        let visible = visible.map(|mask| {assert_eq!(mask.dims(),[batch,tokens],"parallel normalization token visibility differs");mask.reshape([rows])});
        self.log_softmax(logits.reshape([rows,width]),communicator,visible).map(|values|values.reshape([batch,tokens,width]))
    }
}
