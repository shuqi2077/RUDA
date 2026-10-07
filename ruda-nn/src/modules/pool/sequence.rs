use ruda_model::{config::Config,tensor::{Bool,DType,FloatDType,Int,IntDType,Tensor,backend::Backend}};
use alloc::vec::Vec;
use crate::attention::PackedSequenceLayout;

/// Explicit reduction over real tokens, independent of left/right padding.
#[derive(Config,Debug,Copy)]
pub enum SequencePooling {
    /// Average actual visible token rows with FP32 statistics for half storage.
    Mean,
    /// Featurewise maximum over actual visible tokens.
    Max,
    /// Select the first actual visible token, not physical column zero.
    First,
    /// Select the last actual visible token, not the physical last column.
    Last,
}

/// Pooled activations and actual visibility/count information.
#[derive(Clone,Debug)]
pub struct SequencePoolOutput<B: Backend> {
    /// [batch,width], retaining hidden storage. Empty/all-excluded rows are zeros.
    pub values: Tensor<B,2>,
    /// [batch], True exactly when a real token contributed to that row.
    pub valid_rows: Tensor<B,1,Bool>,
    /// [batch], actual visible token counts in I64, not physical sequence length.
    pub token_counts: Tensor<B,1,Int>,
}

/// Pool native [batch,tokens,width] states using the actual visibility mask.
///
/// Mask True means a real token. Excluded NaN/Inf values do not contaminate
/// valid tokens or all-excluded rows. Selected nonfinite values are not repaired.
/// This does not infer padding IDs, a CLS token, position IDs or loss labels.
pub fn pool_sequence<B: Backend>(hidden: Tensor<B,3>,visible: Tensor<B,2,Bool>,
    pooling: SequencePooling) -> SequencePoolOutput<B> {
    let [batch,tokens,width] = hidden.dims();
    assert!(width > 0,"sequence pooling requires a feature axis");
    assert_eq!(visible.dims(),[batch,tokens],"pool visibility differs from actual token geometry");
    let device = hidden.device();
    assert_eq!(visible.device(),device,"sequence pool payload/mask devices differ");
    let storage = hidden.dtype();
    assert!(matches!(storage,DType::F16|DType::BF16|DType::F32|DType::F64),"pooling requires supported floating storage");
    let compute = if storage == DType::F64 { DType::F64 } else { DType::F32 };
    if batch == 0 || tokens == 0 {
        let excluded = Tensor::<B,3,Bool>::zeros(hidden.dims(),&device).bool_not();
        let zero = hidden.cast(compute).mask_fill(excluded,0).sum().reshape([1,1]);
        return SequencePoolOutput {
            values:(Tensor::<B,2>::zeros([batch,width],(&device,compute))+zero).cast(storage),
            valid_rows:Tensor::<B,1,Bool>::zeros([batch],&device),
            token_counts:Tensor::<B,1,Int>::zeros([batch],(&device,DType::I64)),
        };
    }
    assert!(tokens <= i64::MAX as usize,"sequence token indices exceed I64");
    let counts = visible.clone().cast(IntDType::I64).sum_dim(1).reshape([batch]);
    let valid = counts.clone().greater_elem(0);
    let excluded = visible.clone().bool_not().reshape([batch,tokens,1]).expand([batch,tokens,width]);
    let values = match pooling {
        SequencePooling::Mean => {
            let total = hidden.cast(compute).mask_fill(excluded,0).sum_dim(1).reshape([batch,width]);
            total / counts.clone().cast(FloatDType::from(compute)).clamp_min(1).reshape([batch,1])
        }
        SequencePooling::Max => {
            let maximum = hidden.cast(compute).mask_fill(excluded,f64::NEG_INFINITY).max_dim(1).reshape([batch,width]);
            maximum.mask_fill(valid.clone().bool_not().reshape([batch,1]).expand([batch,width]),0)
        }
        SequencePooling::First | SequencePooling::Last => {
            let positions = Tensor::<B,1,Int>::arange(0..tokens as i64,(&device,DType::I64))
                .reshape([1,tokens]).expand([batch,tokens]);
            let positions = if matches!(pooling,SequencePooling::First) {
                positions.mask_fill(visible.clone().bool_not(),tokens as i64).min_dim(1)
            } else { positions.mask_fill(visible.bool_not(),-1).max_dim(1) };
            let positions = positions.mask_fill(valid.clone().bool_not().reshape([batch,1]),0)
                .reshape([batch,1,1]).expand([batch,1,width]);
            hidden.gather(1,positions).reshape([batch,width]).cast(compute)
                .mask_fill(valid.clone().bool_not().reshape([batch,1]).expand([batch,width]),0)
        }
    };
    SequencePoolOutput {values:values.cast(storage),valid_rows:valid,token_counts:counts}
}

/// Gather explicitly supplied token positions without replacing them with CLS/last.
/// Backend indexing validates actual I32/I64 values; no activation readback occurs.
pub fn gather_sequence_positions<B: Backend>(hidden: Tensor<B,3>,positions: Tensor<B,1,Int>) -> Tensor<B,2> {
    let [batch,_,width] = hidden.dims();
    assert_eq!(positions.dims(),[batch],"one explicit pooling position per sequence is required");
    assert_eq!(positions.device(),hidden.device(),"sequence position/device mismatch");
    hidden.gather(1,positions.reshape([batch,1,1]).expand([batch,1,width])).reshape([batch,width])
}

/// Pool each actual packed document independently, including empty documents.
/// Optional True entries select contributing real tokens; no prompt span is guessed.
/// Returned counts describe pooling, not loss supervision or physical batch padding.
pub fn pool_packed_sequences<B: Backend>(hidden: Tensor<B,2>,layout: &PackedSequenceLayout,
    visible: Option<Tensor<B,1,Bool>>,pooling: SequencePooling) -> SequencePoolOutput<B> {
    let [tokens,width] = hidden.dims();
    assert!(width > 0,"packed pooling requires a feature axis");
    assert_eq!(tokens,layout.tokens(),"packed pooling boundaries differ from actual payload");
    let device = hidden.device();
    let storage = hidden.dtype();
    assert!(matches!(storage,DType::F16|DType::BF16|DType::F32|DType::F64),"packed pooling requires supported floating storage");
    if let Some(mask) = &visible {
        assert_eq!(mask.dims(),[tokens],"packed pooling visibility differs from actual tokens");
        assert_eq!(mask.device(),device,"packed pooling payload/mask devices differ");
    }
    if layout.documents() == 0 {
        let excluded = Tensor::<B,2,Bool>::zeros(hidden.dims(),&device).bool_not();
        let compute = if storage == DType::F64 { DType::F64 } else { DType::F32 };
        let zero = hidden.cast(compute).mask_fill(excluded,0).sum().reshape([1,1]);
        return SequencePoolOutput {values:(Tensor::<B,2>::zeros([0,width],(&device,compute))+zero).cast(storage),
            valid_rows:Tensor::<B,1,Bool>::zeros([0],&device),token_counts:Tensor::<B,1,Int>::zeros([0],(&device,DType::I64))};
    }
    let mut values = Vec::with_capacity(layout.documents());
    let mut valid = Vec::with_capacity(layout.documents());
    let mut counts = Vec::with_capacity(layout.documents());
    for boundary in layout.boundaries().windows(2) {
        let length = boundary[1]-boundary[0];
        let input = hidden.clone().slice_dim(0,boundary[0]..boundary[1]).reshape([1,length,width]);
        let mask = if let Some(mask) = &visible {
            mask.clone().slice_dim(0,boundary[0]..boundary[1]).reshape([1,length])
        } else { Tensor::<B,2,Bool>::zeros([1,length],&device).bool_not() };
        let pooled = pool_sequence(input,mask,pooling);
        values.push(pooled.values);
        valid.push(pooled.valid_rows);
        counts.push(pooled.token_counts);
    }
    SequencePoolOutput {values:Tensor::cat(values,0),valid_rows:Tensor::cat(valid,0),token_counts:Tensor::cat(counts,0)}
}
