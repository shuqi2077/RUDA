use core::ops::Range;
use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,tensor_parallel as region};
use ruda_model::tensor::{backend::Backend,Tensor};
use crate::attention::{DenseAttentionMask,PackedDocumentAttentionMask};
use region::BroadcastTensorCollective;

fn check_range(heads: usize,range: &Range<usize>) {
    assert!(heads > 0 && range.start < range.end && range.end <= heads,"invalid actual global query-head interval");
}

/// Slice an explicit replicated mask into the caller's actual query-head interval.
/// Singleton-head broadcasts retain their meaning. A trainable replicated score
/// bias receives SUM gradients across head shards before its original parameters;
/// boolean visibility remains nondifferentiable and no labels/edges are inferred.
pub fn parallel_dense_head_mask<B,S,C>(mut mask: DenseAttentionMask<Autodiff<B,S>>,global_heads: usize,
    local: Range<usize>,communicator: C) -> Result<DenseAttentionMask<Autodiff<B,S>>,C::Error>
    where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B> {
    check_range(global_heads,&local);
    mask.allowed = mask.allowed.map(|allowed| {
        assert!(allowed.dims()[1] == 1 || allowed.dims()[1] == global_heads,"allowed edges do not describe singleton/global heads");
        if allowed.dims()[1] == 1 {allowed} else {allowed.slice_dim(1,local.clone())}
    });
    mask.bias = mask.bias.map(|bias| -> Result<Tensor<Autodiff<B,S>,4>,C::Error> {
        assert!(bias.dims()[1] == 1 || bias.dims()[1] == global_heads,"replicated score bias does not describe actual global heads");
        let bias = region::copy_to_region(bias,communicator)?;
        Ok(if bias.dims()[1] == 1 {bias} else {bias.slice_dim(1,local)})
    }).transpose()?;
    Ok(mask)
}

/// Corresponding per-document packed head selection with replicated bias derivatives.
/// The helper does not build a global token-by-token packed mask or gather logits.
pub fn parallel_packed_head_masks<B,S,C>(masks: &[PackedDocumentAttentionMask<Autodiff<B,S>>],global_heads: usize,
    local: Range<usize>,communicator: C) -> Result<alloc::vec::Vec<PackedDocumentAttentionMask<Autodiff<B,S>>>,C::Error>
    where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B> {
    check_range(global_heads,&local);
    masks.iter().map(|mask| {
        let allowed = mask.allowed.as_ref().map(|allowed| {
            assert!(allowed.dims()[0] == 1 || allowed.dims()[0] == global_heads,"packed allowed edges differ from actual global heads");
            if allowed.dims()[0] == 1 {allowed.clone()} else {allowed.clone().slice_dim(0,local.clone())}
        });
        let bias = mask.bias.as_ref().map(|bias| -> Result<_,C::Error> {
            assert!(bias.dims()[0] == 1 || bias.dims()[0] == global_heads,"packed replicated bias differs from actual global heads");
            let bias = region::copy_to_region(bias.clone(),communicator.clone())?;
            Ok(if bias.dims()[0] == 1 {bias} else {bias.slice_dim(0,local.clone())})
        }).transpose()?;
        Ok(PackedDocumentAttentionMask {query_valid:mask.query_valid.clone(),key_valid:mask.key_valid.clone(),allowed,bias})
    }).collect()
}
