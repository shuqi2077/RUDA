use super::{GroupedExpertRows,MoeError,elements,float_tensor,same_device};
use rublas::tensor_grouped::{grouped_matmul_nt_segmented,grouped_matmul_nt_backward_segmented_selected,
    GroupedStrategy,GroupedGradientSelection,GroupedBackwardSelected};
use ruda_kernel::{dsl::Runtime,tensor::RudaTensor};

/// Actual original floating expert projection operands and private native row segments.
#[derive(Clone,Debug)]
pub struct ExpertProjectionCache<R:Runtime> {rows:GroupedExpertRows<R>,weights:RudaTensor<R>}
/// Safe original grouped `[rows,input] @ [experts,output,input].mT`.
/// Only private native dispatch/received producers can supply the complete expert prefix.
pub fn expert_projection<R:Runtime>(rows:&GroupedExpertRows<R>,weights:RudaTensor<R>,strategy:GroupedStrategy)
    -> Result<(RudaTensor<R>,ExpertProjectionCache<R>),MoeError> {
    float_tensor(&weights)?;same_device(&rows.values,&weights)?;elements(weights.meta.shape())?;
    if weights.meta.num_dims()!=3 || weights.meta.shape()[0]!=rows.experts || weights.meta.shape()[1]==0
        || weights.meta.shape()[2]!=rows.values.meta.shape()[1] || weights.dtype!=rows.values.dtype
        || !weights.client.same_execution_queue(&rows.values.client) {
        return Err(MoeError("floating expert projection requires actual matching cube width/storage/device/queue"));
    }
    // SAFETY: rows retain the original immutable private producer's complete monotone expert segments.
    let output=unsafe {grouped_matmul_nt_segmented(rows.values.clone(),weights.clone(),rows.row_experts.clone(),rows.offsets.clone(),strategy)}?;
    Ok((output,ExpertProjectionCache {rows:rows.clone(),weights}))
}
impl<R:Runtime> ExpertProjectionCache<R> {
    /// Original selected first-order derivatives. Weight gradients retain native FP32,
    /// independently of stored cube dtype; omitted gradients are not allocated or launched.
    pub fn backward(self,gradient:RudaTensor<R>,strategy:GroupedStrategy,selection:GroupedGradientSelection) -> Result<GroupedBackwardSelected<R>,MoeError> {
        // SAFETY: this exact original private producer's row metadata is retained unchanged.
        Ok(unsafe {grouped_matmul_nt_backward_segmented_selected(self.rows.values,self.weights,gradient,self.rows.row_experts,self.rows.offsets,strategy,selection)}?)
    }
}
