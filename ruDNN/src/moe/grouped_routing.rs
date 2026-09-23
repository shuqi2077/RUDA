use super::{MoeError, RoutingPlan, elements, empty, float_tensor, kernels};
use ruda_core::{device::Device,tensor::DType};
use ruda_kernel::{dsl::{Runtime,calculate_ruda_count_elemwise,prelude::RudaDim},tensor::{RudaTensor,contiguous::into_contiguous}};

/// Model configuration, not a model-name switch. Bias changes expert selection
/// ONLY; returned weights always come from the uncorrected sigmoid scores.
#[derive(Debug,Clone,Copy)]
pub struct GroupRoutingOptions {
    pub top_k:usize,
    pub groups:usize,
    pub selected_groups:usize,
    /// Sum of the two largest corrected scores per group; false selects max.
    pub group_top_two:bool,
    pub renormalize:bool,
    pub scale:f32,
}
impl GroupRoutingOptions {
    pub fn validate(&self,experts:usize)->Result<(),MoeError> {
        if experts==0 || experts>1024 || self.groups==0 || self.groups>128
            || experts%self.groups!=0 || self.selected_groups==0 || self.selected_groups>self.groups
            || self.top_k==0 || self.top_k>64 || self.top_k>self.selected_groups*(experts/self.groups)
            || (self.group_top_two && experts/self.groups<2) || !self.scale.is_finite() || self.scale<=0.0
        {return Err(MoeError("invalid grouped sigmoid routing configuration"));}
        Ok(())
    }
}
/// Inference router supporting group-limited sigmoid selection and a correction
/// bias in FP32. Logits [tokens,experts], bias [experts]; tensors must share an
/// execution queue. Local arrays can spill: benchmark rather than assume faster.
/// Exact ties choose lower IDs, which can differ from an unstable reference topk.
pub fn route_sigmoid_grouped<R:Runtime>(logits:RudaTensor<R>, bias:Option<RudaTensor<R>>,
    options:GroupRoutingOptions)->Result<RoutingPlan<R>,MoeError> {
    float_tensor(&logits)?;
    if logits.meta.num_dims()!=2 {return Err(MoeError("grouped router logits must be rank two"));}
    let tokens=logits.meta.shape()[0];let experts=logits.meta.shape()[1];options.validate(experts)?;
    elements(&[tokens,experts])?;elements(&[tokens,options.top_k])?;
    if let Some(b)=&bias {
        if b.meta.shape()[..]!=[experts] || b.dtype!=DType::F32 || b.qparams.is_some()
            || b.device.to_id()!=logits.device.to_id() || !b.client.same_execution_queue(&logits.client)
        {return Err(MoeError("router correction bias requires same-queue FP32 [experts]"));}
    }
    let indices=empty(&logits,[tokens,options.top_k],DType::U32);
    let weights=empty(&logits,[tokens,options.top_k],logits.dtype);
    if tokens!=0 {
        let logits=into_contiguous(logits);let has_bias=bias.is_some();
        // When bias is absent the kernel's compile-time branch never reads this.
        let bias=bias.map(into_contiguous).unwrap_or_else(||empty(&logits,[1],DType::F32));
        let dim=RudaDim::new(logits.client.properties(),tokens);
        kernels::grouped_routing::route::launch::<R>(&logits.client,
            calculate_ruda_count_elemwise(&logits.client,tokens,dim),dim,
            logits.clone().into_array_arg(),bias.into_array_arg(),indices.clone().into_array_arg(),weights.clone().into_array_arg(),
            options.scale,experts,options.groups,options.selected_groups,options.top_k,
            options.group_top_two,has_bias,options.renormalize,logits.dtype.into());
    }
    Ok(RoutingPlan{indices,weights,tokens,experts,top_k:options.top_k})
}
