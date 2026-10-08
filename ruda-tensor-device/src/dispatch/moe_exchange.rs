use crate::{DeviceBackend,DeviceRuntime,FloatElement,IntElement,element::BoolElement};
use ruda_tensor::{moe::{MoeOptions,MoeCombineGradientStrategy},moe_exchange::*,tensor::{FloatTensor,IntTensor}};
use rudnn::moe::{self,MoeError,RouterTrainingPlan,DispatchedTokens,ReceivedExpertRows,ExpertTrainingCache,SwiGluExperts};
use ruda_core::tensor::DType;
use ruda_kernel::tensor::{RudaTensor,contiguous::into_contiguous,allocation::empty_device_contiguous_dtype};
use super::moe::{weight_options,expert_strategy,select};

/// Source-local actual native discrete mappings and continuous router VJP.
#[derive(Debug,Clone)]
pub struct NativeMoeDispatchState<R:DeviceRuntime> {router:RouterTrainingPlan<R>,dispatched:DispatchedTokens<R>}
/// Actual local received rows/cache, or the exact empty-owner original source metadata.
#[derive(Debug,Clone)]
pub enum NativeMoeReceivedState<R:DeviceRuntime> {
    /// Validated native private row mapping and original expert cache when requested.
    Grouped {rows:ReceivedExpertRows<R>,cache:Option<ExpertTrainingCache<R>>,options:MoeReceivedOptions},
    /// True zero-expert owner: every original local cube and input has zero rows.
    Empty {input:RudaTensor<R>,gate:RudaTensor<R>,up:RudaTensor<R>,down:RudaTensor<R>},
}
fn read_u32<R:DeviceRuntime>(tensor:RudaTensor<R>) -> Result<Vec<u32>,MoeError> {
    ruda_core::future::block_on(ruda_kernel::tensor::readback::into_data(tensor)).map_err(|_|MoeError("native expert prefix metadata read failed"))?
        .to_vec::<u32>().map_err(|_|MoeError("native expert prefix metadata storage mismatch"))
}
fn combine_strategy(strategy:MoeCombineGradientStrategy) -> moe::CombineGradientStrategy {
    match strategy {MoeCombineGradientStrategy::Serial=>moe::CombineGradientStrategy::Serial,MoeCombineGradientStrategy::Plane=>moe::CombineGradientStrategy::Plane}
}
impl<R,F,I,BT> MoeDispatchOps for DeviceBackend<R,F,I,BT> where R:DeviceRuntime,F:FloatElement,I:IntElement,BT:BoolElement {
    type MoeDispatchState=NativeMoeDispatchState<R>;
    fn moe_dispatch(input:FloatTensor<Self>,logits:FloatTensor<Self>,bias:Option<FloatTensor<Self>>,options:MoeOptions) -> Result<MoeDispatched<Self>,Self::MoeError> {
        let logits=into_contiguous(logits);
        for value in [&input].into_iter().chain(bias.iter()) {if value.device!=logits.device || !value.client.same_execution_queue(&logits.client) {
            return Err(MoeError("native dispatch operands must share the original device and queue"));}}
        let router=select(logits.clone(),bias,options.selection)?.into_training(logits,weight_options(options.weights))?;
        let dispatched=router.routing().clone().dispatch(input)?;
        Ok(MoeDispatched {values:dispatched.values().clone(),weights:router.routing().weights().clone(),selected_experts:router.routing().expert_indices().clone(),
            row_experts:dispatched.row_experts().clone(),state:NativeMoeDispatchState {router,dispatched}})
    }
    fn moe_dispatch_counts(state:&Self::MoeDispatchState,expert_prefix:&[usize]) -> Result<Vec<usize>,Self::MoeError> {
        let experts=state.router.routing().experts();if expert_prefix.len()<2 || expert_prefix[0]!=0 || expert_prefix.last()!=Some(&experts)
            || expert_prefix.windows(2).any(|pair|pair[0]>pair[1]) {return Err(MoeError("expert ownership must be a complete nondecreasing original global prefix"));}
        let offsets=read_u32(state.dispatched.expert_offsets().clone())?;
        if offsets.len()!=experts+1 || offsets.first()!=Some(&0) || offsets.windows(2).any(|pair|pair[0]>pair[1])
            || offsets.last().map(|&value|value as usize)!=Some(state.dispatched.values().meta.shape()[0]) {return Err(MoeError("native expert assignment prefix metadata mismatch"));}
        Ok(expert_prefix.windows(2).map(|pair|(offsets[pair[1]]-offsets[pair[0]]) as usize).collect())
    }
    fn moe_dispatch_backward(state:Self::MoeDispatchState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::MoeError> {state.dispatched.dispatch_backward(gradient)}
    fn moe_dispatch_weights_backward(state:Self::MoeDispatchState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::MoeError> {state.router.backward(&gradient)}
    fn moe_combine(state:Self::MoeDispatchState,expert_values:FloatTensor<Self>,weights:FloatTensor<Self>,_backward_strategy:MoeCombineGradientStrategy)
        -> Result<FloatTensor<Self>,Self::MoeError> {
        state.dispatched.with_weights(weights)?.combine(expert_values)
    }
    fn moe_combine_backward(state:Self::MoeDispatchState,expert_values:FloatTensor<Self>,weights:FloatTensor<Self>,gradient:FloatTensor<Self>,
        strategy:MoeCombineGradientStrategy,selection:MoeCombineSelection) -> Result<MoeCombineBackward<Self>,Self::MoeError> {
        let result=state.dispatched.with_weights(weights)?.combine_backward_selected(&expert_values,gradient,combine_strategy(strategy),
            moe::CombineGradientSelection {experts:selection.experts,weights:selection.weights})?;
        Ok(MoeCombineBackward {experts:result.dexpert,weights:result.dweights})
    }
}
impl<R,F,I,BT> MoeReceivedOps for DeviceBackend<R,F,I,BT> where R:DeviceRuntime,F:FloatElement,I:IntElement,BT:BoolElement {
    type MoeReceivedState=NativeMoeReceivedState<R>;
    fn moe_received_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,
        options:MoeReceivedOptions,selection:MoeReceivedSelection) -> Result<(FloatTensor<Self>,Self::MoeReceivedState),Self::MoeError> {
        if input.meta.num_dims()!=2 || gate.meta.num_dims()!=3 || up.meta.shape()!=gate.meta.shape() || down.meta.num_dims()!=3 {
            return Err(MoeError("native received experts require original input and gate/up/down cube ranks"));}
        let (experts,inner,hidden)=(gate.meta.shape()[0],gate.meta.shape()[1],gate.meta.shape()[2]);
        if inner==0 || hidden==0 || down.meta.shape()[..]!=[experts,hidden,inner] || input.meta.shape()[1]!=hidden
            || global_ids.meta.shape()[..]!=[input.meta.shape()[0]] || global_ids.dtype!=DType::U32 || global_ids.qparams.is_some()
            || options.expert_start.checked_add(experts).is_none_or(|end|end>u32::MAX as usize) {return Err(MoeError("native received local cube geometry/global expert range mismatch"));}
        for value in [&input,&gate,&up,&down] {
            if !matches!(value.dtype,DType::F16|DType::BF16|DType::F32) || value.qparams.is_some() || value.dtype!=input.dtype || value.device!=input.device
                || !value.client.same_execution_queue(&input.client) {return Err(MoeError("native received expert storage/device/queue mismatch"));}}
        for value in [&input,&gate,&up,&down] {if value.meta.shape().iter().any(|&axis|axis>u32::MAX as usize)
            || value.meta.shape().iter().try_fold(1usize,|size,&axis|size.checked_mul(axis)).is_none_or(|size|size>u32::MAX as usize) {
            return Err(MoeError("native received expert tensor exceeds U32 indexing"));}}
        if global_ids.device!=input.device || !global_ids.client.same_execution_queue(&input.client) {return Err(MoeError("received U32 assignments must share the actual input queue"));}
        if experts==0 {
            if input.meta.shape()[0]!=0 {return Err(MoeError("a zero-expert owner cannot receive nonempty assignment rows"));}
            let output=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),input.meta.shape().clone(),input.dtype);
            return Ok((output,NativeMoeReceivedState::Empty {input,gate,up,down}));
        }
        let rows=ReceivedExpertRows::new(input,global_ids,options.expert_start,experts)?;let experts=SwiGluExperts::new(gate,up,down)?;
        let (expert_output,cache)=if selection.input || selection.gate || selection.up || selection.down {
            let trained=experts.forward_grouped_training(rows.grouped(),expert_strategy(options.forward))?;(trained.output,Some(trained.cache))
        } else {(experts.forward_grouped_rows(rows.grouped(),expert_strategy(options.forward))?,None)};
        let output=rows.restore(expert_output)?;Ok((output,NativeMoeReceivedState::Grouped {rows,cache,options}))
    }
    fn moe_received_backward(state:Self::MoeReceivedState,gradient:FloatTensor<Self>,selection:MoeReceivedSelection)
        -> Result<MoeReceivedBackward<Self>,Self::MoeError> {
        match state {
            NativeMoeReceivedState::Empty {input,gate,up,down}=>{
                if gradient.meta.shape()!=input.meta.shape() || gradient.dtype!=input.dtype || gradient.device!=input.device
                    || !gradient.client.same_execution_queue(&input.client) || gradient.qparams.is_some() {return Err(MoeError("empty-owner received seed metadata mismatch"));}
                let empty=|like:RudaTensor<R>,dtype|empty_device_contiguous_dtype(like.client.clone(),like.device.clone(),like.meta.shape().clone(),dtype);
                Ok(MoeReceivedBackward {input:selection.input.then(||empty(input,gradient.dtype)),gate:selection.gate.then(||empty(gate,DType::F32)),
                    up:selection.up.then(||empty(up,DType::F32)),down:selection.down.then(||empty(down,DType::F32))})
            },
            NativeMoeReceivedState::Grouped {rows,cache,options}=>{
                if !selection.input && !selection.gate && !selection.up && !selection.down {return Ok(MoeReceivedBackward {input:None,gate:None,up:None,down:None});}
                let cache=cache.ok_or(MoeError("received forward did not retain the requested expert VJP cache"))?;
                let result=cache.backward_selected(rows.sort_gradient(gradient)?,expert_strategy(options.backward),
                    moe::ExpertGradientSelection {input:selection.input,gate:selection.gate,up:selection.up,down:selection.down})?;
                Ok(MoeReceivedBackward {input:result.dinput.map(|value|rows.restore(value)).transpose()?,gate:result.dgate,up:result.dup,down:result.ddown})
            },
        }
    }
}
