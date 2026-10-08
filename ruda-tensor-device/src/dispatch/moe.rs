use crate::{DeviceBackend,DeviceRuntime,FloatElement,IntElement,element::BoolElement};
use ruda_tensor::{moe::{MoeOps,MoeOptions,MoeSelectionOptions,MoeRouterScoring,MoeRouterWeightOptions,
    MoeExpertStrategy,MoeCombineGradientStrategy,MoeBackward,MoeGradientSelection,MoeBackwardSelected},tensor::{FloatTensor,IntTensor}};
use rudnn::moe::{self,RoutingPlan,RouterTrainingPlan,DispatchedTokens,ExpertTrainingCache,MoeError,SwiGluExperts};
use ruda_kernel::tensor::{RudaTensor,contiguous::into_contiguous};

/// Actual native forward state retaining unchanged original logits and dispatch metadata.
#[derive(Debug,Clone)]
pub struct NativeMoeState<R:DeviceRuntime> {
    router:RouterTrainingPlan<R>,
    dispatched:DispatchedTokens<R>,
    experts:ExpertTrainingCache<R>,
    expert_output:RudaTensor<R>,
    options:MoeOptions,
}
fn weight_options(options:MoeRouterWeightOptions) -> moe::RouterWeightOptions {
    moe::RouterWeightOptions {scoring:match options.scoring {MoeRouterScoring::Softmax=>moe::RouterScoring::Softmax,MoeRouterScoring::Sigmoid=>moe::RouterScoring::Sigmoid},
        renormalize:options.renormalize,scale:options.scale}
}
fn expert_strategy(strategy:MoeExpertStrategy) -> moe::GroupedStrategy {
    match strategy {MoeExpertStrategy::Scalar=>moe::GroupedStrategy::Scalar,MoeExpertStrategy::Auto=>moe::GroupedStrategy::Auto,MoeExpertStrategy::TensorCore=>moe::GroupedStrategy::TensorCore}
}
fn select<R:DeviceRuntime>(logits:RudaTensor<R>,bias:Option<RudaTensor<R>>,options:MoeSelectionOptions) -> Result<RoutingPlan<R>,MoeError> {
    match options {
        MoeSelectionOptions::Softmax {top_k,renormalize}=>{
            if bias.is_some() {return Err(MoeError("softmax selection does not accept a sigmoid correction bias"));}
            moe::route(logits,moe::RoutingOptions {top_k,renormalize})
        },
        MoeSelectionOptions::SigmoidGrouped {top_k,groups,selected_groups,group_top_two,renormalize,scale}=>
            moe::route_sigmoid_grouped(logits,bias,moe::GroupRoutingOptions {top_k,groups,selected_groups,group_top_two,renormalize,scale}),
    }
}
fn prepare<R:DeviceRuntime>(input:RudaTensor<R>,logits:RudaTensor<R>,bias:Option<RudaTensor<R>>,gate:RudaTensor<R>,up:RudaTensor<R>,down:RudaTensor<R>,options:MoeOptions)
    -> Result<(SwiGluExperts<R>,DispatchedTokens<R>,RouterTrainingPlan<R>),MoeError> {
    let logits=into_contiguous(logits);
    for value in [&input,&gate,&up,&down].into_iter().chain(bias.iter()) {
        if value.device!=logits.device || !value.client.same_execution_queue(&logits.client) {return Err(MoeError("MoE operands must share an actual device and execution queue"));}
    }
    let experts=SwiGluExperts::new(gate,up,down)?;
    let routing=select(logits.clone(),bias,options.selection)?.into_training(logits,weight_options(options.weights))?;
    let dispatched=routing.routing().clone().dispatch(input)?;
    Ok((experts,dispatched,routing))
}
impl<R,F,I,BT> MoeOps for DeviceBackend<R,F,I,BT>
    where R:DeviceRuntime,F:FloatElement,I:IntElement,BT:BoolElement {
    type MoeError=MoeError;
    type MoeState=NativeMoeState<R>;
    fn moe_selected_weights(logits:FloatTensor<Self>,indices:IntTensor<Self>,options:MoeRouterWeightOptions) -> Result<FloatTensor<Self>,Self::MoeError> {
        moe::selected_router_weights(&logits,&indices,weight_options(options))
    }
    fn moe_selected_weights_backward(logits:FloatTensor<Self>,indices:IntTensor<Self>,gradient:FloatTensor<Self>,options:MoeRouterWeightOptions)
        -> Result<FloatTensor<Self>,Self::MoeError> {
        moe::selected_router_backward(&logits,&indices,&gradient,weight_options(options))
    }
    fn moe_forward(input:FloatTensor<Self>,logits:FloatTensor<Self>,correction_bias:Option<FloatTensor<Self>>,
        gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,options:MoeOptions) -> Result<(FloatTensor<Self>,Self::MoeState),Self::MoeError> {
        let (experts,dispatched,routing)=prepare(input,logits,correction_bias,gate,up,down,options)?;
        let trained=experts.forward_dispatched_training(&dispatched,expert_strategy(options.forward))?;
        let output=dispatched.combine(trained.output.clone())?;
        Ok((output,NativeMoeState {router:routing,dispatched,experts:trained.cache,expert_output:trained.output,options}))
    }
    fn moe_inference(input:FloatTensor<Self>,logits:FloatTensor<Self>,correction_bias:Option<FloatTensor<Self>>,
        gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,options:MoeOptions) -> Result<FloatTensor<Self>,Self::MoeError> {
        let (experts,dispatched,_)=prepare(input,logits,correction_bias,gate,up,down,options)?;
        let output=experts.forward_dispatched_with_strategy(&dispatched,expert_strategy(options.forward))?;
        dispatched.combine(output)
    }
    fn moe_route_indices(state:&Self::MoeState) -> IntTensor<Self> {state.router.routing().expert_indices().clone()}
    fn moe_backward(state:Self::MoeState,gradient:FloatTensor<Self>) -> Result<MoeBackward<Self>,Self::MoeError> {
        let combine=state.dispatched.combine_backward_with_strategy(&state.expert_output,gradient,
            match state.options.combine_backward {MoeCombineGradientStrategy::Serial=>moe::CombineGradientStrategy::Serial,MoeCombineGradientStrategy::Plane=>moe::CombineGradientStrategy::Plane})?;
        let logits=state.router.backward(&combine.dweights)?;
        let experts=state.experts.backward_with_strategy(combine.dexpert,expert_strategy(state.options.backward))?;
        let input=state.dispatched.dispatch_backward(experts.dinput)?;
        Ok(MoeBackward {input,logits,gate:experts.dgate,up:experts.dup,down:experts.ddown})
    }
    fn moe_backward_selected(state:Self::MoeState,gradient:FloatTensor<Self>,selection:MoeGradientSelection)
        -> Result<MoeBackwardSelected<Self>,Self::MoeError> {
        let combine=state.dispatched.combine_backward_selected(&state.expert_output,gradient,
            match state.options.combine_backward {MoeCombineGradientStrategy::Serial=>moe::CombineGradientStrategy::Serial,MoeCombineGradientStrategy::Plane=>moe::CombineGradientStrategy::Plane},
            moe::CombineGradientSelection {experts:selection.input || selection.gate || selection.up || selection.down,weights:selection.logits})?;
        let logits=combine.dweights.map(|gradient|state.router.backward(&gradient)).transpose()?;
        let experts=combine.dexpert.map(|gradient|state.experts.backward_selected(gradient,expert_strategy(state.options.backward),
            moe::ExpertGradientSelection {input:selection.input,gate:selection.gate,up:selection.up,down:selection.down})).transpose()?;
        let (input,gate,up,down)=if let Some(experts)=experts {
            (experts.dinput.map(|gradient|state.dispatched.dispatch_backward(gradient)).transpose()?,experts.dgate,experts.dup,experts.ddown)
        } else {(None,None,None,None)};
        Ok(MoeBackwardSelected {input,logits,gate,up,down})
    }
}
