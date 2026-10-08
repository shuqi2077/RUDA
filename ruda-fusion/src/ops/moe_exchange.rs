use crate::{Fusion,FusionBackend,get_client,ops::NoOp,stream::OperationStreams};
use super::frozen_awq::register_output;
use ruda_tensor::{TensorMetadata,moe::{MoeOptions,MoeCombineGradientStrategy},moe_exchange::*,tensor::{FloatTensor,IntTensor},
    graph::{InitOperationIr,OperationIr,OperationOutput}};

fn resolve<B:FusionBackend>(value:FloatTensor<Fusion<B>>) -> FloatTensor<B> {value.client.clone().resolve_tensor_float::<B>(value)}
fn register_int<B:FusionBackend>(value:IntTensor<B>) -> IntTensor<Fusion<B>> {
    let client=get_client::<B>(&B::int_device(&value));
    let desc=InitOperationIr::create(value.shape(),value.dtype(),||client.register_tensor_handle(B::int_tensor_handle(value)));
    client.register(OperationStreams::default(),OperationIr::Init(desc),NoOp::<B>::new()).output()
}
impl<B:FusionBackend+MoeDispatchOps> MoeDispatchOps for Fusion<B> {
    type MoeDispatchState=B::MoeDispatchState;
    fn moe_dispatch(input:FloatTensor<Self>,logits:FloatTensor<Self>,bias:Option<FloatTensor<Self>>,options:MoeOptions) -> Result<MoeDispatched<Self>,Self::MoeError> {
        let result=B::moe_dispatch(resolve::<B>(input),resolve::<B>(logits),bias.map(resolve::<B>),options)?;
        Ok(MoeDispatched {values:register_output::<B>(result.values),weights:register_output::<B>(result.weights),
            selected_experts:register_int::<B>(result.selected_experts),row_experts:register_int::<B>(result.row_experts),state:result.state})
    }
    fn moe_dispatch_counts(state:&Self::MoeDispatchState,expert_prefix:&[usize]) -> Result<Vec<usize>,Self::MoeError> {B::moe_dispatch_counts(state,expert_prefix)}
    fn moe_dispatch_backward(state:Self::MoeDispatchState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::MoeError> {
        B::moe_dispatch_backward(state,resolve::<B>(gradient)).map(register_output::<B>)
    }
    fn moe_dispatch_weights_backward(state:Self::MoeDispatchState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::MoeError> {
        B::moe_dispatch_weights_backward(state,resolve::<B>(gradient)).map(register_output::<B>)
    }
    fn moe_combine(state:Self::MoeDispatchState,expert_values:FloatTensor<Self>,weights:FloatTensor<Self>,strategy:MoeCombineGradientStrategy)
        -> Result<FloatTensor<Self>,Self::MoeError> {B::moe_combine(state,resolve::<B>(expert_values),resolve::<B>(weights),strategy).map(register_output::<B>)}
    fn moe_combine_backward(state:Self::MoeDispatchState,expert_values:FloatTensor<Self>,weights:FloatTensor<Self>,gradient:FloatTensor<Self>,
        strategy:MoeCombineGradientStrategy,selection:MoeCombineSelection) -> Result<MoeCombineBackward<Self>,Self::MoeError> {
        let result=B::moe_combine_backward(state,resolve::<B>(expert_values),resolve::<B>(weights),resolve::<B>(gradient),strategy,selection)?;
        Ok(MoeCombineBackward {experts:result.experts.map(register_output::<B>),weights:result.weights.map(register_output::<B>)})
    }
}
impl<B:FusionBackend+MoeReceivedOps> MoeReceivedOps for Fusion<B> {
    type MoeReceivedState=B::MoeReceivedState;
    fn moe_received_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,
        options:MoeReceivedOptions,selection:MoeReceivedSelection) -> Result<(FloatTensor<Self>,Self::MoeReceivedState),Self::MoeError> {
        let ids=global_ids.client.clone().resolve_tensor_int::<B>(global_ids);
        let (output,state)=B::moe_received_forward(resolve::<B>(input),ids,resolve::<B>(gate),resolve::<B>(up),resolve::<B>(down),options,selection)?;
        Ok((register_output::<B>(output),state))
    }
    fn moe_received_backward(state:Self::MoeReceivedState,gradient:FloatTensor<Self>,selection:MoeReceivedSelection)
        -> Result<MoeReceivedBackward<Self>,Self::MoeError> {
        let result=B::moe_received_backward(state,resolve::<B>(gradient),selection)?;
        Ok(MoeReceivedBackward {input:result.input.map(register_output::<B>),gate:result.gate.map(register_output::<B>),
            up:result.up.map(register_output::<B>),down:result.down.map(register_output::<B>)})
    }
}
