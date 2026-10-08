use crate::{Fusion,FusionBackend,get_client,ops::NoOp,stream::OperationStreams};
use super::frozen_awq::register_output;
use ruda_tensor::{TensorMetadata,moe::{MoeOps,MoeOptions,MoeRouterWeightOptions,MoeBackward,MoeGradientSelection,MoeBackwardSelected},tensor::{FloatTensor,IntTensor},
    graph::{InitOperationIr,OperationIr,OperationOutput}};

fn resolve<B:FusionBackend>(value:FloatTensor<Fusion<B>>) -> FloatTensor<B> {value.client.clone().resolve_tensor_float::<B>(value)}
impl<B:FusionBackend+MoeOps> MoeOps for Fusion<B> {
    type MoeError=B::MoeError;
    type MoeState=B::MoeState;
    fn moe_selected_weights(logits:FloatTensor<Self>,indices:IntTensor<Self>,options:MoeRouterWeightOptions) -> Result<FloatTensor<Self>,Self::MoeError> {
        let indices=indices.client.clone().resolve_tensor_int::<B>(indices);
        B::moe_selected_weights(resolve::<B>(logits),indices,options).map(register_output::<B>)
    }
    fn moe_selected_weights_backward(logits:FloatTensor<Self>,indices:IntTensor<Self>,gradient:FloatTensor<Self>,options:MoeRouterWeightOptions)
        -> Result<FloatTensor<Self>,Self::MoeError> {
        let indices=indices.client.clone().resolve_tensor_int::<B>(indices);
        B::moe_selected_weights_backward(resolve::<B>(logits),indices,resolve::<B>(gradient),options).map(register_output::<B>)
    }
    fn moe_forward(input:FloatTensor<Self>,logits:FloatTensor<Self>,correction_bias:Option<FloatTensor<Self>>,
        gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,options:MoeOptions) -> Result<(FloatTensor<Self>,Self::MoeState),Self::MoeError> {
        let (output,state)=B::moe_forward(resolve::<B>(input),resolve::<B>(logits),correction_bias.map(resolve::<B>),resolve::<B>(gate),resolve::<B>(up),resolve::<B>(down),options)?;
        Ok((register_output::<B>(output),state))
    }
    fn moe_forward_selected(input:FloatTensor<Self>,logits:FloatTensor<Self>,correction_bias:Option<FloatTensor<Self>>,
        gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,options:MoeOptions,selection:MoeGradientSelection)
        -> Result<(FloatTensor<Self>,Self::MoeState),Self::MoeError> {
        let (output,state)=B::moe_forward_selected(resolve::<B>(input),resolve::<B>(logits),correction_bias.map(resolve::<B>),
            resolve::<B>(gate),resolve::<B>(up),resolve::<B>(down),options,selection)?;Ok((register_output::<B>(output),state))
    }
    fn moe_inference(input:FloatTensor<Self>,logits:FloatTensor<Self>,correction_bias:Option<FloatTensor<Self>>,
        gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,options:MoeOptions) -> Result<FloatTensor<Self>,Self::MoeError> {
        B::moe_inference(resolve::<B>(input),resolve::<B>(logits),correction_bias.map(resolve::<B>),resolve::<B>(gate),resolve::<B>(up),resolve::<B>(down),options).map(register_output::<B>)
    }
    fn moe_route_indices(state:&Self::MoeState) -> IntTensor<Self> {
        let output=B::moe_route_indices(state);let client=get_client::<B>(&B::int_device(&output));
        let desc=InitOperationIr::create(output.shape(),output.dtype(),||client.register_tensor_handle(B::int_tensor_handle(output)));
        client.register(OperationStreams::default(),OperationIr::Init(desc),NoOp::<B>::new()).output()
    }
    fn moe_backward(state:Self::MoeState,gradient:FloatTensor<Self>) -> Result<MoeBackward<Self>,Self::MoeError> {
        let result=B::moe_backward(state,resolve::<B>(gradient))?;
        Ok(MoeBackward {input:register_output::<B>(result.input),logits:register_output::<B>(result.logits),
            gate:register_output::<B>(result.gate),up:register_output::<B>(result.up),down:register_output::<B>(result.down)})
    }
    fn moe_backward_selected(state:Self::MoeState,gradient:FloatTensor<Self>,selection:MoeGradientSelection)
        -> Result<MoeBackwardSelected<Self>,Self::MoeError> {
        let result=B::moe_backward_selected(state,resolve::<B>(gradient),selection)?;
        Ok(MoeBackwardSelected {input:result.input.map(register_output::<B>),logits:result.logits.map(register_output::<B>),
            gate:result.gate.map(register_output::<B>),up:result.up.map(register_output::<B>),down:result.down.map(register_output::<B>)})
    }
}
