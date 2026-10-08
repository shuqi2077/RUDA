use crate::{Fusion,FusionBackend};
use super::frozen_awq::register_output;
use ruda_tensor::{expert_projection::*,tensor::{FloatTensor,IntTensor}};

impl<B:FusionBackend+ExpertProjectionOps> ExpertProjectionOps for Fusion<B> {
    type ExpertProjectionError=B::ExpertProjectionError;
    type ExpertProjectionState=B::ExpertProjectionState;
    fn expert_projection_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,weights:FloatTensor<Self>,options:ExpertProjectionOptions)
        -> Result<(FloatTensor<Self>,Self::ExpertProjectionState),Self::ExpertProjectionError> {
        let input=input.client.clone().resolve_tensor_float::<B>(input);let ids=global_ids.client.clone().resolve_tensor_int::<B>(global_ids);
        let weights=weights.client.clone().resolve_tensor_float::<B>(weights);
        B::expert_projection_forward(input,ids,weights,options).map(|(output,state)|(register_output::<B>(output),state))
    }
    fn expert_projection_backward(state:Self::ExpertProjectionState,gradient:FloatTensor<Self>,selection:ExpertProjectionSelection)
        -> Result<ExpertProjectionBackward<Self>,Self::ExpertProjectionError> {
        let gradient=gradient.client.clone().resolve_tensor_float::<B>(gradient);let result=B::expert_projection_backward(state,gradient,selection)?;
        Ok(ExpertProjectionBackward {input:result.input.map(register_output::<B>),weights:result.weights.map(register_output::<B>)})
    }
}
impl<B:FusionBackend+NativeSwiGluOps> NativeSwiGluOps for Fusion<B> {
    type SwiGluError=B::SwiGluError;
    fn native_swiglu(gate:FloatTensor<Self>,up:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::SwiGluError> {
        let gate=gate.client.clone().resolve_tensor_float::<B>(gate);let up=up.client.clone().resolve_tensor_float::<B>(up);
        B::native_swiglu(gate,up).map(register_output::<B>)
    }
    fn native_swiglu_backward(gate:FloatTensor<Self>,up:FloatTensor<Self>,gradient:FloatTensor<Self>,selection:NativeSwiGluSelection)
        -> Result<NativeSwiGluBackward<Self>,Self::SwiGluError> {
        let gate=gate.client.clone().resolve_tensor_float::<B>(gate);let up=up.client.clone().resolve_tensor_float::<B>(up);
        let gradient=gradient.client.clone().resolve_tensor_float::<B>(gradient);let result=B::native_swiglu_backward(gate,up,gradient,selection)?;
        Ok(NativeSwiGluBackward {gate:result.gate.map(register_output::<B>),up:result.up.map(register_output::<B>)})
    }
}
