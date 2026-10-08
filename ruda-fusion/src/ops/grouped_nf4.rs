use crate::{Fusion,FusionBackend};
use super::frozen_awq::register_output;
use ruda_tensor::{grouped_nf4::*,tensor::{FloatTensor,IntTensor}};

fn resolve_payload<B:FusionBackend>(payload:Nf4ExpertPayload<Fusion<B>>) -> Nf4ExpertPayload<B> {
    Nf4ExpertPayload {packed:payload.packed.client.clone().resolve_tensor_int::<B>(payload.packed),
        scales:payload.scales.client.clone().resolve_tensor_float::<B>(payload.scales),
        codebook:payload.codebook.client.clone().resolve_tensor_float::<B>(payload.codebook),options:payload.options}
}
impl<B:FusionBackend+FrozenNf4GroupedOps> FrozenNf4GroupedOps for Fusion<B> {
    type Nf4GroupedError=B::Nf4GroupedError;
    type Nf4GroupedState=B::Nf4GroupedState;
    fn frozen_nf4_grouped_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,payload:Nf4ExpertPayload<Self>)
        -> Result<(FloatTensor<Self>,Self::Nf4GroupedState),Self::Nf4GroupedError> {
        let input=input.client.clone().resolve_tensor_float::<B>(input);let ids=global_ids.client.clone().resolve_tensor_int::<B>(global_ids);
        B::frozen_nf4_grouped_forward(input,ids,resolve_payload::<B>(payload)).map(|(output,state)|(register_output::<B>(output),state))
    }
    fn frozen_nf4_grouped_input_backward(state:Self::Nf4GroupedState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::Nf4GroupedError> {
        let gradient=gradient.client.clone().resolve_tensor_float::<B>(gradient);
        B::frozen_nf4_grouped_input_backward(state,gradient).map(register_output::<B>)
    }
}
impl<B:FusionBackend+FrozenNf4SwiGluOps> FrozenNf4SwiGluOps for Fusion<B> {
    type Nf4SwiGluState=B::Nf4SwiGluState;
    fn frozen_nf4_swiglu_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:Nf4ExpertPayload<Self>,up:Nf4ExpertPayload<Self>,down:Nf4ExpertPayload<Self>,retain_input:bool)
        -> Result<(FloatTensor<Self>,Self::Nf4SwiGluState),Self::Nf4GroupedError> {
        let input=input.client.clone().resolve_tensor_float::<B>(input);let ids=global_ids.client.clone().resolve_tensor_int::<B>(global_ids);
        B::frozen_nf4_swiglu_forward(input,ids,resolve_payload::<B>(gate),resolve_payload::<B>(up),resolve_payload::<B>(down),retain_input)
            .map(|(output,state)|(register_output::<B>(output),state))
    }
    fn frozen_nf4_swiglu_input_backward(state:Self::Nf4SwiGluState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::Nf4GroupedError> {
        let gradient=gradient.client.clone().resolve_tensor_float::<B>(gradient);B::frozen_nf4_swiglu_input_backward(state,gradient).map(register_output::<B>)
    }
}
