use crate::{Fusion,FusionBackend};
use super::frozen_awq::register_output;
use ruda_tensor::{packed_experts::*,grouped_nf4::Nf4ExpertPayload,tensor::{FloatTensor,IntTensor}};

fn resolve_payload<B:FusionBackend>(payload:PackedExpertPayload<Fusion<B>>) -> PackedExpertPayload<B> {
    match payload {
        PackedExpertPayload::Nf4(value)=>PackedExpertPayload::Nf4(Nf4ExpertPayload {packed:value.packed.client.clone().resolve_tensor_int::<B>(value.packed),
            scales:value.scales.client.clone().resolve_tensor_float::<B>(value.scales),codebook:value.codebook.client.clone().resolve_tensor_float::<B>(value.codebook),options:value.options}),
        PackedExpertPayload::Awq(value)=>PackedExpertPayload::Awq(AwqExpertPayload {qweight:value.qweight.client.clone().resolve_tensor_int::<B>(value.qweight),
            qzeros:value.qzeros.client.clone().resolve_tensor_int::<B>(value.qzeros),scales:value.scales.client.clone().resolve_tensor_float::<B>(value.scales),
            bias:value.bias.map(|value|value.client.clone().resolve_tensor_float::<B>(value)),options:value.options}),
    }
}
impl<B:FusionBackend+FrozenPackedExpertOps> FrozenPackedExpertOps for Fusion<B> {
    type PackedExpertError=B::PackedExpertError;
    type PackedProjectionState=B::PackedProjectionState;
    type PackedSwiGluState=B::PackedSwiGluState;
    fn packed_expert_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,payload:PackedExpertPayload<Self>) -> Result<(FloatTensor<Self>,Self::PackedProjectionState),Self::PackedExpertError> {
        let input=input.client.clone().resolve_tensor_float::<B>(input);let ids=global_ids.client.clone().resolve_tensor_int::<B>(global_ids);
        B::packed_expert_forward(input,ids,resolve_payload::<B>(payload)).map(|(output,state)|(register_output::<B>(output),state))
    }
    fn packed_expert_input_backward(state:Self::PackedProjectionState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::PackedExpertError> {
        let gradient=gradient.client.clone().resolve_tensor_float::<B>(gradient);B::packed_expert_input_backward(state,gradient).map(register_output::<B>)
    }
    fn packed_swiglu_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:PackedExpertPayload<Self>,up:PackedExpertPayload<Self>,down:PackedExpertPayload<Self>,retain_input:bool)
        -> Result<(FloatTensor<Self>,Self::PackedSwiGluState),Self::PackedExpertError> {
        let input=input.client.clone().resolve_tensor_float::<B>(input);let ids=global_ids.client.clone().resolve_tensor_int::<B>(global_ids);
        B::packed_swiglu_forward(input,ids,resolve_payload::<B>(gate),resolve_payload::<B>(up),resolve_payload::<B>(down),retain_input)
            .map(|(output,state)|(register_output::<B>(output),state))
    }
    fn packed_swiglu_input_backward(state:Self::PackedSwiGluState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::PackedExpertError> {
        let gradient=gradient.client.clone().resolve_tensor_float::<B>(gradient);B::packed_swiglu_input_backward(state,gradient).map(register_output::<B>)
    }
}
