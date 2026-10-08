use crate::{Autodiff,tensor::AutodiffTensor,checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},grads::Gradients,ops::{Backward,Ops,OpsKind,unary}};
use ruda_tensor::{packed_experts::*,grouped_nf4::Nf4ExpertPayload,tensor::{FloatTensor,IntTensor}};
use core::marker::PhantomData;

/// Original native projection with the actual first-order manual input boundary.
#[derive(Clone,Debug)]
pub struct PackedProjectionAutodiffState<B:FrozenPackedExpertOps> {native:B::PackedProjectionState,tracked_input:bool}
/// Original actual native packed chain cache with the same first-order manual boundary.
#[derive(Clone,Debug)]
pub struct PackedSwiGluAutodiffState<B:FrozenPackedExpertOps> {native:B::PackedSwiGluState,tracked_input:bool}
fn frozen_payload<B:FrozenPackedExpertOps,S:CheckpointStrategy>(payload:PackedExpertPayload<Autodiff<B,S>>) -> Result<PackedExpertPayload<B>,PackedExpertAutodiffError<B::PackedExpertError>> {
    Ok(match payload {
        PackedExpertPayload::Nf4(value)=>{
            if value.scales.is_tracked() || value.codebook.is_tracked() {return Err(PackedExpertAutodiffError::TrainableMetadata);}
            PackedExpertPayload::Nf4(Nf4ExpertPayload {packed:value.packed,scales:value.scales.primitive,codebook:value.codebook.primitive,options:value.options})
        },
        PackedExpertPayload::Awq(value)=>{
            if value.scales.is_tracked() || value.bias.as_ref().is_some_and(|value|value.is_tracked()) {return Err(PackedExpertAutodiffError::TrainableMetadata);}
            PackedExpertPayload::Awq(AwqExpertPayload {qweight:value.qweight,qzeros:value.qzeros,scales:value.scales.primitive,
                bias:value.bias.map(|value|value.primitive),options:value.options})
        },
    })
}
#[derive(Debug)]
struct PackedProjection<B:FrozenPackedExpertOps>(PhantomData<B>);
impl<B:FrozenPackedExpertOps> Backward<B,1> for PackedProjection<B> {
    type State=B::PackedProjectionState;
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        unary::<B,_>(ops.parents,ops.node,grads,|gradient|B::packed_expert_input_backward(ops.state,gradient)
            .unwrap_or_else(|error|panic!("native packed expert input backward failed: {error:?}")));
    }
}
#[derive(Debug)]
struct PackedSwiGlu<B:FrozenPackedExpertOps>(PhantomData<B>);
impl<B:FrozenPackedExpertOps> Backward<B,1> for PackedSwiGlu<B> {
    type State=B::PackedSwiGluState;
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        unary::<B,_>(ops.parents,ops.node,grads,|gradient|B::packed_swiglu_input_backward(ops.state,gradient)
            .unwrap_or_else(|error|panic!("native mixed packed SwiGLU input backward failed: {error:?}")));
    }
}
impl<B:FrozenPackedExpertOps,S:CheckpointStrategy> FrozenPackedExpertOps for Autodiff<B,S> {
    type PackedExpertError=PackedExpertAutodiffError<B::PackedExpertError>;
    type PackedProjectionState=PackedProjectionAutodiffState<B>;
    type PackedSwiGluState=PackedSwiGluAutodiffState<B>;
    fn packed_expert_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,payload:PackedExpertPayload<Self>) -> Result<(FloatTensor<Self>,Self::PackedProjectionState),Self::PackedExpertError> {
        let payload=frozen_payload(payload)?;let tracked_input=input.is_tracked();
        let (output,native)=B::packed_expert_forward(input.primitive,global_ids,payload).map_err(PackedExpertAutodiffError::Native)?;
        let output=match PackedProjection::<B>(PhantomData).prepare::<S>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish(native.clone(),output),OpsKind::UnTracked(prep)=>prep.finish(output)};
        Ok((output,PackedProjectionAutodiffState {native,tracked_input}))
    }
    fn packed_expert_input_backward(state:Self::PackedProjectionState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::PackedExpertError> {
        if state.tracked_input || gradient.is_tracked() {return Err(PackedExpertAutodiffError::HigherDerivativeUnsupported);}
        B::packed_expert_input_backward(state.native,gradient.primitive).map(AutodiffTensor::new).map_err(PackedExpertAutodiffError::Native)
    }
    fn packed_swiglu_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:PackedExpertPayload<Self>,up:PackedExpertPayload<Self>,down:PackedExpertPayload<Self>,retain_input:bool)
        -> Result<(FloatTensor<Self>,Self::PackedSwiGluState),Self::PackedExpertError> {
        let (gate,up,down)=(frozen_payload(gate)?,frozen_payload(up)?,frozen_payload(down)?);let tracked_input=input.is_tracked();
        let (output,native)=B::packed_swiglu_forward(input.primitive,global_ids,gate,up,down,retain_input || tracked_input).map_err(PackedExpertAutodiffError::Native)?;
        let output=match PackedSwiGlu::<B>(PhantomData).prepare::<S>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish(native.clone(),output),OpsKind::UnTracked(prep)=>prep.finish(output)};
        Ok((output,PackedSwiGluAutodiffState {native,tracked_input}))
    }
    fn packed_swiglu_input_backward(state:Self::PackedSwiGluState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::PackedExpertError> {
        if state.tracked_input || gradient.is_tracked() {return Err(PackedExpertAutodiffError::HigherDerivativeUnsupported);}
        B::packed_swiglu_input_backward(state.native,gradient.primitive).map(AutodiffTensor::new).map_err(PackedExpertAutodiffError::Native)
    }
}
