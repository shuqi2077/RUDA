use crate::{Autodiff,tensor::AutodiffTensor,checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},
    grads::Gradients,ops::{Backward,Ops,OpsKind,unary}};
use ruda_tensor::{frozen_nf4::FrozenNf4Error,grouped_nf4::*,tensor::{FloatTensor,IntTensor}};
use core::marker::PhantomData;

/// Original native permutation/packed operands, with explicit manual first-order boundary.
#[derive(Clone,Debug)]
pub struct Nf4GroupedAutodiffState<B:FrozenNf4GroupedOps> {native:B::Nf4GroupedState,tracked_input:bool}
/// Original selected native SwiGLU cache; no detached reconstruction of the real input path.
#[derive(Clone,Debug)]
pub struct Nf4SwiGluAutodiffState<B:FrozenNf4SwiGluOps> {native:B::Nf4SwiGluState,tracked_input:bool}
fn frozen_payload<B:FrozenNf4GroupedOps,S:CheckpointStrategy>(payload:Nf4ExpertPayload<Autodiff<B,S>>)
    -> Result<Nf4ExpertPayload<B>,FrozenNf4Error<B::Nf4GroupedError>> {
    if payload.scales.is_tracked() || payload.codebook.is_tracked() {return Err(FrozenNf4Error::TrainableBase);}
    Ok(Nf4ExpertPayload {packed:payload.packed,scales:payload.scales.primitive,codebook:payload.codebook.primitive,options:payload.options})
}
#[derive(Debug)]
struct GroupedProjection<B:FrozenNf4GroupedOps>(PhantomData<B>);
impl<B:FrozenNf4GroupedOps> Backward<B,1> for GroupedProjection<B> {
    type State=B::Nf4GroupedState;
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        unary::<B,_>(ops.parents,ops.node,grads,|gradient|B::frozen_nf4_grouped_input_backward(ops.state,gradient)
            .unwrap_or_else(|error|panic!("native selected NF4 expert input backward failed: {error:?}")));
    }
}
#[derive(Debug)]
struct GroupedSwiGlu<B:FrozenNf4SwiGluOps>(PhantomData<B>);
impl<B:FrozenNf4SwiGluOps> Backward<B,1> for GroupedSwiGlu<B> {
    type State=B::Nf4SwiGluState;
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        unary::<B,_>(ops.parents,ops.node,grads,|gradient|B::frozen_nf4_swiglu_input_backward(ops.state,gradient)
            .unwrap_or_else(|error|panic!("native selected NF4 SwiGLU input backward failed: {error:?}")));
    }
}
impl<B:FrozenNf4GroupedOps,S:CheckpointStrategy> FrozenNf4GroupedOps for Autodiff<B,S> {
    type Nf4GroupedError=FrozenNf4Error<B::Nf4GroupedError>;
    type Nf4GroupedState=Nf4GroupedAutodiffState<B>;
    fn frozen_nf4_grouped_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,payload:Nf4ExpertPayload<Self>)
        -> Result<(FloatTensor<Self>,Self::Nf4GroupedState),Self::Nf4GroupedError> {
        let payload=frozen_payload(payload)?;let tracked_input=input.is_tracked();
        let (output,native)=B::frozen_nf4_grouped_forward(input.primitive,global_ids,payload).map_err(FrozenNf4Error::Native)?;
        let output=match GroupedProjection::<B>(PhantomData).prepare::<S>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish(native.clone(),output),OpsKind::UnTracked(prep)=>prep.finish(output)};
        Ok((output,Nf4GroupedAutodiffState {native,tracked_input}))
    }
    fn frozen_nf4_grouped_input_backward(state:Self::Nf4GroupedState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::Nf4GroupedError> {
        if state.tracked_input || gradient.is_tracked() {return Err(FrozenNf4Error::InputGradientNotDifferentiable);}
        B::frozen_nf4_grouped_input_backward(state.native,gradient.primitive).map(AutodiffTensor::new).map_err(FrozenNf4Error::Native)
    }
}
impl<B:FrozenNf4SwiGluOps,S:CheckpointStrategy> FrozenNf4SwiGluOps for Autodiff<B,S> {
    type Nf4SwiGluState=Nf4SwiGluAutodiffState<B>;
    fn frozen_nf4_swiglu_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:Nf4ExpertPayload<Self>,up:Nf4ExpertPayload<Self>,down:Nf4ExpertPayload<Self>,retain_input:bool)
        -> Result<(FloatTensor<Self>,Self::Nf4SwiGluState),Self::Nf4GroupedError> {
        let (gate,up,down)=(frozen_payload(gate)?,frozen_payload(up)?,frozen_payload(down)?);let tracked_input=input.is_tracked();
        let (output,native)=B::frozen_nf4_swiglu_forward(input.primitive,global_ids,gate,up,down,retain_input || tracked_input).map_err(FrozenNf4Error::Native)?;
        let output=match GroupedSwiGlu::<B>(PhantomData).prepare::<S>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish(native.clone(),output),OpsKind::UnTracked(prep)=>prep.finish(output)};
        Ok((output,Nf4SwiGluAutodiffState {native,tracked_input}))
    }
    fn frozen_nf4_swiglu_input_backward(state:Self::Nf4SwiGluState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::Nf4GroupedError> {
        if state.tracked_input || gradient.is_tracked() {return Err(FrozenNf4Error::InputGradientNotDifferentiable);}
        B::frozen_nf4_swiglu_input_backward(state.native,gradient.primitive).map(AutodiffTensor::new).map_err(FrozenNf4Error::Native)
    }
}
