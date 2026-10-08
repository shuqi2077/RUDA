use crate::{Autodiff,tensor::AutodiffTensor,checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},grads::Gradients,ops::{Backward,Ops,OpsKind}};
use ruda_tensor::{TensorMetadata,FloatDType,expert_projection::*,moe::MoeAutodiffError,ops::FloatTensorOps,tensor::{FloatTensor,IntTensor}};
use core::marker::PhantomData;

/// Original native row/cube state with explicit unsupported manual higher-order differentiation.
#[derive(Clone,Debug)]
pub struct ExpertProjectionAutodiffState<B:ExpertProjectionOps> {native:B::ExpertProjectionState,tracked_primals:bool}
#[derive(Debug)]
struct Projection<B:ExpertProjectionOps>(PhantomData<B>);
impl<B:ExpertProjectionOps> Backward<B,2> for Projection<B> {
    type State=(B::ExpertProjectionState,[FloatDType;2]);
    fn backward(self,ops:Ops<Self::State,2>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (state,dtypes)=ops.state;let gradient=B::float_cast(grads.consume::<B>(&ops.node),dtypes[0]);
        let selection=ExpertProjectionSelection {input:ops.parents[0].is_some(),weights:ops.parents[1].is_some()};
        let result=B::expert_projection_backward(state,gradient,selection).unwrap_or_else(|error|panic!("native trainable expert projection backward failed: {error:?}"));
        for ((parent,gradient),dtype) in ops.parents.into_iter().zip([result.input,result.weights]).zip(dtypes) {
            if let Some(parent)=parent {grads.register::<B>(parent.id,B::float_cast(gradient.expect("requested native expert projection derivative"),dtype));}}
    }
}
impl<B:ExpertProjectionOps,S:CheckpointStrategy> ExpertProjectionOps for Autodiff<B,S> {
    type ExpertProjectionError=MoeAutodiffError<B::ExpertProjectionError>;
    type ExpertProjectionState=ExpertProjectionAutodiffState<B>;
    fn expert_projection_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,weights:FloatTensor<Self>,options:ExpertProjectionOptions)
        -> Result<(FloatTensor<Self>,Self::ExpertProjectionState),Self::ExpertProjectionError> {
        let tracked_primals=input.is_tracked() || weights.is_tracked();let dtypes=[input.primitive.dtype().into(),weights.primitive.dtype().into()];
        let (output,native)=B::expert_projection_forward(input.primitive,global_ids,weights.primitive,options).map_err(MoeAutodiffError::Native)?;
        let output=match Projection::<B>(PhantomData).prepare::<S>([input.node,weights.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((native.clone(),dtypes),output),OpsKind::UnTracked(prep)=>prep.finish(output)};
        Ok((output,ExpertProjectionAutodiffState {native,tracked_primals}))
    }
    fn expert_projection_backward(state:Self::ExpertProjectionState,gradient:FloatTensor<Self>,selection:ExpertProjectionSelection)
        -> Result<ExpertProjectionBackward<Self>,Self::ExpertProjectionError> {
        if state.tracked_primals || gradient.is_tracked() {return Err(MoeAutodiffError::HigherDerivativeUnsupported);}
        let result=B::expert_projection_backward(state.native,gradient.primitive,selection).map_err(MoeAutodiffError::Native)?;
        Ok(ExpertProjectionBackward {input:result.input.map(AutodiffTensor::new),weights:result.weights.map(AutodiffTensor::new)})
    }
}
#[derive(Debug)]
struct SwiGlu<B:NativeSwiGluOps>(PhantomData<B>);
impl<B:NativeSwiGluOps> Backward<B,2> for SwiGlu<B> {
    type State=(FloatTensor<B>,FloatTensor<B>,[FloatDType;2]);
    fn backward(self,ops:Ops<Self::State,2>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (gate,up,dtypes)=ops.state;let gradient=B::float_cast(grads.consume::<B>(&ops.node),dtypes[0]);
        let selection=NativeSwiGluSelection {gate:ops.parents[0].is_some(),up:ops.parents[1].is_some()};
        let result=B::native_swiglu_backward(gate,up,gradient,selection).unwrap_or_else(|error|panic!("original native SwiGLU input backward failed: {error:?}"));
        for ((parent,gradient),dtype) in ops.parents.into_iter().zip([result.gate,result.up]).zip(dtypes) {
            if let Some(parent)=parent {grads.register::<B>(parent.id,B::float_cast(gradient.expect("requested original native SwiGLU derivative"),dtype));}}
    }
}
impl<B:NativeSwiGluOps,S:CheckpointStrategy> NativeSwiGluOps for Autodiff<B,S> {
    type SwiGluError=MoeAutodiffError<B::SwiGluError>;
    fn native_swiglu(gate:FloatTensor<Self>,up:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::SwiGluError> {
        let dtypes=[gate.primitive.dtype().into(),up.primitive.dtype().into()];let saved=(gate.primitive.clone(),up.primitive.clone(),dtypes);
        let output=B::native_swiglu(gate.primitive,up.primitive).map_err(MoeAutodiffError::Native)?;
        Ok(match SwiGlu::<B>(PhantomData).prepare::<S>([gate.node,up.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish(saved,output),OpsKind::UnTracked(prep)=>prep.finish(output)})
    }
    fn native_swiglu_backward(gate:FloatTensor<Self>,up:FloatTensor<Self>,gradient:FloatTensor<Self>,selection:NativeSwiGluSelection)
        -> Result<NativeSwiGluBackward<Self>,Self::SwiGluError> {
        if gate.is_tracked() || up.is_tracked() || gradient.is_tracked() {return Err(MoeAutodiffError::HigherDerivativeUnsupported);}
        let result=B::native_swiglu_backward(gate.primitive,up.primitive,gradient.primitive,selection).map_err(MoeAutodiffError::Native)?;
        Ok(NativeSwiGluBackward {gate:result.gate.map(AutodiffTensor::new),up:result.up.map(AutodiffTensor::new)})
    }
}
