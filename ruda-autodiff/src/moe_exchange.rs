use crate::{Autodiff,tensor::AutodiffTensor,checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},
    grads::Gradients,ops::{Backward,Ops,OpsKind,unary}};
use alloc::vec::Vec;
use core::marker::PhantomData;
use ruda_tensor::{TensorMetadata,FloatDType,ops::FloatTensorOps,tensor::{FloatTensor,IntTensor},moe::{MoeOptions,MoeCombineGradientStrategy,MoeAutodiffError},moe_exchange::*};

/// Original native source mappings with an explicit first-order manual VJP boundary.
#[derive(Debug,Clone)]
pub struct MoeAutodiffDispatchState<B:MoeDispatchOps> {native:B::MoeDispatchState,tracked_primals:bool}
/// Original received expert cache, not a detached reconstruction of source activations.
#[derive(Debug,Clone)]
pub struct MoeAutodiffReceivedState<B:MoeReceivedOps> {native:B::MoeReceivedState,tracked_primals:bool}
#[derive(Debug)]
struct DispatchCopy<B:MoeDispatchOps>(PhantomData<B>);
impl<B:MoeDispatchOps> Backward<B,1> for DispatchCopy<B> {
    type State=(B::MoeDispatchState,FloatDType);
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        unary::<B,_>(ops.parents,ops.node,grads,|gradient|B::moe_dispatch_backward(ops.state.0,B::float_cast(gradient,ops.state.1))
            .unwrap_or_else(|error|panic!("native dispatch COPY backward failed: {error:?}")));
    }
}
#[derive(Debug)]
struct DispatchWeights<B:MoeDispatchOps>(PhantomData<B>);
impl<B:MoeDispatchOps> Backward<B,1> for DispatchWeights<B> {
    type State=B::MoeDispatchState;
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        unary::<B,_>(ops.parents,ops.node,grads,|gradient|B::moe_dispatch_weights_backward(ops.state,B::float_cast(gradient,FloatDType::F32))
            .unwrap_or_else(|error|panic!("native dispatched routing-weight backward failed: {error:?}")));
    }
}
#[derive(Debug)]
struct Combine<B:MoeDispatchOps>(PhantomData<B>);
impl<B:MoeDispatchOps> Backward<B,2> for Combine<B> {
    type State=(B::MoeDispatchState,FloatTensor<B>,FloatTensor<B>,[FloatDType;2],MoeCombineGradientStrategy);
    fn backward(self,ops:Ops<Self::State,2>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (state,expert_values,weights,dtypes,strategy)=ops.state;let selection=MoeCombineSelection {experts:ops.parents[0].is_some(),weights:ops.parents[1].is_some()};
        let gradient=B::float_cast(grads.consume::<B>(&ops.node),dtypes[0]);
        let result=B::moe_combine_backward(state,expert_values,weights,gradient,strategy,selection)
            .unwrap_or_else(|error|panic!("native source combine backward failed: {error:?}"));
        for ((parent,gradient),dtype) in ops.parents.into_iter().zip([result.experts,result.weights]).zip(dtypes) {
            if let Some(parent)=parent {grads.register::<B>(parent.id,B::float_cast(gradient.expect("requested native combine derivative"),dtype));}}
    }
}
impl<B:MoeDispatchOps,S:CheckpointStrategy> MoeDispatchOps for Autodiff<B,S> {
    type MoeDispatchState=MoeAutodiffDispatchState<B>;
    fn moe_dispatch(input:FloatTensor<Self>,logits:FloatTensor<Self>,bias:Option<FloatTensor<Self>>,options:MoeOptions) -> Result<MoeDispatched<Self>,Self::MoeError> {
        let tracked_primals=input.is_tracked() || logits.is_tracked();let dtype=input.primitive.dtype().into();
        let result=B::moe_dispatch(input.primitive,logits.primitive,bias.map(|bias|bias.primitive),options).map_err(MoeAutodiffError::Native)?;
        let values=match DispatchCopy::<B>(PhantomData).prepare::<S>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((result.state.clone(),dtype),result.values),OpsKind::UnTracked(prep)=>prep.finish(result.values)};
        let weights=match DispatchWeights::<B>(PhantomData).prepare::<S>([logits.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish(result.state.clone(),result.weights),OpsKind::UnTracked(prep)=>prep.finish(result.weights)};
        Ok(MoeDispatched {values,weights,selected_experts:result.selected_experts,row_experts:result.row_experts,
            state:MoeAutodiffDispatchState {native:result.state,tracked_primals}})
    }
    fn moe_dispatch_counts(state:&Self::MoeDispatchState,expert_prefix:&[usize]) -> Result<Vec<usize>,Self::MoeError> {
        B::moe_dispatch_counts(&state.native,expert_prefix).map_err(MoeAutodiffError::Native)
    }
    fn moe_dispatch_backward(state:Self::MoeDispatchState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::MoeError> {
        if state.tracked_primals || gradient.is_tracked() {return Err(MoeAutodiffError::HigherDerivativeUnsupported);}
        B::moe_dispatch_backward(state.native,gradient.primitive).map(AutodiffTensor::new).map_err(MoeAutodiffError::Native)
    }
    fn moe_dispatch_weights_backward(state:Self::MoeDispatchState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::MoeError> {
        if state.tracked_primals || gradient.is_tracked() {return Err(MoeAutodiffError::HigherDerivativeUnsupported);}
        B::moe_dispatch_weights_backward(state.native,gradient.primitive).map(AutodiffTensor::new).map_err(MoeAutodiffError::Native)
    }
    fn moe_combine(state:Self::MoeDispatchState,expert_values:FloatTensor<Self>,weights:FloatTensor<Self>,strategy:MoeCombineGradientStrategy)
        -> Result<FloatTensor<Self>,Self::MoeError> {
        let dtypes=[expert_values.primitive.dtype().into(),weights.primitive.dtype().into()];let native=state.native;
        let saved=(native.clone(),expert_values.primitive.clone(),weights.primitive.clone(),dtypes,strategy);
        let output=B::moe_combine(native,expert_values.primitive,weights.primitive,strategy).map_err(MoeAutodiffError::Native)?;
        Ok(match Combine::<B>(PhantomData).prepare::<S>([expert_values.node,weights.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish(saved,output),OpsKind::UnTracked(prep)=>prep.finish(output)})
    }
    fn moe_combine_backward(state:Self::MoeDispatchState,expert_values:FloatTensor<Self>,weights:FloatTensor<Self>,gradient:FloatTensor<Self>,
        strategy:MoeCombineGradientStrategy,selection:MoeCombineSelection) -> Result<MoeCombineBackward<Self>,Self::MoeError> {
        if state.tracked_primals || expert_values.is_tracked() || weights.is_tracked() || gradient.is_tracked() {return Err(MoeAutodiffError::HigherDerivativeUnsupported);}
        let result=B::moe_combine_backward(state.native,expert_values.primitive,weights.primitive,gradient.primitive,strategy,selection).map_err(MoeAutodiffError::Native)?;
        Ok(MoeCombineBackward {experts:result.experts.map(AutodiffTensor::new),weights:result.weights.map(AutodiffTensor::new)})
    }
}
#[derive(Debug)]
struct Received<B:MoeReceivedOps>(PhantomData<B>);
impl<B:MoeReceivedOps> Backward<B,4> for Received<B> {
    type State=(B::MoeReceivedState,[FloatDType;4]);
    fn backward(self,ops:Ops<Self::State,4>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (state,dtypes)=ops.state;let mask=ops.parents.each_ref().map(Option::is_some);let gradient=B::float_cast(grads.consume::<B>(&ops.node),dtypes[0]);
        let result=B::moe_received_backward(state,gradient,MoeReceivedSelection {input:mask[0],gate:mask[1],up:mask[2],down:mask[3]})
            .unwrap_or_else(|error|panic!("native received expert backward failed: {error:?}"));
        for ((parent,gradient),dtype) in ops.parents.into_iter().zip([result.input,result.gate,result.up,result.down]).zip(dtypes) {
            if let Some(parent)=parent {grads.register::<B>(parent.id,B::float_cast(gradient.expect("requested native received expert derivative"),dtype));}}
    }
}
impl<B:MoeReceivedOps,S:CheckpointStrategy> MoeReceivedOps for Autodiff<B,S> {
    type MoeReceivedState=MoeAutodiffReceivedState<B>;
    fn moe_received_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,
        options:MoeReceivedOptions,selection:MoeReceivedSelection) -> Result<(FloatTensor<Self>,Self::MoeReceivedState),Self::MoeError> {
        let actual=MoeReceivedSelection {input:input.is_tracked(),gate:gate.is_tracked(),up:up.is_tracked(),down:down.is_tracked()};
        let tracked_primals=actual.input || actual.gate || actual.up || actual.down;
        let dtypes=[input.primitive.dtype().into(),gate.primitive.dtype().into(),up.primitive.dtype().into(),down.primitive.dtype().into()];
        let (output,state)=B::moe_received_forward(input.primitive,global_ids,gate.primitive,up.primitive,down.primitive,options,
            if tracked_primals {actual} else {selection}).map_err(MoeAutodiffError::Native)?;
        let output=match Received::<B>(PhantomData).prepare::<S>([input.node,gate.node,up.node,down.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((state.clone(),dtypes),output),OpsKind::UnTracked(prep)=>prep.finish(output)};
        Ok((output,MoeAutodiffReceivedState {native:state,tracked_primals}))
    }
    fn moe_received_backward(state:Self::MoeReceivedState,gradient:FloatTensor<Self>,selection:MoeReceivedSelection)
        -> Result<MoeReceivedBackward<Self>,Self::MoeError> {
        if state.tracked_primals || gradient.is_tracked() {return Err(MoeAutodiffError::HigherDerivativeUnsupported);}
        let result=B::moe_received_backward(state.native,gradient.primitive,selection).map_err(MoeAutodiffError::Native)?;
        Ok(MoeReceivedBackward {input:result.input.map(AutodiffTensor::new),gate:result.gate.map(AutodiffTensor::new),
            up:result.up.map(AutodiffTensor::new),down:result.down.map(AutodiffTensor::new)})
    }
}
