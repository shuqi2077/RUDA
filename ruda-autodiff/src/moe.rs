use crate::{Autodiff,tensor::AutodiffTensor,checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},
    grads::Gradients,ops::{Backward,Ops,OpsKind,unary}};
use core::marker::PhantomData;
use ruda_tensor::{FloatDType,TensorMetadata,ops::FloatTensorOps,
    moe::{MoeOps,MoeOptions,MoeRouterWeightOptions,MoeBackward,MoeAutodiffError},tensor::{FloatTensor,IntTensor}};

/// Original opaque native state with an explicit first-order-only AD boundary.
#[derive(Clone,Debug)]
pub struct MoeAutodiffState<B:MoeOps> {
    native:B::MoeState,
    tracked_primals:bool,
}

#[derive(Debug)]
struct SelectedWeights<B:MoeOps>(PhantomData<B>);
impl<B:MoeOps> Backward<B,1> for SelectedWeights<B> {
    type State=(FloatTensor<B>,IntTensor<B>,MoeRouterWeightOptions);
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (logits,indices,options)=ops.state;
        unary::<B,_>(ops.parents,ops.node,grads,|gradient|B::moe_selected_weights_backward(logits,indices,B::float_cast(gradient,FloatDType::F32),options)
            .unwrap_or_else(|error|panic!("native selected router backward failed: {error:?}")));
    }
}
#[derive(Debug)]
struct RoutedExperts<B:MoeOps>(PhantomData<B>);
impl<B:MoeOps> Backward<B,5> for RoutedExperts<B> {
    type State=(B::MoeState,[FloatDType;5]);
    fn backward(self,ops:Ops<Self::State,5>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (state,dtypes)=ops.state;
        let gradient=B::float_cast(grads.consume::<B>(&ops.node),dtypes[0]);
        let result=B::moe_backward(state,gradient).unwrap_or_else(|error|panic!("native routed expert backward failed: {error:?}"));
        for ((parent,gradient),dtype) in ops.parents.into_iter().zip([result.input,result.logits,result.gate,result.up,result.down]).zip(dtypes) {
            if let Some(parent)=parent {grads.register::<B>(parent.id,B::float_cast(gradient,dtype));}
        }
    }
}
impl<B:MoeOps,S:CheckpointStrategy> MoeOps for Autodiff<B,S> {
    type MoeError=MoeAutodiffError<B::MoeError>;
    type MoeState=MoeAutodiffState<B>;
    fn moe_selected_weights(logits:FloatTensor<Self>,indices:IntTensor<Self>,options:MoeRouterWeightOptions) -> Result<FloatTensor<Self>,Self::MoeError> {
        let saved=logits.primitive.clone();
        let output=B::moe_selected_weights(logits.primitive,indices.clone(),options).map_err(MoeAutodiffError::Native)?;
        Ok(match SelectedWeights::<B>(PhantomData).prepare::<S>([logits.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((saved,indices,options),output),OpsKind::UnTracked(prep)=>prep.finish(output),
        })
    }
    fn moe_selected_weights_backward(logits:FloatTensor<Self>,indices:IntTensor<Self>,gradient:FloatTensor<Self>,options:MoeRouterWeightOptions)
        -> Result<FloatTensor<Self>,Self::MoeError> {
        if logits.is_tracked() || gradient.is_tracked() {return Err(MoeAutodiffError::HigherDerivativeUnsupported);}
        B::moe_selected_weights_backward(logits.primitive,indices,gradient.primitive,options).map(AutodiffTensor::new).map_err(MoeAutodiffError::Native)
    }
    fn moe_forward(input:FloatTensor<Self>,logits:FloatTensor<Self>,correction_bias:Option<FloatTensor<Self>>,
        gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,options:MoeOptions) -> Result<(FloatTensor<Self>,Self::MoeState),Self::MoeError> {
        let tracked_primals=input.is_tracked() || logits.is_tracked() || gate.is_tracked() || up.is_tracked() || down.is_tracked();
        let dtypes=[input.primitive.dtype().into(),logits.primitive.dtype().into(),gate.primitive.dtype().into(),up.primitive.dtype().into(),down.primitive.dtype().into()];
        let (output,state)=B::moe_forward(input.primitive,logits.primitive,correction_bias.map(|bias|bias.primitive),gate.primitive,up.primitive,down.primitive,options)
            .map_err(MoeAutodiffError::Native)?;
        let output=match RoutedExperts::<B>(PhantomData).prepare::<S>([input.node,logits.node,gate.node,up.node,down.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((state.clone(),dtypes),output),OpsKind::UnTracked(prep)=>prep.finish(output),
        };Ok((output,MoeAutodiffState {native:state,tracked_primals}))
    }
    fn moe_inference(input:FloatTensor<Self>,logits:FloatTensor<Self>,correction_bias:Option<FloatTensor<Self>>,
        gate:FloatTensor<Self>,up:FloatTensor<Self>,down:FloatTensor<Self>,options:MoeOptions) -> Result<FloatTensor<Self>,Self::MoeError> {
        if input.is_tracked() || logits.is_tracked() || gate.is_tracked() || up.is_tracked() || down.is_tracked() {
            return Self::moe_forward(input,logits,correction_bias,gate,up,down,options).map(|(output,_)|output);
        }
        B::moe_inference(input.primitive,logits.primitive,correction_bias.map(|bias|bias.primitive),gate.primitive,up.primitive,down.primitive,options)
            .map(AutodiffTensor::new).map_err(MoeAutodiffError::Native)
    }
    fn moe_route_indices(state:&Self::MoeState) -> IntTensor<Self> {B::moe_route_indices(&state.native)}
    fn moe_backward(state:Self::MoeState,gradient:FloatTensor<Self>) -> Result<MoeBackward<Self>,Self::MoeError> {
        if state.tracked_primals || gradient.is_tracked() {return Err(MoeAutodiffError::HigherDerivativeUnsupported);}
        let result=B::moe_backward(state.native,gradient.primitive).map_err(MoeAutodiffError::Native)?;
        Ok(MoeBackward {input:AutodiffTensor::new(result.input),logits:AutodiffTensor::new(result.logits),
            gate:AutodiffTensor::new(result.gate),up:AutodiffTensor::new(result.up),down:AutodiffTensor::new(result.down)})
    }
}
