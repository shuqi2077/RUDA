use crate::{Autodiff,tensor::AutodiffTensor,checkpoint::{base::Checkpointer,strategy::CheckpointStrategy},
    grads::Gradients,ops::{Backward,Ops,OpsKind,unary}};
use core::marker::PhantomData;
use ruda_tensor::{FloatDType,TensorMetadata,frozen_nf4::{FrozenNf4Ops,FrozenNf4Error,Nf4ProjectionOptions},tensor::{FloatTensor,IntTensor}};

#[derive(Debug)]
struct PackedNf4<B:FrozenNf4Ops>(PhantomData<B>);
impl<B:FrozenNf4Ops> Backward<B,1> for PackedNf4<B> {
    type State=(IntTensor<B>,FloatTensor<B>,FloatTensor<B>,Nf4ProjectionOptions,FloatDType);
    fn backward(self,ops:Ops<Self::State,1>,grads:&mut Gradients,_:&mut Checkpointer) {
        let (packed,scales,book,options,dtype)=ops.state;
        unary::<B,_>(ops.parents,ops.node,grads,|gradient|B::frozen_nf4_input_backward(gradient,packed,scales,book,options,dtype)
            .unwrap_or_else(|error|panic!("native frozen NF4 backward failed: {error:?}")));
    }
}
impl<B:FrozenNf4Ops,S:CheckpointStrategy> FrozenNf4Ops for Autodiff<B,S> {
    type Nf4Error=FrozenNf4Error<B::Nf4Error>;
    fn frozen_nf4_forward(input:FloatTensor<Self>,packed:IntTensor<Self>,scales:FloatTensor<Self>,codebook:FloatTensor<Self>,
        bias:Option<FloatTensor<Self>>,options:Nf4ProjectionOptions) -> Result<FloatTensor<Self>,Self::Nf4Error> {
        if scales.is_tracked() || codebook.is_tracked() || bias.as_ref().is_some_and(|value|value.is_tracked()) {return Err(FrozenNf4Error::TrainableBase);}
        let dtype=input.primitive.dtype();
        let state=(packed.clone(),scales.primitive.clone(),codebook.primitive.clone(),options);
        let output=B::frozen_nf4_forward(input.primitive,packed,scales.primitive,codebook.primitive,bias.map(|value|value.primitive),options)
            .map_err(FrozenNf4Error::Native)?;
        Ok(match PackedNf4::<B>(PhantomData).prepare::<S>([input.node]).compute_bound().stateful() {
            OpsKind::Tracked(prep)=>prep.finish((state.0,state.1,state.2,state.3,dtype.into()),output),OpsKind::UnTracked(prep)=>prep.finish(output),
        })
    }
    fn frozen_nf4_input_backward(gradient:FloatTensor<Self>,packed:IntTensor<Self>,scales:FloatTensor<Self>,codebook:FloatTensor<Self>,
        options:Nf4ProjectionOptions,activation_dtype:FloatDType) -> Result<FloatTensor<Self>,Self::Nf4Error> {
        if scales.is_tracked() || codebook.is_tracked() {return Err(FrozenNf4Error::TrainableBase);}
        if gradient.is_tracked() {return Err(FrozenNf4Error::InputGradientNotDifferentiable);}
        B::frozen_nf4_input_backward(gradient.primitive,packed,scales.primitive,codebook.primitive,options,activation_dtype)
            .map(AutodiffTensor::new).map_err(FrozenNf4Error::Native)
    }
}
