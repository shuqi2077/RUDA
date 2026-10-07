use crate::{Fusion,FusionBackend};
use super::frozen_awq::register_output;
use ruda_tensor::{FloatDType,frozen_nf4::{FrozenNf4Ops,Nf4ProjectionOptions},tensor::{FloatTensor,IntTensor}};

impl<B:FusionBackend+FrozenNf4Ops> FrozenNf4Ops for Fusion<B> {
    type Nf4Error=B::Nf4Error;
    fn frozen_nf4_forward(input:FloatTensor<Self>,packed:IntTensor<Self>,scales:FloatTensor<Self>,codebook:FloatTensor<Self>,
        bias:Option<FloatTensor<Self>>,options:Nf4ProjectionOptions) -> Result<FloatTensor<Self>,Self::Nf4Error> {
        let input=input.client.clone().resolve_tensor_float::<B>(input);
        let packed=packed.client.clone().resolve_tensor_int::<B>(packed);
        let scales=scales.client.clone().resolve_tensor_float::<B>(scales);
        let book=codebook.client.clone().resolve_tensor_float::<B>(codebook);
        let bias=bias.map(|value|value.client.clone().resolve_tensor_float::<B>(value));
        B::frozen_nf4_forward(input,packed,scales,book,bias,options).map(register_output::<B>)
    }
    fn frozen_nf4_input_backward(gradient:FloatTensor<Self>,packed:IntTensor<Self>,scales:FloatTensor<Self>,codebook:FloatTensor<Self>,
        options:Nf4ProjectionOptions,activation_dtype:FloatDType) -> Result<FloatTensor<Self>,Self::Nf4Error> {
        let gradient=gradient.client.clone().resolve_tensor_float::<B>(gradient);
        let packed=packed.client.clone().resolve_tensor_int::<B>(packed);
        let scales=scales.client.clone().resolve_tensor_float::<B>(scales);
        let book=codebook.client.clone().resolve_tensor_float::<B>(codebook);
        B::frozen_nf4_input_backward(gradient,packed,scales,book,options,activation_dtype).map(register_output::<B>)
    }
}
