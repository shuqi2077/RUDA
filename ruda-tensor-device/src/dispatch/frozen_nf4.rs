use crate::{DeviceBackend,DeviceRuntime,FloatElement,IntElement,element::BoolElement};
use ruda_tensor::{FloatDType,frozen_nf4::{FrozenNf4Ops,Nf4ProjectionOptions},ops::FloatTensorOps,tensor::{FloatTensor,IntTensor}};
use rublas::tensor_nf4::{Nf4Gemm,Nf4Layout,Nf4Error};

impl<R,F,I,BT> FrozenNf4Ops for DeviceBackend<R,F,I,BT>
    where R:DeviceRuntime,F:FloatElement,I:IntElement,BT:BoolElement {
    type Nf4Error=Nf4Error;
    fn frozen_nf4_forward(input:FloatTensor<Self>,packed:IntTensor<Self>,scales:FloatTensor<Self>,codebook:FloatTensor<Self>,
        bias:Option<FloatTensor<Self>>,options:Nf4ProjectionOptions) -> Result<FloatTensor<Self>,Self::Nf4Error> {
        let layout=Nf4Layout::new(options.input_features,options.output_features,options.block_size)?;
        Nf4Gemm::new(packed,scales,codebook,bias,layout)?.forward(input,options.tile_rows,options.use_tensor_core)
    }
    fn frozen_nf4_input_backward(gradient:FloatTensor<Self>,packed:IntTensor<Self>,scales:FloatTensor<Self>,codebook:FloatTensor<Self>,
        options:Nf4ProjectionOptions,activation_dtype:FloatDType) -> Result<FloatTensor<Self>,Self::Nf4Error> {
        let layout=Nf4Layout::new(options.input_features,options.output_features,options.block_size)?;
        let gradient=Self::float_cast(gradient,activation_dtype);
        let gradient=Nf4Gemm::new(packed,scales,codebook,None,layout)?.input_backward_f32(gradient,options.tile_rows,options.use_tensor_core)?;
        Ok(Self::float_cast(gradient,activation_dtype))
    }
}
