use super::{MoeError,elements,empty,float_tensor,same_device,kernels};
use ruda_kernel::{dsl::{Runtime,calculate_ruda_count_elemwise,prelude::RudaDim},tensor::{RudaTensor,contiguous::into_contiguous}};

/// Requested actual native storage-rounded SwiGLU input derivatives.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct SwiGluActivationSelection {pub gate:bool,pub up:bool}
/// Only actual requested original native activation derivatives, without placeholder outputs.
#[derive(Debug)]
pub struct SwiGluActivationBackward<R:Runtime> {pub gate:Option<RudaTensor<R>>,pub up:Option<RudaTensor<R>>}
fn validate<R:Runtime>(gate:&RudaTensor<R>,up:&RudaTensor<R>) -> Result<usize,MoeError> {
    float_tensor(gate)?;float_tensor(up)?;same_device(gate,up)?;
    if gate.meta.shape()!=up.meta.shape() || gate.dtype!=up.dtype || !gate.client.same_execution_queue(&up.client) {
        return Err(MoeError("native SwiGLU requires matching original floating geometry/storage/device/queue"));
    }
    elements(gate.meta.shape())
}
/// Original ruDNN SwiGLU: round stored SiLU before multiplying by up, then round output storage.
/// Source input handles are never overwritten, so actual primals remain available for AD.
pub fn swiglu_activation<R:Runtime>(gate:RudaTensor<R>,up:RudaTensor<R>) -> Result<RudaTensor<R>,MoeError> {
    let size=validate(&gate,&up)?;let output=empty(&gate,gate.meta.shape().clone(),gate.dtype);
    if size!=0 {let gate=into_contiguous(gate);let dim=RudaDim::new(gate.client.properties(),size);
        kernels::experts::swiglu_out::launch::<R>(&gate.client,calculate_ruda_count_elemwise(&gate.client,size,dim),dim,
            gate.clone().into_array_arg(),into_contiguous(up).into_array_arg(),output.clone().into_array_arg(),gate.dtype.into());}
    Ok(output)
}
/// Original first-order storage-rounded input VJPs, with only actual requested outputs.
pub fn swiglu_activation_backward<R:Runtime>(gate:RudaTensor<R>,up:RudaTensor<R>,gradient:RudaTensor<R>,selection:SwiGluActivationSelection)
    -> Result<SwiGluActivationBackward<R>,MoeError> {
    let size=validate(&gate,&up)?;validate(&gate,&gradient)?;
    let dg=selection.gate.then(||empty(&gate,gate.meta.shape().clone(),gate.dtype));
    let du=selection.up.then(||empty(&up,up.meta.shape().clone(),up.dtype));
    if size!=0 {let gate=into_contiguous(gate);let up=into_contiguous(up);let gradient=into_contiguous(gradient);let dim=RudaDim::new(gate.client.properties(),size);
        match (&dg,&du) {
            (Some(dg),Some(du))=>kernels::experts::swiglu_backward::launch::<R>(&gate.client,calculate_ruda_count_elemwise(&gate.client,size,dim),dim,
                gate.clone().into_array_arg(),up.into_array_arg(),gradient.into_array_arg(),dg.clone().into_array_arg(),du.clone().into_array_arg(),gate.dtype.into()),
            (Some(dg),None)=>kernels::experts::swiglu_backward_gate::launch::<R>(&gate.client,calculate_ruda_count_elemwise(&gate.client,size,dim),dim,
                gate.clone().into_array_arg(),up.into_array_arg(),gradient.into_array_arg(),dg.clone().into_array_arg(),gate.dtype.into()),
            (None,Some(du))=>kernels::experts::swiglu_backward_up::launch::<R>(&gate.client,calculate_ruda_count_elemwise(&gate.client,size,dim),dim,
                gate.clone().into_array_arg(),gradient.into_array_arg(),du.clone().into_array_arg(),gate.dtype.into()),
            (None,None)=>{},
        }
    }
    Ok(SwiGluActivationBackward {gate:dg,up:du})
}
