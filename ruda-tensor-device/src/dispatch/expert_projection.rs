use crate::{DeviceBackend,DeviceRuntime,FloatElement,IntElement,element::BoolElement};
use ruda_tensor::{FloatDType,expert_projection::*,ops::FloatTensorOps,tensor::{FloatTensor,IntTensor}};
use rudnn::moe::{MoeError,ReceivedExpertRows,ExpertProjectionCache,expert_projection,swiglu_activation,swiglu_activation_backward,SwiGluActivationSelection};
use rublas::tensor_grouped::GroupedGradientSelection;
use super::moe::expert_strategy;

/// Actual original native grouped operands, valid private row COPY mapping and independent VJP policy.
#[derive(Clone,Debug)]
pub struct NativeExpertProjectionState<R:DeviceRuntime> {rows:ReceivedExpertRows<R>,cache:ExpertProjectionCache<R>,options:ExpertProjectionOptions,dtype:FloatDType}
impl<R,F,I,BT> ExpertProjectionOps for DeviceBackend<R,F,I,BT> where R:DeviceRuntime,F:FloatElement,I:IntElement,BT:BoolElement {
    type ExpertProjectionError=MoeError;
    type ExpertProjectionState=NativeExpertProjectionState<R>;
    fn expert_projection_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,weights:FloatTensor<Self>,options:ExpertProjectionOptions)
        -> Result<(FloatTensor<Self>,Self::ExpertProjectionState),Self::ExpertProjectionError> {
        if weights.meta.num_dims()!=3 {return Err(MoeError("native floating expert projection requires actual rank-three weights"));}
        let experts=weights.meta.shape()[0];let dtype=input.dtype.into();let rows=ReceivedExpertRows::new(input,global_ids,options.expert_start,experts)?;
        let (output,cache)=expert_projection(rows.grouped(),weights,expert_strategy(options.forward))?;
        Ok((rows.restore(output)?,NativeExpertProjectionState {rows,cache,options,dtype}))
    }
    fn expert_projection_backward(state:Self::ExpertProjectionState,gradient:FloatTensor<Self>,selection:ExpertProjectionSelection)
        -> Result<ExpertProjectionBackward<Self>,Self::ExpertProjectionError> {
        let gradient=state.rows.sort_gradient(Self::float_cast(gradient,state.dtype))?;
        let result=state.cache.backward(gradient,expert_strategy(state.options.backward),GroupedGradientSelection {input:selection.input,weights:selection.weights})?;
        Ok(ExpertProjectionBackward {input:result.dinput.map(|value|state.rows.restore(value)).transpose()?,weights:result.dweights})
    }
}
impl<R,F,I,BT> NativeSwiGluOps for DeviceBackend<R,F,I,BT> where R:DeviceRuntime,F:FloatElement,I:IntElement,BT:BoolElement {
    type SwiGluError=MoeError;
    fn native_swiglu(gate:FloatTensor<Self>,up:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::SwiGluError> {swiglu_activation(gate,up)}
    fn native_swiglu_backward(gate:FloatTensor<Self>,up:FloatTensor<Self>,gradient:FloatTensor<Self>,selection:NativeSwiGluSelection)
        -> Result<NativeSwiGluBackward<Self>,Self::SwiGluError> {
        let gradient=Self::float_cast(gradient,gate.dtype.into());
        let result=swiglu_activation_backward(gate,up,gradient,SwiGluActivationSelection {gate:selection.gate,up:selection.up})?;
        Ok(NativeSwiGluBackward {gate:result.gate,up:result.up})
    }
}
