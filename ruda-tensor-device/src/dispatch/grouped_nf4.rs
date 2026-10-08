use crate::{DeviceBackend,DeviceRuntime,FloatElement,IntElement,element::BoolElement};
use ruda_tensor::{FloatDType,grouped_nf4::*,ops::FloatTensorOps,tensor::{FloatTensor,IntTensor}};
use rudnn::moe::{MoeError,ReceivedExpertRows,Nf4ExpertProjection,Nf4ExpertError,Nf4SwiGluExperts,Nf4SwiGluCache,Nf4ExpertExecution};
use rublas::tensor_nf4::Nf4Layout;

/// Actual source row mapping, packed projection and original activation dtype.
#[derive(Clone,Debug)]
pub struct NativeNf4GroupedState<R:DeviceRuntime> {
    rows:ReceivedExpertRows<R>,projection:Nf4ExpertProjection<R>,options:Nf4GroupedOptions,dtype:FloatDType,
}
/// Actual original selected SwiGLU row mapping and optional first-order cache.
#[derive(Clone,Debug)]
pub struct NativeNf4SwiGluState<R:DeviceRuntime> {rows:ReceivedExpertRows<R>,cache:Option<Nf4SwiGluCache<R>>,dtype:FloatDType}
fn execution(options:Nf4GroupedOptions) -> Nf4ExpertExecution {
    Nf4ExpertExecution {tile_rows:options.projection.tile_rows,use_tensor_core:options.projection.use_tensor_core}
}
fn projection<R,F,I,BT>(payload:Nf4ExpertPayload<DeviceBackend<R,F,I,BT>>) -> Result<Nf4ExpertProjection<R>,Nf4ExpertError>
    where R:DeviceRuntime,F:FloatElement,I:IntElement,BT:BoolElement {
    let o=payload.options;
    if o.projection.tile_rows==0 {return Err(MoeError("NF4 expert decoded tile rows must be positive").into());}
    Ok(Nf4ExpertProjection::new(payload.packed,payload.scales,payload.codebook,o.experts,
        Nf4Layout::new(o.projection.input_features,o.projection.output_features,o.projection.block_size)?)?)
}
impl<R,F,I,BT> FrozenNf4GroupedOps for DeviceBackend<R,F,I,BT> where R:DeviceRuntime,F:FloatElement,I:IntElement,BT:BoolElement {
    type Nf4GroupedError=Nf4ExpertError;
    type Nf4GroupedState=NativeNf4GroupedState<R>;
    fn frozen_nf4_grouped_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,payload:Nf4ExpertPayload<Self>)
        -> Result<(FloatTensor<Self>,Self::Nf4GroupedState),Self::Nf4GroupedError> {
        let options=payload.options;let dtype=input.dtype.into();let projection=projection(payload)?;
        let rows=ReceivedExpertRows::new(input,global_ids,options.expert_start,options.experts)?;
        let output=projection.forward(rows.grouped(),options.projection.tile_rows,options.projection.use_tensor_core)?;
        Ok((rows.restore(output)?,NativeNf4GroupedState {rows,projection,options,dtype}))
    }
    fn frozen_nf4_grouped_input_backward(state:Self::Nf4GroupedState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::Nf4GroupedError> {
        let gradient=state.rows.sort_gradient(Self::float_cast(gradient,state.dtype))?;
        let gradient=state.projection.input_backward_f32(state.rows.grouped(),gradient,state.options.projection.tile_rows,state.options.projection.use_tensor_core)?;
        Ok(state.rows.restore(Self::float_cast(gradient,state.dtype))?)
    }
}
impl<R,F,I,BT> FrozenNf4SwiGluOps for DeviceBackend<R,F,I,BT> where R:DeviceRuntime,F:FloatElement,I:IntElement,BT:BoolElement {
    type Nf4SwiGluState=NativeNf4SwiGluState<R>;
    fn frozen_nf4_swiglu_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:Nf4ExpertPayload<Self>,up:Nf4ExpertPayload<Self>,down:Nf4ExpertPayload<Self>,retain_input:bool)
        -> Result<(FloatTensor<Self>,Self::Nf4SwiGluState),Self::Nf4GroupedError> {
        let options=[gate.options,up.options,down.options];
        if options.iter().any(|o|o.experts!=options[0].experts || o.expert_start!=options[0].expert_start) {
            return Err(MoeError("NF4 gate/up/down payloads must describe the same explicit global expert range").into());
        }
        let experts=Nf4SwiGluExperts::new(projection(gate)?,projection(up)?,projection(down)?)?;
        let dtype=input.dtype.into();let rows=ReceivedExpertRows::new(input,global_ids,options[0].expert_start,options[0].experts)?;
        let execution=options.map(execution);
        let (output,cache)=if retain_input {let (output,cache)=experts.forward_training(rows.grouped(),execution)?;(output,Some(cache))}
            else {(experts.forward(rows.grouped(),execution)?,None)};
        Ok((rows.restore(output)?,NativeNf4SwiGluState {rows,cache,dtype}))
    }
    fn frozen_nf4_swiglu_input_backward(state:Self::Nf4SwiGluState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::Nf4GroupedError> {
        let cache=state.cache.ok_or(MoeError("NF4 SwiGLU forward did not retain the input VJP cache"))?;
        let gradient=state.rows.sort_gradient(Self::float_cast(gradient,state.dtype))?;
        Ok(state.rows.restore(cache.input_backward(gradient)?)?)
    }
}
