use crate::{DeviceBackend,DeviceRuntime,FloatElement,IntElement,element::BoolElement};
use ruda_tensor::{FloatDType,packed_experts::*,ops::FloatTensorOps,tensor::{FloatTensor,IntTensor}};
use rudnn::moe::{MoeError,ReceivedExpertRows,PackedExpertError,PackedExpertProjection,PackedSwiGluExperts,PackedSwiGluCache,Nf4ExpertProjection,Nf4ExpertExecution};
use rublas::{tensor_int4::AwqGroupedGemm,tensor_nf4::Nf4Layout};

/// Original actual packed projection and validated native row COPY mapping.
#[derive(Clone,Debug)]
pub struct NativePackedProjectionState<R:DeviceRuntime> {rows:ReceivedExpertRows<R>,projection:PackedExpertProjection<R>,dtype:FloatDType}
/// Original actual selected packed chain cache, without a dense base or frozen parameter gradients.
#[derive(Clone,Debug)]
pub struct NativePackedSwiGluState<R:DeviceRuntime> {rows:ReceivedExpertRows<R>,cache:Option<PackedSwiGluCache<R>>,dtype:FloatDType}
fn projection<R,F,I,BT>(payload:PackedExpertPayload<DeviceBackend<R,F,I,BT>>) -> Result<PackedExpertProjection<R>,PackedExpertError>
    where R:DeviceRuntime,F:FloatElement,I:IntElement,BT:BoolElement {
    Ok(match payload {
        PackedExpertPayload::Nf4(value)=>{
            let o=value.options.projection;if o.tile_rows==0 {return Err(MoeError("packed NF4 expert tile rows must be positive").into());}
            let layout=Nf4Layout::new(o.input_features,o.output_features,o.block_size).map_err(rudnn::moe::Nf4ExpertError::from)?;
            PackedExpertProjection::Nf4 {projection:Nf4ExpertProjection::new(value.packed,value.scales,value.codebook,value.options.experts,layout)?,
                execution:Nf4ExpertExecution {tile_rows:o.tile_rows,use_tensor_core:o.use_tensor_core}}
        },
        PackedExpertPayload::Awq(value)=>{
            let projection=AwqGroupedGemm::new(value.qweight,value.qzeros,value.scales,value.bias,value.options.group_size)?;
            let (e,l)=projection.layout();
            if [e,l.input_features,l.output_features]!=[value.options.experts,value.options.input_features,value.options.output_features] {
                return Err(MoeError("AWQ expert options differ from original source packed cube geometry").into());}
            PackedExpertProjection::Awq(projection)
        },
        PackedExpertPayload::Nf4Window {payload:value,element_offset}=>{
            let o=value.options.projection;if o.tile_rows==0 {return Err(MoeError("packed NF4 window tile rows must be positive").into());}
            let layout=Nf4Layout::new(o.input_features,o.output_features,o.block_size).map_err(rudnn::moe::Nf4ExpertError::from)?;
            PackedExpertProjection::Nf4 {projection:Nf4ExpertProjection::from_window(value.packed,value.scales,value.codebook,value.options.experts,layout,element_offset)?,
                execution:Nf4ExpertExecution {tile_rows:o.tile_rows,use_tensor_core:o.use_tensor_core}}
        },
    })
}
impl<R,F,I,BT> FrozenPackedExpertOps for DeviceBackend<R,F,I,BT> where R:DeviceRuntime,F:FloatElement,I:IntElement,BT:BoolElement {
    type PackedExpertError=PackedExpertError;
    type PackedProjectionState=NativePackedProjectionState<R>;
    type PackedSwiGluState=NativePackedSwiGluState<R>;
    fn packed_expert_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,payload:PackedExpertPayload<Self>)
        -> Result<(FloatTensor<Self>,Self::PackedProjectionState),Self::PackedExpertError> {
        let (e,start)=payload.expert_range();let dtype=input.dtype.into();let projection=projection(payload)?;
        let rows=ReceivedExpertRows::new(input,global_ids,start,e)?;let output=projection.forward(rows.grouped())?;
        Ok((rows.restore(output)?,NativePackedProjectionState {rows,projection,dtype}))
    }
    fn packed_expert_input_backward(state:Self::PackedProjectionState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::PackedExpertError> {
        let gradient=state.rows.sort_gradient(Self::float_cast(gradient,state.dtype))?;
        Ok(state.rows.restore(state.projection.input_backward(state.rows.grouped(),gradient)?)?)
    }
    fn packed_swiglu_forward(input:FloatTensor<Self>,global_ids:IntTensor<Self>,gate:PackedExpertPayload<Self>,up:PackedExpertPayload<Self>,down:PackedExpertPayload<Self>,retain_input:bool)
        -> Result<(FloatTensor<Self>,Self::PackedSwiGluState),Self::PackedExpertError> {
        let range=gate.expert_range();if up.expert_range()!=range || down.expert_range()!=range {
            return Err(MoeError("packed gate/up/down must describe the same explicit resident global expert range").into());}
        let experts=PackedSwiGluExperts::new(projection(gate)?,projection(up)?,projection(down)?)?;let dtype=input.dtype.into();
        let rows=ReceivedExpertRows::new(input,global_ids,range.1,range.0)?;
        let (output,cache)=if retain_input {let (output,cache)=experts.forward_training(rows.grouped())?;(output,Some(cache))}
            else {(experts.forward(rows.grouped())?,None)};
        Ok((rows.restore(output)?,NativePackedSwiGluState {rows,cache,dtype}))
    }
    fn packed_swiglu_input_backward(state:Self::PackedSwiGluState,gradient:FloatTensor<Self>) -> Result<FloatTensor<Self>,Self::PackedExpertError> {
        let cache=state.cache.ok_or(MoeError("packed expert forward did not retain actual input VJP intermediates"))?;
        let gradient=state.rows.sort_gradient(Self::float_cast(gradient,state.dtype))?;
        Ok(state.rows.restore(cache.input_backward(gradient)?)?)
    }
}
