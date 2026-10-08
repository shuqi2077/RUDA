use super::{GroupedExpertRows,MoeError,Nf4ExpertProjection,Nf4ExpertExecution,Nf4ExpertError};
use rublas::tensor_int4::{AwqGroupedGemm,Int4Error};
use ruda_kernel::{dsl::{Runtime,calculate_ruda_count_elemwise,prelude::RudaDim},tensor::RudaTensor};
use std::fmt;

/// Original native row contract or actual selected packed representation failure.
#[derive(Debug)]
pub enum PackedExpertError {Rows(MoeError),Nf4(Nf4ExpertError),Awq(Int4Error)}
impl From<MoeError> for PackedExpertError {fn from(error:MoeError) -> Self {Self::Rows(error)}}
impl From<Nf4ExpertError> for PackedExpertError {fn from(error:Nf4ExpertError) -> Self {Self::Nf4(error)}}
impl From<Int4Error> for PackedExpertError {fn from(error:Int4Error) -> Self {Self::Awq(error)}}
impl fmt::Display for PackedExpertError {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {Self::Rows(error)=>write!(f,"{error}"),Self::Nf4(error)=>write!(f,"{error}"),Self::Awq(error)=>write!(f,"{error}")}}
}
impl std::error::Error for PackedExpertError {}
/// Caller-selected actual expert projection, without conversion between packed formats.
#[derive(Clone,Debug)]
pub enum PackedExpertProjection<R:Runtime> {
    /// Original NF4 flat-block metadata and explicit native execution choice.
    Nf4 {projection:Nf4ExpertProjection<R>,execution:Nf4ExpertExecution},
    /// Original AWQ permuted I32 words/zero points and independent scale storage.
    Awq(AwqGroupedGemm<R>),
}
impl<R:Runtime> PackedExpertProjection<R> {
    /// Original logical `[experts,input,output]` dimensions, independent of representation.
    pub fn dimensions(&self) -> [usize;3] {match self {
        Self::Nf4 {projection,..}=>{let (e,l)=projection.layout();[e,l.input_features,l.output_features]},
        Self::Awq(projection)=>{let (e,l)=projection.layout();[e,l.input_features,l.output_features]}}}
    /// Compute only the actual selected expert rows using each original coefficient-rounding policy.
    pub fn forward(&self,rows:&GroupedExpertRows<R>) -> Result<RudaTensor<R>,PackedExpertError> {
        if rows.experts!=self.dimensions()[0] {return Err(MoeError("packed payload and private native expert count differ").into());}
        Ok(match self {Self::Nf4 {projection,execution}=>projection.forward(rows,execution.tile_rows,execution.use_tensor_core)?,
            Self::Awq(projection)=>projection.forward(rows.values.clone(),rows.row_experts.clone())?})
    }
    /// First-order native input VJP, retaining the original stored activation boundary.
    pub fn input_backward(&self,rows:&GroupedExpertRows<R>,gradient:RudaTensor<R>) -> Result<RudaTensor<R>,PackedExpertError> {
        if rows.experts!=self.dimensions()[0] || gradient.meta.num_dims()!=2 || gradient.meta.shape()[0]!=rows.values.meta.shape()[0]
            || gradient.dtype!=rows.values.dtype {return Err(MoeError("packed expert seed rows/storage differ from original forward").into());}
        Ok(match self {Self::Nf4 {projection,execution}=>ruprim::elementwise::cast::cast(
            projection.input_backward_f32(rows,gradient,execution.tile_rows,execution.use_tensor_core)?,rows.values.dtype),
            Self::Awq(projection)=>projection.input_backward(gradient,rows.row_experts.clone())?})
    }
}
/// Actual independent AWQ/NF4 choices per gate/up/down, with source-native SwiGLU.
#[derive(Clone,Debug)]
pub struct PackedSwiGluExperts<R:Runtime> {gate:PackedExpertProjection<R>,up:PackedExpertProjection<R>,down:PackedExpertProjection<R>}
/// Actual original first-order cache; no packed base gradients or dense base shadows.
#[derive(Clone,Debug)]
pub struct PackedSwiGluCache<R:Runtime> {rows:GroupedExpertRows<R>,gate:RudaTensor<R>,up:RudaTensor<R>,experts:PackedSwiGluExperts<R>}
impl<R:Runtime> PackedSwiGluExperts<R> {
    /// Connect actual gate/up `[E,I,H]` and down `[E,H,I]`, preserving independent storage.
    pub fn new(gate:PackedExpertProjection<R>,up:PackedExpertProjection<R>,down:PackedExpertProjection<R>) -> Result<Self,PackedExpertError> {
        let [e,h,i]=gate.dimensions();
        if up.dimensions()!=[e,h,i] || down.dimensions()!=[e,i,h] {return Err(MoeError("packed expert gate/up/down source geometry differs").into());}
        Ok(Self {gate,up,down})
    }
    /// Source-native inference retains no first-order activation cache.
    pub fn forward(&self,rows:&GroupedExpertRows<R>) -> Result<RudaTensor<R>,PackedExpertError> {
        let gate=self.gate.forward(rows)?;let up=self.up.forward(rows)?;let size=gate.meta.num_elements();
        if size!=0 {let dim=RudaDim::new(gate.client.properties(),size);
            super::kernels::experts::swiglu::launch::<R>(&gate.client,calculate_ruda_count_elemwise(&gate.client,size,dim),dim,
                gate.clone().into_array_arg(),up.into_array_arg(),gate.dtype.into());}
        let mut activated=rows.clone();activated.values=gate;self.down.forward(&activated)
    }
    /// Preserve actual source gate/up values only when the real input VJP is needed.
    pub fn forward_training(&self,rows:&GroupedExpertRows<R>) -> Result<(RudaTensor<R>,PackedSwiGluCache<R>),PackedExpertError> {
        let gate=self.gate.forward(rows)?;let up=self.up.forward(rows)?;let activated=super::empty(&gate,gate.meta.shape().clone(),gate.dtype);let size=activated.meta.num_elements();
        if size!=0 {let dim=RudaDim::new(gate.client.properties(),size);
            super::kernels::experts::swiglu_out::launch::<R>(&gate.client,calculate_ruda_count_elemwise(&gate.client,size,dim),dim,
                gate.clone().into_array_arg(),up.clone().into_array_arg(),activated.clone().into_array_arg(),gate.dtype.into());}
        let mut group=rows.clone();group.values=activated;
        Ok((self.down.forward(&group)?,PackedSwiGluCache {rows:rows.clone(),gate,up,experts:self.clone()}))
    }
}
impl<R:Runtime> PackedSwiGluCache<R> {
    /// Original packed down input VJP, native storage-rounded SwiGLU VJP, then gate/up input VJPs.
    pub fn input_backward(self,gradient:RudaTensor<R>) -> Result<RudaTensor<R>,PackedExpertError> {
        let dtype=self.rows.values.dtype;let down=self.experts.down.input_backward(&self.rows,gradient)?;
        let dg=super::empty(&self.gate,self.gate.meta.shape().clone(),dtype);let du=super::empty(&self.up,self.up.meta.shape().clone(),dtype);let size=dg.meta.num_elements();
        if size!=0 {let dim=RudaDim::new(dg.client.properties(),size);
            super::kernels::experts::swiglu_backward::launch::<R>(&dg.client,calculate_ruda_count_elemwise(&dg.client,size,dim),dim,
                self.gate.into_array_arg(),self.up.into_array_arg(),down.into_array_arg(),dg.clone().into_array_arg(),du.clone().into_array_arg(),dtype.into());}
        let dg=self.experts.gate.input_backward(&self.rows,dg)?;let du=self.experts.up.input_backward(&self.rows,du)?;
        let output=super::empty(&dg,dg.meta.shape().clone(),dtype);let size=output.meta.num_elements();
        if size!=0 {let dim=RudaDim::new(output.client.properties(),size);
            super::kernels::experts::add::launch::<R>(&output.client,calculate_ruda_count_elemwise(&output.client,size,dim),dim,
                dg.into_array_arg(),du.into_array_arg(),output.clone().into_array_arg(),dtype.into());}
        Ok(output)
    }
}
