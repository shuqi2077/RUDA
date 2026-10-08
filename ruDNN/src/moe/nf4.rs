use super::{GroupedExpertRows,MoeError};
use rublas::tensor_nf4::{Nf4Error,Nf4GroupedGemm,Nf4Layout};
use ruda_kernel::{dsl::{Runtime,calculate_ruda_count_elemwise,prelude::RudaDim},tensor::RudaTensor};
use std::fmt;

/// Original expert routing or original packed NF4 projection failure.
#[derive(Debug)]
pub enum Nf4ExpertError {Rows(MoeError),Projection(Nf4Error)}
impl From<MoeError> for Nf4ExpertError {fn from(error:MoeError) -> Self {Self::Rows(error)}}
impl From<Nf4Error> for Nf4ExpertError {fn from(error:Nf4Error) -> Self {Self::Projection(error)}}
impl fmt::Display for Nf4ExpertError {
    fn fmt(&self,f:&mut fmt::Formatter<'_>) -> fmt::Result {match self {Self::Rows(error)=>write!(f,"{error}"),Self::Projection(error)=>write!(f,"{error}")}}
}
impl std::error::Error for Nf4ExpertError {}

/// Safe selected expert projection on native privately produced dispatch segments.
#[derive(Clone,Debug)]
pub struct Nf4ExpertProjection<R:Runtime> {weights:Nf4GroupedGemm<R>}
impl<R:Runtime> Nf4ExpertProjection<R> {
    /// Connect actual logical `[experts,output,input]` packed payload without reblocking.
    pub fn new(packed:RudaTensor<R>,scales:RudaTensor<R>,codebook:RudaTensor<R>,experts:usize,layout:Nf4Layout) -> Result<Self,Nf4ExpertError> {
        Ok(Self {weights:Nf4GroupedGemm::new(packed,scales,codebook,experts,layout)?})
    }
    /// Actual expert count and per-expert projection geometry.
    pub fn layout(&self) -> (usize,Nf4Layout) {self.weights.layout()}
    /// Resident original byte/scale/book payload size, not peak device memory.
    pub fn packed_payload_bytes(&self) -> usize {self.weights.packed_payload_bytes()}
    /// Native forward over actual selected groups, with no output for unselected experts.
    pub fn forward(&self,rows:&GroupedExpertRows<R>,tile_rows:usize,use_tensor_core:bool) -> Result<RudaTensor<R>,Nf4ExpertError> {
        if rows.experts!=self.weights.layout().0 {return Err(MoeError("NF4 expert payload and private dispatch expert counts differ").into());}
        // SAFETY: only native dispatch/received producers can create these immutable complete segments.
        Ok(unsafe {self.weights.forward_segmented(rows.values.clone(),rows.offsets.clone(),tile_rows,use_tensor_core)}?)
    }
    /// Original selected expert input VJP in FP32; seed storage is caller-selected explicitly.
    pub fn input_backward_f32(&self,rows:&GroupedExpertRows<R>,gradient:RudaTensor<R>,tile_rows:usize,use_tensor_core:bool) -> Result<RudaTensor<R>,Nf4ExpertError> {
        if rows.experts!=self.weights.layout().0 || gradient.meta.num_dims()!=2 || gradient.meta.shape()[0]!=rows.values.meta.shape()[0]
            || gradient.dtype!=rows.values.dtype {return Err(MoeError("NF4 expert seed rows/storage differ from original forward").into());}
        // SAFETY: the original private native group is retained unchanged by forward/cache.
        Ok(unsafe {self.weights.input_backward_segmented_f32(gradient,rows.offsets.clone(),tile_rows,use_tensor_core)}?)
    }
}

/// Explicit execution choice for each original frozen expert projection.
#[derive(Clone,Copy,Debug)]
pub struct Nf4ExpertExecution {
    /// Maximum decoded output-row tile; applies independently within each expert.
    pub tile_rows:usize,
    /// Original fused half/BF16 CMMA path; errors are never retried as tiled GEMM.
    pub use_tensor_core:bool,
}
/// Three actual packed expert projections and the original native SwiGLU arithmetic.
#[derive(Clone,Debug)]
pub struct Nf4SwiGluExperts<R:Runtime> {
    gate:Nf4ExpertProjection<R>,up:Nf4ExpertProjection<R>,down:Nf4ExpertProjection<R>,
}
/// Real first-order input VJP state, with no frozen base matrix gradients or dense shadows.
#[derive(Clone,Debug)]
pub struct Nf4SwiGluCache<R:Runtime> {
    rows:GroupedExpertRows<R>,gate:RudaTensor<R>,up:RudaTensor<R>,experts:Nf4SwiGluExperts<R>,execution:[Nf4ExpertExecution;3],
}
impl<R:Runtime> Nf4SwiGluExperts<R> {
    /// Connect actual gate/up `[E,I,H]` and down `[E,H,I]` source payloads.
    pub fn new(gate:Nf4ExpertProjection<R>,up:Nf4ExpertProjection<R>,down:Nf4ExpertProjection<R>) -> Result<Self,Nf4ExpertError> {
        let (e,g)=gate.layout();let (ue,u)=up.layout();let (de,d)=down.layout();
        if e!=ue || e!=de || g.input_features!=u.input_features || g.output_features!=u.output_features
            || d.input_features!=g.output_features || d.output_features!=g.input_features {
            return Err(MoeError("NF4 gate/up/down expert cube geometry differs").into());
        }
        Ok(Self {gate,up,down})
    }
    /// Native inference retains no gate/up backward cache.
    pub fn forward(&self,rows:&GroupedExpertRows<R>,execution:[Nf4ExpertExecution;3]) -> Result<RudaTensor<R>,Nf4ExpertError> {
        let gate=self.gate.forward(rows,execution[0].tile_rows,execution[0].use_tensor_core)?;
        let up=self.up.forward(rows,execution[1].tile_rows,execution[1].use_tensor_core)?;
        let size=gate.meta.num_elements();
        if size!=0 {let dim=RudaDim::new(gate.client.properties(),size);
            super::kernels::experts::swiglu::launch::<R>(&gate.client,calculate_ruda_count_elemwise(&gate.client,size,dim),dim,
                gate.clone().into_array_arg(),up.into_array_arg(),gate.dtype.into());}
        let mut activated=rows.clone();activated.values=gate;
        self.down.forward(&activated,execution[2].tile_rows,execution[2].use_tensor_core)
    }
    /// Preserve only actual input-path intermediates for the original native first-order chain.
    pub fn forward_training(&self,rows:&GroupedExpertRows<R>,execution:[Nf4ExpertExecution;3]) -> Result<(RudaTensor<R>,Nf4SwiGluCache<R>),Nf4ExpertError> {
        let gate=self.gate.forward(rows,execution[0].tile_rows,execution[0].use_tensor_core)?;
        let up=self.up.forward(rows,execution[1].tile_rows,execution[1].use_tensor_core)?;
        let activated=super::empty(&gate,gate.meta.shape().clone(),gate.dtype);let size=activated.meta.num_elements();
        if size!=0 {let dim=RudaDim::new(gate.client.properties(),size);
            super::kernels::experts::swiglu_out::launch::<R>(&gate.client,calculate_ruda_count_elemwise(&gate.client,size,dim),dim,
                gate.clone().into_array_arg(),up.clone().into_array_arg(),activated.clone().into_array_arg(),gate.dtype.into());}
        let mut group=rows.clone();group.values=activated;
        let output=self.down.forward(&group,execution[2].tile_rows,execution[2].use_tensor_core)?;
        Ok((output,Nf4SwiGluCache {rows:rows.clone(),gate,up,experts:self.clone(),execution}))
    }
}
impl<R:Runtime> Nf4SwiGluCache<R> {
    /// Original packed down VJP -> storage-rounded SwiGLU VJP -> packed gate/up VJPs.
    /// Each packed GEMM accumulates FP32; original activation boundaries retain their storage.
    pub fn input_backward(self,gradient:RudaTensor<R>) -> Result<RudaTensor<R>,Nf4ExpertError> {
        let dtype=self.rows.values.dtype;
        let down=self.experts.down.input_backward_f32(&self.rows,gradient,self.execution[2].tile_rows,self.execution[2].use_tensor_core)?;
        let down=ruprim::elementwise::cast::cast(down,dtype);
        let dg=super::empty(&self.gate,self.gate.meta.shape().clone(),dtype);let du=super::empty(&self.up,self.up.meta.shape().clone(),dtype);
        let size=dg.meta.num_elements();
        if size!=0 {let dim=RudaDim::new(dg.client.properties(),size);
            super::kernels::experts::swiglu_backward::launch::<R>(&dg.client,calculate_ruda_count_elemwise(&dg.client,size,dim),dim,
                self.gate.into_array_arg(),self.up.into_array_arg(),down.into_array_arg(),dg.clone().into_array_arg(),du.clone().into_array_arg(),dtype.into());}
        let dg=self.experts.gate.input_backward_f32(&self.rows,dg,self.execution[0].tile_rows,self.execution[0].use_tensor_core)?;
        let du=self.experts.up.input_backward_f32(&self.rows,du,self.execution[1].tile_rows,self.execution[1].use_tensor_core)?;
        let dg=ruprim::elementwise::cast::cast(dg,dtype);let du=ruprim::elementwise::cast::cast(du,dtype);
        let output=super::empty(&dg,dg.meta.shape().clone(),dtype);let size=output.meta.num_elements();
        if size!=0 {let dim=RudaDim::new(output.client.properties(),size);
            super::kernels::experts::add::launch::<R>(&output.client,calculate_ruda_count_elemwise(&output.client,size,dim),dim,
                dg.into_array_arg(),du.into_array_arg(),output.clone().into_array_arg(),dtype.into());}
        Ok(output)
    }
}
