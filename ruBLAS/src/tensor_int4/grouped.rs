use super::{AwqGemmLayout,Int4Error,grouped_kernel};
use ruda_core::{device::Device,tensor::{DType,Shape}};
use ruda_kernel::{dsl::{Runtime,calculate_ruda_count_elemwise,prelude::RudaDim},
    tensor::{RudaTensor,allocation::empty_device_contiguous_dtype,contiguous::into_contiguous}};

/// Actual AWQ expert cubes with the original permuted eight-code I32 word layout.
/// Scale storage controls coefficient rounding independently of activation storage.
#[derive(Clone,Debug)]
pub struct AwqGroupedGemm<R:Runtime> {
    qweight:RudaTensor<R>,qzeros:RudaTensor<R>,scales:RudaTensor<R>,bias:Option<RudaTensor<R>>,experts:usize,layout:AwqGemmLayout,
}
impl<R:Runtime> AwqGroupedGemm<R> {
    /// Connect original `[E,K,N/8]`, `[E,K/group,N/8]`, `[E,K/group,N]` and optional `[E,N]`.
    pub fn new(qweight:RudaTensor<R>,qzeros:RudaTensor<R>,scales:RudaTensor<R>,bias:Option<RudaTensor<R>>,group_size:usize) -> Result<Self,Int4Error> {
        if qweight.meta.num_dims()!=3 || scales.meta.num_dims()!=3 {return Err(Int4Error("AWQ expert weights/scales must retain rank-three source storage"));}
        let experts=qweight.meta.shape()[0];let layout=AwqGemmLayout::new(qweight.meta.shape()[1],scales.meta.shape()[2],group_size)?;
        let (k,n,g)=(layout.input_features,layout.output_features,layout.groups());
        if experts==0 || experts>=u32::MAX as usize || experts.checked_mul(k).and_then(|size|size.checked_mul(n)).is_none_or(|size|size>u32::MAX as usize)
            || qweight.dtype!=DType::I32 || qzeros.dtype!=DType::I32 || qweight.meta.shape()[..]!=[experts,k,n/8]
            || qzeros.meta.shape()[..]!=[experts,g,n/8] || scales.meta.shape()[..]!=[experts,g,n]
            || !matches!(scales.dtype,DType::F32|DType::F16|DType::BF16) {
            return Err(Int4Error("AWQ expert cube geometry/storage differs from original complete-group packed layout"));
        }
        for value in [&qweight,&qzeros,&scales].into_iter().chain(bias.iter()) {
            if value.qparams.is_some() || value.device.to_id()!=qweight.device.to_id() || !value.client.same_execution_queue(&qweight.client) {
                return Err(Int4Error("AWQ expert operands must retain plain original storage on one device/queue"));
            }
        }
        if bias.as_ref().is_some_and(|value|value.dtype!=scales.dtype || value.meta.shape()[..]!=[experts,n]) {
            return Err(Int4Error("AWQ expert bias must retain original per-expert scale-dtype output vectors"));
        }
        Ok(Self {qweight:into_contiguous(qweight),qzeros:into_contiguous(qzeros),scales:into_contiguous(scales),bias:bias.map(into_contiguous),experts,layout})
    }
    /// Actual resident expert count and original per-expert layout.
    pub fn layout(&self) -> (usize,AwqGemmLayout) {(self.experts,self.layout)}
    /// Actual resident packed words/zeros/scales/bias bytes, excluding activations.
    pub fn packed_payload_bytes(&self) -> Result<usize,Int4Error> {
        self.layout.packed_payload_bytes(self.scales.dtype,self.bias.is_some())?.checked_mul(self.experts).ok_or(Int4Error("AWQ expert payload bytes overflow"))
    }
    /// Selected native forward, with original FP32 scalar accumulation and activation rounding.
    /// Like native floating grouped GEMM, out-of-range local row IDs denote padding and produce zero.
    pub fn forward(&self,input:RudaTensor<R>,row_experts:RudaTensor<R>) -> Result<RudaTensor<R>,Int4Error> {self.project(input,row_experts,false)}
    /// Original rounded-coefficient transpose VJP, retaining incoming activation seed storage.
    pub fn input_backward(&self,gradient:RudaTensor<R>,row_experts:RudaTensor<R>) -> Result<RudaTensor<R>,Int4Error> {self.project(gradient,row_experts,true)}
    fn project(&self,input:RudaTensor<R>,row_experts:RudaTensor<R>,backward:bool) -> Result<RudaTensor<R>,Int4Error> {
        let (k,n)=(self.layout.input_features,self.layout.output_features);let (source,target)=if backward {(n,k)}else {(k,n)};
        if input.meta.num_dims()!=2 || input.meta.shape()[1]!=source || !matches!(input.dtype,DType::F32|DType::F16|DType::BF16)
            || row_experts.dtype!=DType::U32 || row_experts.meta.shape()[..]!=[input.meta.shape()[0]] {
            return Err(Int4Error("AWQ grouped projection requires original floating row width and native U32 local expert IDs"));
        }
        for value in [&input,&row_experts] {if value.qparams.is_some() || value.device.to_id()!=self.qweight.device.to_id()
            || !value.client.same_execution_queue(&self.qweight.client) {return Err(Int4Error("AWQ grouped rows and payload must share original native device/queue"));}}
        let rows=input.meta.shape()[0];
        if rows.checked_mul(source).is_none_or(|size|size>u32::MAX as usize) || rows.checked_mul(target).is_none_or(|size|size>u32::MAX as usize) {
            return Err(Int4Error("AWQ grouped activations exceed U32 indexing"));
        }
        let output=empty_device_contiguous_dtype(input.client.clone(),input.device.clone(),Shape::from([rows,target]),input.dtype);
        let size=rows*target;if size==0 {return Ok(output);}
        let input=into_contiguous(input);let ids=into_contiguous(row_experts);let dim=RudaDim::new(input.client.properties(),size);
        if backward {grouped_kernel::input_backward::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,size,dim),dim,
            input.clone().into_array_arg(),self.qweight.clone().into_array_arg(),self.qzeros.clone().into_array_arg(),self.scales.clone().into_array_arg(),
            ids.into_array_arg(),output.clone().into_array_arg(),self.experts as u32,k as u32,n as u32,self.layout.group_size as u32,input.dtype.into(),self.scales.dtype.into());}
        else {grouped_kernel::forward::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,size,dim),dim,
            input.clone().into_array_arg(),self.qweight.clone().into_array_arg(),self.qzeros.clone().into_array_arg(),self.scales.clone().into_array_arg(),
            self.bias.as_ref().unwrap_or(&self.scales).clone().into_array_arg(),ids.into_array_arg(),output.clone().into_array_arg(),
            self.experts as u32,k as u32,n as u32,self.layout.group_size as u32,u32::from(self.bias.is_some()),input.dtype.into(),self.scales.dtype.into());}
        Ok(output)
    }
}
