use super::{GroupedExpertRows,MoeError,elements,empty,float_tensor,kernels,same_device};
use super::dispatch::sort_token_rows;
use ruda_core::tensor::DType;
use ruda_kernel::{dsl::{Runtime,calculate_ruda_count_elemwise,prelude::RudaDim},tensor::{RudaTensor,contiguous::into_contiguous}};

/// Actual received assignments, grouped locally with an exact inverse COPY permutation.
#[derive(Debug,Clone)]
pub struct ReceivedExpertRows<R:Runtime> {
    grouped:GroupedExpertRows<R>,original_rows:RudaTensor<R>,sorted_rows:RudaTensor<R>,
}
impl<R:Runtime> ReceivedExpertRows<R> {
    /// Group actual received rows by explicitly owned global expert range. IDs stay U32.
    /// Only one validity counter is read as coordination metadata before indexing;
    /// no activation/weight values are downloaded or evaluated on the host.
    pub fn new(input:RudaTensor<R>,global_ids:RudaTensor<R>,begin:usize,experts:usize) -> Result<Self,MoeError> {
        float_tensor(&input)?;same_device(&input,&global_ids)?;
        let end=begin.checked_add(experts).filter(|&end|end<=u32::MAX as usize).ok_or(MoeError("received expert range overflows U32"))?;
        if input.meta.num_dims()!=2 || input.meta.shape()[1]==0 || global_ids.meta.shape()[..]!=[input.meta.shape()[0]]
            || global_ids.dtype!=DType::U32 || global_ids.qparams.is_some() || !input.client.same_execution_queue(&global_ids.client) {
            return Err(MoeError("received expert rows require floating [rows,width] and native U32 IDs on one queue"));
        }
        elements(input.meta.shape())?;elements(&[experts.checked_add(1).ok_or(MoeError("received expert prefix overflows"))?])?;
        let rows=input.meta.shape()[0];
        if experts==0 && rows!=0 {return Err(MoeError("a zero-expert owner cannot receive assignment rows"));}
        let local=empty(&input,[rows],DType::U32);
        if rows!=0 {
            let invalid=empty(&input,[1],DType::U32);let dim=RudaDim::new(input.client.properties(),1);
            kernels::dispatch::clear_counts::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,1,dim),dim,invalid.clone().into_array_arg());
            let dim=RudaDim::new(input.client.properties(),rows);
            kernels::received::localize::launch::<R>(&input.client,calculate_ruda_count_elemwise(&input.client,rows,dim),dim,
                into_contiguous(global_ids).into_array_arg(),local.clone().into_array_arg(),invalid.clone().into_array_arg(),begin as u32,end as u32);
            let flag=ruda_core::future::block_on(ruda_kernel::tensor::readback::into_data(invalid)).map_err(|_|MoeError("received expert validity read failed"))?
                .to_vec::<u32>().map_err(|_|MoeError("received expert validity metadata has invalid storage"))?;
            if flag.as_slice()!=[0u32] {return Err(MoeError("received expert ID is outside the explicitly owned global range"));}
        }
        let sorted=sort_token_rows(input,local,experts,1)?;
        Ok(Self {grouped:GroupedExpertRows {values:sorted.values,row_experts:sorted.row_experts,offsets:sorted.offsets,experts},
            original_rows:sorted.slot_rows,sorted_rows:sorted.sorted_slots})
    }
    /// Original native contiguous expert segments; raw imported offsets are never accepted.
    pub fn grouped(&self) -> &GroupedExpertRows<R> {&self.grouped}
    fn permute(&self,values:RudaTensor<R>,rows:&RudaTensor<R>) -> Result<RudaTensor<R>,MoeError> {
        float_tensor(&values)?;same_device(&self.grouped.values,&values)?;
        if values.meta.num_dims()!=2 || values.meta.shape()[0]!=self.grouped.values.meta.shape()[0] || values.meta.shape()[1]==0
            || !values.client.same_execution_queue(&self.grouped.values.client) {return Err(MoeError("received row permutation shape/device/queue mismatch"));}
        let shape=values.meta.shape().clone();let size=elements(&shape)?;let width=shape[1];let output=empty(&values,shape,values.dtype);
        if size!=0 {let values=into_contiguous(values);let dim=RudaDim::new(values.client.properties(),size);
            kernels::received::permute::launch::<R>(&values.client,calculate_ruda_count_elemwise(&values.client,size,dim),dim,
                values.clone().into_array_arg(),rows.clone().into_array_arg(),output.clone().into_array_arg(),width as u32,values.dtype.into());}Ok(output)
    }
    /// Restore original source-rank receive order without a routing-weight multiplication or reduction.
    pub fn restore(&self,expert_values:RudaTensor<R>) -> Result<RudaTensor<R>,MoeError> {self.permute(expert_values,&self.original_rows)}
    /// COPY VJP: reorder original receive-axis seeds to the exact native sorted expert rows.
    pub fn sort_gradient(&self,received_gradient:RudaTensor<R>) -> Result<RudaTensor<R>,MoeError> {self.permute(received_gradient,&self.sorted_rows)}
}
