use super::{TensorDevice,TensorDeviceError,TensorElement,storage};
use crate::rank::communicator::RankCommunicator;
use ruda_tensor::{Backend,DType,Shape,TensorMetadata,bf16,f16,collective::{VariableTensorCollective,VariableTensorExchange}};

struct RowExchangePlan {
    shape:Shape,
    row_elements:usize,
    input_elements:usize,
    send_elements:Vec<usize>,
}
impl RowExchangePlan {
    fn new(shape:Shape,counts:&[usize],world:u32) -> Result<Self,TensorDeviceError> {
        if shape.is_empty() || world==0 || counts.len()!=world as usize {
            return Err(TensorDeviceError::InvalidBuffer("row exchange requires a leading axis and one count per rank"));
        }
        let rows=counts.iter().try_fold(0usize,|sum,&count|sum.checked_add(count))
            .ok_or(TensorDeviceError::InvalidBuffer("row exchange row count overflows"))?;
        if rows!=shape[0] {return Err(TensorDeviceError::InvalidBuffer("row exchange counts differ from actual leading-axis rows"));}
        let row_elements=shape[1..].iter().try_fold(1usize,|size,&dim|size.checked_mul(dim))
            .filter(|&size|size!=0).ok_or(TensorDeviceError::InvalidBuffer("row exchange trailing axes are zero or overflowed"))?;
        let input_elements=rows.checked_mul(row_elements).ok_or(TensorDeviceError::InvalidBuffer("row exchange input size overflows"))?;
        let send_elements=counts.iter().map(|count|count.checked_mul(row_elements)).collect::<Option<Vec<_>>>()
            .ok_or(TensorDeviceError::InvalidBuffer("row exchange send element count overflows"))?;
        Ok(Self {shape,row_elements,input_elements,send_elements})
    }
    fn receive(mut self,counts:Vec<usize>,world:u32) -> Result<(Shape,Vec<usize>,usize),TensorDeviceError> {
        if counts.len()!=world as usize || counts.iter().any(|count|!count.is_multiple_of(self.row_elements)) {
            return Err(TensorDeviceError::InvalidBuffer("row exchange response does not contain complete source rows"));
        }
        let total=counts.iter().try_fold(0usize,|sum,&count|sum.checked_add(count))
            .ok_or(TensorDeviceError::InvalidBuffer("row exchange receive count overflows"))?;
        self.shape[0]=total/self.row_elements;
        Ok((self.shape,counts.into_iter().map(|count|count/self.row_elements).collect(),total))
    }
}
impl<B:Backend> VariableTensorCollective<B> for RankCommunicator<TensorDevice<B>> {
    fn all_to_all_v_float(&self,value:B::FloatTensorPrimitive,send_counts:&[usize])
        -> Result<VariableTensorExchange<B::FloatTensorPrimitive>,Self::Error> {
        match value.dtype() {
            DType::F32=>self.exchange_float_rows::<f32>(value,send_counts),DType::F16=>self.exchange_float_rows::<f16>(value,send_counts),
            DType::BF16=>self.exchange_float_rows::<bf16>(value,send_counts),dtype=>Err(TensorDeviceError::UnsupportedDType(dtype)),
        }
    }
    fn all_to_all_v_int(&self,value:B::IntTensorPrimitive,send_counts:&[usize])
        -> Result<VariableTensorExchange<B::IntTensorPrimitive>,Self::Error> {
        match value.dtype() {
            DType::U8=>self.exchange_int_rows::<u8>(value,send_counts),DType::I32=>self.exchange_int_rows::<i32>(value,send_counts),
            DType::U32=>self.exchange_int_rows::<u32>(value,send_counts),
            DType::I64=>self.exchange_int_rows::<i64>(value,send_counts),dtype=>Err(TensorDeviceError::UnsupportedDType(dtype)),
        }
    }
}
impl<B:Backend> RankCommunicator<TensorDevice<B>> {
    fn exchange_float_rows<T:TensorElement>(&self,value:B::FloatTensorPrimitive,counts:&[usize])
        -> Result<VariableTensorExchange<B::FloatTensorPrimitive>,TensorDeviceError> {
        let execution=self.execution();execution.validate_type::<T>()?;
        if &B::float_device(&value)!=execution.device() {return Err(TensorDeviceError::DeviceMismatch);}
        let plan=RowExchangePlan::new(value.shape(),counts,self.world_size())?;storage::checked_length::<T>(plan.input_elements)?;
        let buffer=execution.import_float::<T>(B::float_reshape(value,Shape::new([plan.input_elements])))?;
        // Reuse the original configured direct/pairwise transport. It is host-staged, not peer-memory GPU communication.
        let (received,_)=self.tensor_collective::<T>().all_to_all_v(&buffer,&plan.send_elements)?;
        let (shape,receive_counts,total)=plan.receive(received.counts,self.world_size())?;storage::checked_length::<T>(total)?;
        let value=match received.buffer {Some(buffer)=>buffer.float_tensor()?,None if total==0=>execution.allocate::<T>(0)?.float_tensor()?,
            None=>return Err(TensorDeviceError::InvalidBuffer("nonempty floating row exchange lacks its native output buffer"))};
        if value.shape()!=Shape::new([total]) {return Err(TensorDeviceError::InvalidBuffer("floating row exchange output length differs from actual receive counts"));}
        Ok(VariableTensorExchange {value:B::float_reshape(value,shape),receive_counts})
    }
    fn exchange_int_rows<T:TensorElement>(&self,value:B::IntTensorPrimitive,counts:&[usize])
        -> Result<VariableTensorExchange<B::IntTensorPrimitive>,TensorDeviceError> {
        let execution=self.execution();execution.validate_type::<T>()?;
        if &B::int_device(&value)!=execution.device() {return Err(TensorDeviceError::DeviceMismatch);}
        let plan=RowExchangePlan::new(value.shape(),counts,self.world_size())?;storage::checked_length::<T>(plan.input_elements)?;
        let buffer=execution.import_int::<T>(B::int_reshape(value,Shape::new([plan.input_elements])))?;
        let (received,_)=self.tensor_collective::<T>().all_to_all_v(&buffer,&plan.send_elements)?;
        let (shape,receive_counts,total)=plan.receive(received.counts,self.world_size())?;storage::checked_length::<T>(total)?;
        let value=match received.buffer {Some(buffer)=>buffer.int_tensor()?,None if total==0=>execution.allocate::<T>(0)?.int_tensor()?,
            None=>return Err(TensorDeviceError::InvalidBuffer("nonempty integer row exchange lacks its native output buffer"))};
        if value.shape()!=Shape::new([total]) {return Err(TensorDeviceError::InvalidBuffer("integer row exchange output length differs from actual receive counts"));}
        Ok(VariableTensorExchange {value:B::int_reshape(value,shape),receive_counts})
    }
}
