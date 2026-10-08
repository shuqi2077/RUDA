use super::{Tensor,Int};
use crate::{Backend,primitive::TensorPrimitive,collective::{VariableTensorCollective,VariableTensorExchange}};

impl<B:Backend,const D:usize> Tensor<B,D> {
    /// Native variable row exchange over the explicit underlying backend communicator.
    /// Differentiable exchanges use ruda-autodiff's collective function on its original transport backend.
    pub fn all_to_all_v<C:VariableTensorCollective<B>>(self,communicator:C,send_counts:&[usize])
        -> Result<VariableTensorExchange<Self>,C::Error> {
        let result=communicator.all_to_all_v_float(self.into_primitive().tensor(),send_counts)?;
        Ok(VariableTensorExchange {value:Tensor::from_primitive(TensorPrimitive::Float(result.value)),receive_counts:result.receive_counts})
    }
}
impl<B:Backend,const D:usize> Tensor<B,D,Int> {
    /// Actual U8/U32/I32/I64 source-rank row exchange without numerical widening or floating conversion.
    pub fn all_to_all_v_int<C:VariableTensorCollective<B>>(self,communicator:C,send_counts:&[usize])
        -> Result<VariableTensorExchange<Self>,C::Error> {
        let result=communicator.all_to_all_v_int(self.into_primitive(),send_counts)?;
        Ok(VariableTensorExchange {value:Tensor::from_primitive(result.value),receive_counts:result.receive_counts})
    }
}
