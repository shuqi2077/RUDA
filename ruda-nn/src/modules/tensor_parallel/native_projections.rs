use super::*;

pub(crate) fn inference_gather<B:Backend,C:BroadcastTensorCollective<B>,const D:usize>(value:Tensor<B,D>,communicator:C,axis:usize)
    -> Result<Tensor<B,D>,C::Error> {
    assert!(axis<D,"native projection gather axis differs");
    if communicator.world_size()==1 {return Ok(value);}
    let output = communicator.all_gather_float(value.swap_dims(0,axis).into_primitive().tensor())?;
    Ok(Tensor::<B,D>::from_primitive(TensorPrimitive::Float(output)).swap_dims(0,axis))
}

pub(crate) fn inference_sum<B:Backend,C:BroadcastTensorCollective<B>,const D:usize>(value:Tensor<B,D>,communicator:C)
    -> Result<Tensor<B,D>,C::Error> {
    if communicator.world_size()==1 {return Ok(value);}
    Ok(Tensor::from_primitive(TensorPrimitive::Float(communicator.all_reduce_sum(value.into_primitive().tensor())?)))
}

pub(crate) fn inference_scatter<B:Backend,C:BroadcastTensorCollective<B>,const D:usize>(value:Tensor<B,D>,communicator:&C,axis:usize) -> Tensor<B,D> {
    assert!(axis<D,"native projection scatter axis differs");
    let world = communicator.world_size() as usize;let rank = communicator.rank() as usize;
    assert!(world>0 && rank<world,"native projection scatter topology differs");
    let length = value.dims()[axis];
    assert!(length>0 && length.is_multiple_of(world),"native projection scatter needs equal nonempty feature slices");
    let size = length/world;value.slice_dim(axis,rank*size..(rank+1)*size)
}

impl<B:Backend> ColumnParallelLinear<B> {
    /// Actual native local-column projection, optionally gathering output features in rank order.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C,gather_output:bool)
        -> Result<Tensor<B,D>,C::Error> {
        assert!(D>0,"native column projection needs a feature axis");
        let output = self.local.forward(input);
        if gather_output {inference_gather(output,communicator,D-1)} else {Ok(output)}
    }
}

impl<B:Backend> RowParallelLinear<B> {
    /// Native partial-output sum and one original replicated bias addition, not a rank-multiplied bias.
    pub fn forward_inference<C:BroadcastTensorCollective<B>,const D:usize>(&self,input:Tensor<B,D>,communicator:C,input_is_parallel:bool)
        -> Result<Tensor<B,D>,C::Error> {
        assert!(D>0,"native row projection needs a feature axis");
        let input = if input_is_parallel {input} else {inference_scatter(input,&communicator,D-1)};
        let partial = linear(input,self.local.weight.val(),None);
        let output = inference_sum(partial,communicator)?;
        Ok(if let Some(bias) = &self.local.bias {
            let mut shape = [1;D];shape[D-1] = bias.val().dims()[0];output+bias.val().reshape(shape)
        } else {output})
    }
}
