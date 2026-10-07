use ruda_autodiff::{Autodiff,checkpoint::strategy::CheckpointStrategy,tensor_parallel as region};
use ruda_model::{module::{Module,ModuleMapper,Param,ParamId},tensor::{Tensor,TensorMetadata,backend::Backend,container::TensorContainer}};
use super::BroadcastTensorCollective;

struct ReplicatedParameters<B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B>> {
    communicator: C,
    values: TensorContainer<ParamId>,
    error: Option<C::Error>,
    backend: core::marker::PhantomData<(B,S)>,
}

impl<B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B>> ModuleMapper<Autodiff<B,S>> for ReplicatedParameters<B,S,C> {
    fn map_float<const D: usize>(&mut self,param: Param<Tensor<Autodiff<B,S>,D>>) -> Param<Tensor<Autodiff<B,S>,D>> {
        if self.error.is_some() {return param;}
        let id = param.id;
        param.map(|value| {
            if let Some(existing) = self.values.get::<Autodiff<B,S>>(&id) {
                assert_eq!(existing.rank(),D,"replicated module aliases have different tensor ranks");
                let existing = Tensor::<Autodiff<B,S>,D>::from_primitive(existing);
                assert!(existing.dims() == value.dims() && existing.device() == value.device() && existing.dtype() == value.dtype()
                    && existing.is_require_grad() == value.is_require_grad(),"replicated module aliases have incompatible parameter contracts");
                return existing;
            }
            let result = if value.is_require_grad() {
                if D == 0 {
                    let shape = value.dims();
                    region::copy_to_region(value.clone().reshape([1]),self.communicator.clone()).map(|value|value.reshape(shape))
                } else {region::copy_to_region(value.clone(),self.communicator.clone())}
            } else {Ok(value.clone())};
            match result {
                Ok(value)=>{self.values.register::<Autodiff<B,S>>(id,value.clone().into_primitive());value},
                Err(error)=>{self.error = Some(error);value},
            }
        })
    }
}

/// Preserve an explicitly replicated module's parameter IDs/storage and SUM its shard contributions.
/// Forward values are unchanged; backward combines corresponding replicated parameter gradients.
/// Tied aliases share one region node. Frozen packed values remain in original native quantized storage.
/// The supplied group must hold identical module values and execute the same logical replicated loss;
/// this is not a data-parallel average or a way to reduce unrelated rank-local parameters.
pub fn copy_replicated_module_to_region<B,S,C,M>(module: M,communicator: C) -> Result<M,C::Error>
    where B: Backend,S: CheckpointStrategy,C: BroadcastTensorCollective<B>,M: Module<Autodiff<B,S>> {
    let mut mapper = ReplicatedParameters::<B,S,C> {communicator,values:TensorContainer::new(),error:None,backend:core::marker::PhantomData};
    let module = module.map(&mut mapper);
    if let Some(error) = mapper.error {Err(error)} else {Ok(module)}
}
