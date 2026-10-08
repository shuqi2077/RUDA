
use ruda_model::{
    Tensor,
    tensor::{
        backend::{AutodiffBackend, Backend},
        container::TensorContainer,
    },
};
#[cfg(feature = "collective")]
use ruccl::{CollectiveError, PeerId, ReduceOperation, all_reduce};

use ruda_model::module::{AutodiffModule, ParamId};

use super::visitor::{GradientsParamsChangeDevice, GradientsParamsConverter};

mod record;
pub use record::GradientsParamsRecord;

#[cfg(feature = "collective")]
mod collective;

/// Data type that contains gradients for parameters.
#[derive(Default, Debug)]
pub struct GradientsParams {
    pub(super) container: TensorContainer<ParamId>,
}

/// Two actual gradient containers cannot be joined without replacing a source parameter's derivative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GradientMergeError {
    /// Actual parameter identities occurring in both input containers.
    pub duplicates: alloc::vec::Vec<ParamId>,
}
impl core::fmt::Display for GradientMergeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "gradient containers share parameter IDs: {:?}", self.duplicates)
    }
}
impl core::error::Error for GradientMergeError {}

impl GradientsParams {
    /// Creates a new [GradientsParams](GradientsParams).
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract each tensor gradients for the given [module](AutodiffModule).
    ///
    /// Note: This consumes the gradients. See ['from_module'] to extract gradients only for
    ///  a specific module.
    pub fn from_grads<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: B::Gradients,
        module: &M,
    ) -> Self {
        let mut grads = grads;
        Self::from_module(&mut grads, module)
    }

    /// Extract each tensor gradients for the given [module](AutodiffModule).
    pub fn from_module<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: &mut B::Gradients,
        module: &M,
    ) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor = GradientsParamsConverter::<M, B>::new(grads, &mut grads_params, None);
        module.visit(&mut visitor);
        grads_params
    }

    /// Extract tensor gradients for the given [module](AutodiffModule) and given parameters.
    pub fn from_params<B: AutodiffBackend, M: AutodiffModule<B>>(
        grads: &mut B::Gradients,
        module: &M,
        params: &[ParamId],
    ) -> Self {
        let mut grads_params = GradientsParams::new();
        let mut visitor =
            GradientsParamsConverter::<M, B>::new(grads, &mut grads_params, Some(params.to_vec()));
        module.visit(&mut visitor);
        grads_params
    }

    /// Get the gradients for the given [parameter id](ParamId).
    ///
    /// # Notes
    ///
    /// You should use [remove](GradientsParams::remove) if you want to get the gradients
    /// only one time.
    pub fn get<B, const D: usize>(&self, id: ParamId) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        self.container.get(&id).map(Tensor::from_primitive)
    }

    /// Remove the gradients for the given [parameter id](ParamId).
    pub fn remove<B, const D: usize>(&mut self, id: ParamId) -> Option<Tensor<B, D>>
    where
        B: Backend,
    {
        self.container.remove(&id).map(Tensor::from_primitive)
    }

    /// Register a gradients tensor for the given [parameter id](ParamId).
    ///
    /// # Notes
    ///
    /// If a tensor is already registered for the given [parameter id](ParamId), it will be replaced.
    pub fn register<B, const D: usize>(&mut self, id: ParamId, value: Tensor<B, D>)
    where
        B: Backend,
    {
        self.container.register(id, value.into_primitive())
    }

    /// The number of gradients tensors registered.
    pub fn len(&self) -> usize {
        self.container.len()
    }

    /// If any tensor is contained.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retain original native primitive values/IDs/devices for a pending-gradient
    /// continuation. No readback, cast, normalization or missing value is created.
    /// `B` must be the same backend used to register these actual derivatives.
    pub fn clone_native<B: Backend>(&self) -> Self {
        let mut values = Self::new();
        for id in self.container.ids() {
            if let Some(value) = self.container.get::<B>(id) { values.container.register::<B>(*id, value); }
        }
        values
    }

    /// Move actual native derivatives into an explicit ID subset and its unchanged remainder.
    /// No tensor copies, dtype conversions, normalization or missing-gradient values are created.
    pub fn partition<B: Backend>(mut self, parameters: &[ParamId]) -> (Self, Self) {
        let mut selected = Self::new();
        for &id in parameters {
            if let Some(value) = self.container.remove::<B>(&id) {
                selected.container.register::<B>(id, value);
            }
        }
        (selected, self)
    }

    /// Join disjoint actual native derivative sets, rejecting every collision before moving any value.
    /// Useful after independently reducing explicit data/tensor/expert parameter groups.
    pub fn merge_disjoint<B: Backend>(mut self, mut other: Self) -> Result<Self, GradientMergeError> {
        let ids = other.container.ids().into_iter().copied().collect::<alloc::vec::Vec<_>>();
        let mut duplicates = ids.iter().copied()
            .filter(|id| self.container.get::<B>(id).is_some())
            .collect::<alloc::vec::Vec<_>>();
        if !duplicates.is_empty() {
            duplicates.sort();
            return Err(GradientMergeError { duplicates });
        }
        for id in ids {
            if let Some(value) = other.container.remove::<B>(&id) {
                self.container.register::<B>(id, value);
            }
        }
        Ok(self)
    }

    #[cfg(feature = "collective")]
    pub(crate) fn primitive<B: Backend>(
        &self,
        id: ParamId,
    ) -> Option<ruda_model::tensor::TensorPrimitive<B>> {
        self.container.get(&id)
    }

    /// Change the device of each tensor gradients registered for the given [module](AutodiffModule).
    pub fn to_device<B: AutodiffBackend, M: AutodiffModule<B>>(
        mut self,
        device: &B::Device,
        module: &M,
    ) -> Self {
        let mut visitor = GradientsParamsChangeDevice::<M, B>::new(device, &mut self);
        module.visit(&mut visitor);
        self
    }

    /// Syncs the gradient params with the other peers in the collective.
    #[cfg(feature = "collective")]
    pub fn all_reduce<B: Backend>(
        mut self,
        peer_id: PeerId,
        op: ReduceOperation,
    ) -> Result<Self, CollectiveError> {
        let mut ids = self
            .container
            .ids()
            .into_iter()
            .copied()
            .collect::<Vec<ParamId>>();
        // This is crucial, since the all-reduce operations need to happen in the same order for the same parameters on all nodes!
        ids.sort();

        for id in ids {
            let Some(grad) = self.container.remove::<B>(&id) else {
                todo!()
            };

            let grad = match grad {
                ruda_model::tensor::TensorPrimitive::Float(grad) => {
                    let grad = all_reduce::<B>(peer_id, grad, op)?;
                    ruda_model::tensor::TensorPrimitive::Float(grad)
                }
                ruda_model::tensor::TensorPrimitive::QFloat(_grad) => {
                    unimplemented!("quantized all-reduce unimplemented")
                }
            };

            self.container.register::<B>(id, grad);
        }

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestAutodiffBackend;
    use ruda_model::module::{Module, list_param_ids};
    use ruda_model::tensor::{Distribution, backend::Backend};
    use ruda_nn::{Linear, LinearConfig};

    #[test]
    fn test_convert_grads() {
        let device = Default::default();
        let layer_1 = layer::<TestAutodiffBackend>(&device);
        let mut layer_2 = layer_1.clone();
        layer_2 = layer_2.fork(&device);
        let loss_1 = layer_1.forward(random_tensor(&device));
        let loss_2 = layer_2.forward(random_tensor(&device));
        let grads_1 = GradientsParams::from_grads(loss_1.backward(), &layer_1);
        let grads_2 = GradientsParams::from_grads(loss_2.backward(), &layer_2);

        let param_ids_1 = list_param_ids(&layer_1);
        let param_ids_2 = list_param_ids(&layer_2);

        assert_eq!(param_ids_1, param_ids_2);
        assert_eq!(grads_1.len(), param_ids_1.len());
        assert_eq!(grads_2.len(), param_ids_2.len());
    }

    fn layer<B: Backend>(device: &B::Device) -> Linear<B> {
        LinearConfig::new(20, 20).init(device)
    }

    fn random_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 2>::random([2, 20], Distribution::Default, device)
    }
}
