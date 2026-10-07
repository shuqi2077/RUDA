
use core::marker::PhantomData;

use ruda_model::module::{AutodiffModule, ModuleVisitor, Param};
use ruda_model::record::RecorderError;
use ruda_model::tensor::{FloatDType, Tensor, backend::AutodiffBackend};

use super::{GradientsParams, GradientsParamsRecord};

/// Accumulate gradients into a single [Gradients](AutodiffBackend::Gradients) object.
pub struct GradientsAccumulator<M> {
    grads: GradientsParams,
    phantom: PhantomData<M>,
}

impl<M> Default for GradientsAccumulator<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M> GradientsAccumulator<M> {
    /// Create a new gradients accumulator.
    pub fn new() -> Self {
        Self {
            grads: GradientsParams::new(),
            phantom: PhantomData,
        }
    }
}

impl<M> GradientsAccumulator<M> {
    /// Borrow pending gradients without clearing an accumulation window.
    pub fn pending(&self) -> &GradientsParams { &self.grads }

    /// Snapshot pending gradients without resetting the accumulation window.
    pub fn try_to_record<B: AutodiffBackend>(&self) -> Result<GradientsParamsRecord, RecorderError>
    where
        M: AutodiffModule<B>,
    {
        self.grads.try_to_record::<B::InnerBackend>()
    }

    /// Asynchronously snapshot pending gradients without resetting the accumulator.
    pub async fn to_record_async<B: AutodiffBackend>(
        &self,
    ) -> Result<GradientsParamsRecord, RecorderError>
    where
        M: AutodiffModule<B>,
    {
        self.grads.to_record_async::<B::InnerBackend>().await
    }

    /// Replace pending gradients with checkpoint state on the given device.
    ///
    /// The model IDs and the caller's accumulation count must be restored from
    /// the same checkpoint. A rejected record leaves the accumulator unchanged.
    pub fn load_record<B: AutodiffBackend>(
        &mut self,
        record: GradientsParamsRecord,
        device: &B::Device,
    ) -> Result<(), RecorderError>
    where
        M: AutodiffModule<B>,
    {
        self.grads = GradientsParams::from_record::<B::InnerBackend>(record, device)?;
        Ok(())
    }

    /// Accumulate the given gradients for each parameter in the given module.
    pub fn accumulate<B: AutodiffBackend>(&mut self, module: &M, grads: GradientsParams)
    where
        M: AutodiffModule<B>,
    {
        let mut visitor = ModuleGradsAccumulator::<M>::new(&mut self.grads, grads, None);
        module.visit(&mut visitor);
    }

    /// Accumulate after converting incoming and pending gradients to the given dtype.
    ///
    /// FP32 accumulation can retain small additions to half-precision gradients.
    /// Parameter storage, loss normalization and accumulation counts are unchanged.
    /// Checkpoints preserve the pending gradient dtype; select the same dtype on
    /// subsequent calls after restoring. No accumulator reset is performed.
    pub fn accumulate_with_dtype<B: AutodiffBackend>(
        &mut self,
        module: &M,
        grads: GradientsParams,
        dtype: FloatDType,
    ) where
        M: AutodiffModule<B>,
    {
        let mut visitor = ModuleGradsAccumulator::<M>::new(&mut self.grads, grads, Some(dtype));
        module.visit(&mut visitor);
    }

    /// Return the accumulated gradients and reset the accumulator state.
    pub fn grads(&mut self) -> GradientsParams {
        let mut grads = GradientsParams::new();
        core::mem::swap(&mut self.grads, &mut grads);

        grads
    }
}

#[derive(new)]
struct ModuleGradsAccumulator<'a, M> {
    grads: &'a mut GradientsParams,
    grads_new: GradientsParams,
    dtype: Option<FloatDType>,
    phantom: PhantomData<M>,
}

impl<B: AutodiffBackend, M: AutodiffModule<B>> ModuleVisitor<B> for ModuleGradsAccumulator<'_, M> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let dtype = self.dtype;
        let convert = |tensor: Tensor<B::InnerBackend, D>| match dtype {
            Some(dtype) => tensor.cast(dtype),
            None => tensor,
        };
        let grad_updated = match self
            .grads_new
            .remove::<B::InnerBackend, D>(param.id)
            .map(convert)
        {
            Some(new) => match self
                .grads
                .remove::<B::InnerBackend, D>(param.id)
                .map(convert)
            {
                Some(grad) => grad.add(new),
                None => new,
            },
            None => match self
                .grads
                .remove::<B::InnerBackend, D>(param.id)
                .map(convert)
            {
                Some(grad) => grad,
                None => return,
            },
        };

        self.grads
            .register::<B::InnerBackend, D>(param.id, grad_updated);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestAutodiffBackend, TestBackend};
    use ruda_model::module::Module;
    use ruda_model::tensor::{Distribution, backend::Backend};
    use ruda_nn::{Linear, LinearConfig};

    #[test]
    fn test_accumulate_gradients_one_step() {
        let device = Default::default();
        let mut accumulator = GradientsAccumulator::new();
        let layer = layer::<TestAutodiffBackend>(&device);
        let loss = layer.forward(random_tensor::<TestAutodiffBackend>(&device));
        let grads = GradientsParams::from_grads(loss.backward(), &layer);

        accumulator.accumulate(&layer, grads);

        let grads = accumulator.grads();
        assert!(!grads.is_empty())
    }

    #[test]
    fn test_accumulate_gradients_two_steps() {
        let device = Default::default();
        let mut accumulator = GradientsAccumulator::new();
        let layer = layer::<TestAutodiffBackend>(&device);
        let loss_1 = layer.forward(random_tensor(&device));
        let loss_2 = layer.forward(random_tensor(&device));
        let grads_1 = GradientsParams::from_grads(loss_1.backward(), &layer);
        let grads_2 = GradientsParams::from_grads(loss_2.backward(), &layer);

        accumulator.accumulate(&layer, grads_1);
        accumulator.accumulate(&layer, grads_2);

        let grads = accumulator.grads();
        assert_eq!(grads.len(), 2)
    }

    #[test]
    fn fp32_accumulation_retains_small_half_gradients_after_pending_record_restore() {
        let device = Default::default();
        for dtype in [FloatDType::F16, FloatDType::BF16] {
            let model = LinearConfig::new(1, 1)
                .with_bias(false)
                .init::<TestAutodiffBackend>(&device)
                .to_dtype(dtype);
            let id = model.weight.id;
            let gradient = |value| {
                let mut grads = GradientsParams::new();
                grads.register(
                    id,
                    Tensor::<TestBackend, 2>::full([1, 1], value, &device).cast(dtype),
                );
                grads
            };
            let mut original = GradientsAccumulator::new();
            let mut fp32 = GradientsAccumulator::new();
            original.accumulate(&model, gradient(1024.));
            fp32.accumulate_with_dtype(&model, gradient(1024.), FloatDType::F32);
            for step in 0..8 {
                original.accumulate(&model, gradient(0.125));
                fp32.accumulate_with_dtype(&model, gradient(0.125), FloatDType::F32);
                if step == 3 {
                    let record = fp32.try_to_record::<TestAutodiffBackend>().unwrap();
                    fp32 = GradientsAccumulator::new();
                    fp32.load_record::<TestAutodiffBackend>(record, &device)
                        .unwrap();
                }
            }
            let result = fp32.grads().remove::<TestBackend, 2>(id).unwrap();
            assert_eq!(result.dtype(), FloatDType::F32.into());
            assert_eq!(result.into_scalar(), 1025.);
            let unchanged = original.grads().remove::<TestBackend, 2>(id).unwrap();
            assert_eq!(unchanged.dtype(), dtype.into());
            assert_eq!(unchanged.cast(FloatDType::F32).into_scalar(), 1024.);
            assert!(fp32.grads().is_empty());
        }
    }

    fn layer<B: Backend>(device: &B::Device) -> Linear<B> {
        LinearConfig::new(20, 20).init(device)
    }

    fn random_tensor<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 2>::random([2, 20], Distribution::Default, device)
    }
}
