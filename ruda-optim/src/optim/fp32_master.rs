use super::{SimpleOptimizer, adaptor::OptimizerAdaptor};
use crate::LearningRate;
use crate::grad_clipping::GradientClipping;
use ruda_model::{
    module::AutodiffModule,
    record::{PrecisionSettings, Record},
    tensor::{
        DType, Tensor,
        backend::{AutodiffBackend, Backend},
    },
};

mod sharded_muon;
mod flat_sharded_muon;

/// Opt-in FP32 master parameters for an existing simple optimizer.
///
/// The wrapped optimizer receives FP32 parameters and gradients; its algorithm,
/// hyperparameters and per-parameter state are unchanged. Returned parameters
/// retain their incoming storage dtype. Existing optimizers do not opt in.
#[derive(Clone)]
pub struct Fp32MasterOptimizer<O> {
    optimizer: O,
    grad_clipping: Option<GradientClipping>,
    gradient_scale: f32,
}

/// Master parameter and the original optimizer's state, recorded together.
/// Use full-precision recorder settings to preserve the master for continuation.
#[derive(Clone)]
pub struct Fp32MasterState<B: Backend, const D: usize, S: Record<B> + Clone> {
    /// Authoritative FP32 parameter after the most recent update.
    pub master: Tensor<B, D>,
    /// State returned by the wrapped optimizer.
    pub inner: Option<S>,
}

impl<O> Fp32MasterOptimizer<O> {
    /// Wrap an existing optimizer without modifying its configuration.
    pub fn new(optimizer: O) -> Self {
        Self {
            optimizer,
            grad_clipping: None,
            gradient_scale: 1.,
        }
    }

    /// Explicitly apply RUDA's existing per-parameter clipping after FP32 conversion.
    pub fn with_grad_clipping(mut self, clipping: GradientClipping) -> Self {
        self.grad_clipping = Some(clipping);
        self
    }

    /// Explicitly divide gradients by this loss scale in FP32 before clipping.
    ///
    /// The caller scales the loss and uses one scale for the accumulation window.
    /// No dynamic scaling, non-finite policy or optimizer-step skipping is enabled.
    pub fn with_gradient_scale(mut self, scale: f32) -> Self {
        assert!(
            scale.is_finite() && scale > 0. && scale.recip().is_finite(),
            "gradient scale must be finite, positive and have a finite reciprocal"
        );
        self.gradient_scale = scale;
        self
    }

    /// Build RUDA's original module adaptor, preserving parameter IDs and records.
    /// Clipping is disabled unless explicitly configured on the wrapper or adaptor.
    pub fn init<B, M>(self) -> OptimizerAdaptor<Self, M, B>
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
        O: SimpleOptimizer<B::InnerBackend>,
    {
        OptimizerAdaptor::from(self)
    }
}

impl<B:Backend,O:crate::ElementwiseShardOptimizer<B>>
    crate::ElementwiseShardOptimizer<B> for Fp32MasterOptimizer<O> {
    fn validate_element_sharding(&self)->Result<(),&'static str> {
        if self.grad_clipping.is_some() {
            return Err("tensor-wide clipping must run before ZeRO-2 gradient sharding");
        }
        self.optimizer.validate_element_sharding()
    }
    fn shard_gradient_dtype(&self,_storage:DType)->DType {DType::F32}
    fn validate_fully_sharded_execution(&self) -> Result<(),&'static str> {self.optimizer.validate_fully_sharded_execution()}
    fn step_fully_sharded<C:ruda_model::tensor::BroadcastTensorCollective<B>>(&self,lr:LearningRate,tensor:Tensor<B,1>,gradient:Tensor<B,1>,
        state:Option<Self::State<1>>,binding:&crate::FullyShardedOptimizerParameter<C>)
        -> Result<(Tensor<B,1>,Option<Self::State<1>>),crate::FullyShardedElementwiseError<C::Error>> {
        let storage=tensor.dtype();
        if !matches!(storage,DType::F32|DType::F16|DType::BF16) {return Err(crate::FullyShardedElementwiseError::Configuration("FP32 masters require original FP32/FP16/BF16 model storage"));}
        let (master,inner)=match state {
            Some(state)=>{
                if state.master.dtype()!=DType::F32 || state.master.dims()!=tensor.dims() {return Err(crate::FullyShardedElementwiseError::State("authoritative native local master shape/precision differs"));}
                (state.master,state.inner)
            },None=>(tensor.cast(DType::F32),None),
        };
        let gradient=gradient.cast(DType::F32);let gradient=if self.gradient_scale==1.0 {gradient} else {gradient/self.gradient_scale};
        let gradient=if let Some(clipping)=&self.grad_clipping {super::fully_sharded_elementwise::clip(gradient,binding,clipping)?} else {gradient};
        let (master,inner)=self.optimizer.step_fully_sharded(lr,master,gradient,inner,binding)?;
        let tensor=master.clone().cast(storage);Ok((tensor,Some(Fp32MasterState {master,inner})))
    }
}

impl<B: Backend, O: SimpleOptimizer<B>> SimpleOptimizer<B> for Fp32MasterOptimizer<O> {
    type State<const D: usize> = Fp32MasterState<B, D, O::State<D>>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let storage_dtype = tensor.dtype();
        assert!(
            matches!(storage_dtype, DType::F32 | DType::F16 | DType::BF16),
            "FP32 master optimizer requires FP32/FP16/BF16 parameter storage"
        );
        assert_eq!(
            tensor.dims(),
            grad.dims(),
            "master parameter/gradient shape mismatch"
        );
        let (master, inner) = match state {
            Some(state) => {
                assert_eq!(state.master.dtype(), DType::F32, "master must remain FP32");
                assert_eq!(
                    state.master.dims(),
                    tensor.dims(),
                    "master state shape mismatch"
                );
                (state.master, state.inner)
            }
            None => (tensor.cast(DType::F32), None),
        };
        let grad = grad.cast(DType::F32);
        let grad = if self.gradient_scale == 1. {
            grad
        } else {
            grad / self.gradient_scale
        };
        let grad = match &self.grad_clipping {
            Some(clipping) => clipping.clip_gradient(grad),
            None => grad,
        };
        let (master, inner) = self.optimizer.step(lr, master, grad, inner);
        assert_eq!(
            master.dtype(),
            DType::F32,
            "wrapped optimizer must return FP32 parameters"
        );
        let tensor = master.clone().cast(storage_dtype);
        (tensor, Some(Fp32MasterState { master, inner }))
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &B::Device) -> Self::State<D> {
        Fp32MasterState {
            master: state.master.to_device(device),
            inner: state.inner.map(|inner| O::to_device(inner, device)),
        }
    }
}

impl<B: Backend, const D: usize, S: Record<B> + Clone> Record<B> for Fp32MasterState<B, D, S> {
    type Item<P: PrecisionSettings> = <(Tensor<B, D>, Option<S>) as Record<B>>::Item<P>;

    fn into_item<P: PrecisionSettings>(self) -> Self::Item<P> {
        (self.master, self.inner).into_item::<P>()
    }

    fn from_item<P: PrecisionSettings>(item: Self::Item<P>, device: &B::Device) -> Self {
        let (master, inner): (Tensor<B, D>, Option<S>) = Record::from_item::<P>(item, device);
        Self {
            master: master.cast(DType::F32),
            inner,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AdamWConfig, GradientsParams, Optimizer, SgdConfig, TestAutodiffBackend, TestBackend,
    };
    use ruda_model::{
        module::{Initializer, Module},
        record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    };
    use ruda_nn::LinearConfig;

    #[test]
    fn master_retains_updates_smaller_than_half_storage_resolution() {
        let device = Default::default();
        for dtype in [DType::F16, DType::BF16] {
            let original = SgdConfig::new().build::<TestBackend>();
            let optimizer = Fp32MasterOptimizer::new(original.clone());
            let mut parameter = Tensor::<TestBackend, 1>::from_floats([1.], &device).cast(dtype);
            let gradient = Tensor::<TestBackend, 1>::ones([1], &device).cast(dtype);
            let mut expected = parameter.clone().cast(DType::F32);
            let mut state = None;
            let mut expected_state = None;
            for _ in 0..64 {
                (expected, expected_state) = original.step(
                    0.0001,
                    expected,
                    gradient.clone().cast(DType::F32),
                    expected_state,
                );
                (parameter, state) = optimizer.step(0.0001, parameter, gradient.clone(), state);
                assert_eq!(parameter.dtype(), dtype);
                let master = &state.as_ref().unwrap().master;
                assert_eq!(master.dtype(), DType::F32);
                master.to_data().assert_eq(&expected.to_data(), false);
                parameter
                    .to_data()
                    .assert_eq(&expected.clone().cast(dtype).to_data(), false);
            }
            assert!(parameter.cast(DType::F32).into_scalar() < 1.);
        }
    }

    #[test]
    fn adamw_master_and_original_moments_continue_after_record_roundtrip() {
        let device = Default::default();
        for dtype in [DType::F16, DType::BF16] {
            let original = AdamWConfig::new()
                .with_amsgrad(true)
                .with_cautious_weight_decay(true)
                .build();
            let optimizer = Fp32MasterOptimizer::new(original.clone());
            let mut parameter =
                Tensor::<TestBackend, 1>::from_floats([1., -2.], &device).cast(dtype);
            let gradient = Tensor::<TestBackend, 1>::from_floats([0.25, -0.5], &device).cast(dtype);
            let mut expected = parameter.clone().cast(DType::F32);
            let mut state = None;
            let mut expected_state = None;
            for _ in 0..3 {
                (expected, expected_state) = original.step(
                    0.001,
                    expected,
                    gradient.clone().cast(DType::F32),
                    expected_state,
                );
                (parameter, state) = optimizer.step(0.001, parameter, gradient.clone(), state);
                let saved = state.take().unwrap();
                saved.master.to_data().assert_eq(&expected.to_data(), false);
                let inner = saved.inner.as_ref().unwrap();
                let reference = expected_state.as_ref().unwrap();
                assert_eq!(inner.momentum.time, reference.momentum.time);
                for (actual, expected) in [
                    (&inner.momentum.moment_1, &reference.momentum.moment_1),
                    (&inner.momentum.moment_2, &reference.momentum.moment_2),
                    (
                        inner.momentum.max_moment_2.as_ref().unwrap(),
                        reference.momentum.max_moment_2.as_ref().unwrap(),
                    ),
                ] {
                    assert_eq!(actual.dtype(), DType::F32);
                    actual.to_data().assert_eq(&expected.to_data(), false);
                }
                let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
                let bytes =
                    <BinBytesRecorder<FullPrecisionSettings> as Recorder<TestBackend>>::record(
                        &recorder,
                        saved,
                        (),
                    )
                    .unwrap();
                state = Some(
                    <BinBytesRecorder<FullPrecisionSettings> as Recorder<TestBackend>>::load(
                        &recorder, bytes, &device,
                    )
                    .unwrap(),
                );
                parameter
                    .to_data()
                    .assert_eq(&expected.clone().cast(dtype).to_data(), false);
            }
        }
    }

    #[test]
    fn explicit_clipping_uses_original_algorithm_after_fp32_conversion() {
        let device = Default::default();
        let original = SgdConfig::new().build::<TestBackend>();
        let gradient =
            Tensor::<TestBackend, 1>::from_floats([1000., -2000.], &device).cast(DType::F16);
        for clipping in [GradientClipping::Value(1.), GradientClipping::Norm(1.)] {
            let parameter = Tensor::<TestBackend, 1>::ones([2], &device).cast(DType::F16);
            let fp32_gradient = clipping.clip_gradient(gradient.clone().cast(DType::F32));
            let (expected, _) = original.step(
                0.01,
                parameter.clone().cast(DType::F32),
                fp32_gradient,
                None,
            );
            for scale in [1., 16.] {
                let optimizer = Fp32MasterOptimizer::new(original.clone())
                    .with_gradient_scale(scale)
                    .with_grad_clipping(clipping.clone());
                let (actual, state) =
                    optimizer.step(0.01, parameter.clone(), gradient.clone() * scale, None);
                state
                    .unwrap()
                    .master
                    .to_data()
                    .assert_eq(&expected.to_data(), false);
                actual
                    .to_data()
                    .assert_eq(&expected.clone().cast(DType::F16).to_data(), false);
            }
        }
    }

    #[test]
    fn muon_master_reuses_original_fp32_matrix_update_and_momentum() {
        let device = Default::default();
        for dtype in [DType::F16, DType::BF16] {
            let original = crate::MuonConfig::new()
                .with_stable_normalization(true)
                .build::<TestBackend>();
            let optimizer = Fp32MasterOptimizer::new(original.clone());
            let mut parameter = Tensor::<TestBackend, 2>::ones([2, 3], &device).cast(dtype);
            let gradient =
                Tensor::<TestBackend, 2>::from_floats([[1., 2., 3.], [-2., 1., 4.]], &device)
                    .cast(dtype);
            let mut expected = parameter.clone().cast(DType::F32);
            let mut state = None;
            let mut expected_state = None;
            for _ in 0..3 {
                (expected, expected_state) = original.step(
                    0.01,
                    expected,
                    gradient.clone().cast(DType::F32),
                    expected_state,
                );
                (parameter, state) = optimizer.step(0.01, parameter, gradient.clone(), state);
                let saved = state.as_ref().unwrap();
                saved.master.to_data().assert_eq(&expected.to_data(), false);
                let velocity = saved.inner.as_ref().unwrap().momentum.velocity();
                assert_eq!(velocity.dtype(), DType::F32);
                velocity.to_data().assert_eq(
                    &expected_state
                        .as_ref()
                        .unwrap()
                        .momentum
                        .velocity()
                        .to_data(),
                    false,
                );
                parameter
                    .to_data()
                    .assert_eq(&expected.clone().cast(dtype).to_data(), false);
            }
        }
    }

    #[test]
    fn original_module_adaptor_preserves_param_id_and_recorded_master() {
        let device = Default::default();
        let mut model = LinearConfig::new(2, 1)
            .with_bias(false)
            .with_initializer(Initializer::Ones)
            .init::<TestAutodiffBackend>(&device)
            .to_dtype(ruda_model::tensor::FloatDType::BF16);
        let id = model.weight.id;
        let mut optimizer = Fp32MasterOptimizer::new(AdamWConfig::new().build()).init();
        for _ in 0..3 {
            let input = Tensor::<TestAutodiffBackend, 2>::from_floats([[1., 2.]], &device)
                .cast(DType::BF16);
            let gradients = model.forward(input).sum().backward();
            let gradients = GradientsParams::from_grads(gradients, &model);
            model = optimizer.step(0.001, model, gradients);
            assert_eq!(model.weight.id, id);
            assert_eq!(model.weight.val().dtype(), DType::BF16);
            let record = optimizer.to_record();
            assert_eq!(record.len(), 1);
            let state: Fp32MasterState<TestBackend, 2, crate::AdamWState<TestBackend, 2>> =
                record[&id].clone().into_state();
            assert_eq!(state.master.dtype(), DType::F32);
            assert_eq!(state.master.dims(), [2, 1]);
        }
    }
}
