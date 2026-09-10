// SPDX-License-Identifier: Apache-2.0
//! Small full-model example: hidden Linear uses Muon, head and biases use AdamW.
//! This is a correctness/usage example, not a speed or convergence benchmark.
use ruda_autodiff::Autodiff;
use ruda_model::{module::Module, tensor::{Tensor, TensorData, backend::Backend}};
use ruda_nn::{Linear, LinearConfig};
use ruda_optim::{
    AdamWConfig, AdjustLrFn, GradientsParams, MuonAdamWConfig, MuonConfig,
    MuonMatrixLayout, MuonMomentumMode, Optimizer,
};

#[cfg(feature = "test-cuda")]
type Inner = ruda_tensor_device::cuda::Cuda<f32>;
#[cfg(not(feature = "test-cuda"))]
type Inner = ruda_tensor_host::Host;
type Training = Autodiff<Inner>;

#[derive(Module, Debug)]
struct Network<B: Backend> {
    hidden: Linear<B>,
    head: Linear<B>,
}
impl<B: Backend> Network<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.head.forward(self.hidden.forward(input).tanh())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let steps = match std::env::args().nth(1) { Some(value) => value.parse::<usize>()?, None => 20 };
    if !(1..=1000).contains(&steps) { return Err("choose 1..=1000 demonstration steps".into()); }
    let device = Default::default();
    Inner::seed(&device, 42);
    let mut model = Network::<Training> {
        hidden: LinearConfig::new(4, 8).init(&device),
        head: LinearConfig::new(8, 2).init(&device),
    };
    let mut optim = MuonAdamWConfig::new()
        .with_muon(MuonConfig::new()
            .with_momentum_mode(MuonMomentumMode::Ema)
            .with_stable_normalization(true)
            // RUDA Linear stores [input, output], not PyTorch's [output, input].
            .with_matrix_layout(MuonMatrixLayout::InputOutput)
            .with_adjust_lr_fn(AdjustLrFn::Original))
        .with_adamw(AdamWConfig::new().with_epsilon(1e-8).with_weight_decay(0.01))
        .init(&model, &[model.hidden.weight.id])?;
    let inputs: Vec<f32> = (0..32).map(|i| ((i * 3 % 11) as f32 - 5.0) * 0.1).collect();
    let input = Tensor::<Training, 2>::from_data(TensorData::new(inputs, [8, 4]), &device);
    let target = Tensor::<Training, 2>::zeros([8, 2], &device);
    for step in 0..steps {
        let loss = (model.forward(input.clone()) - target.clone()).square().mean();
        let value = loss.to_data().to_vec::<f32>()?[0];
        if !value.is_finite() { return Err("non-finite demo loss; optimizer step not submitted".into()); }
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        // Gradients here are full, local, unscaled FP32 tensors. No AMP/FSDP.
        model = optim.try_step_with_lrs(0.02, 0.0003, model, grads)?;
        Inner::sync(&device)?;
        println!("step={step} loss={value:.8}");
    }
    let record = optim.to_record();
    println!("muon_states={} adamw_states={}; save model ids and both states together",
        record.muon_state_count(), record.adamw_state_count());
    Ok(())
}
