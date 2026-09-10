
use super::gate_controller::GateController;
use crate::activation::{Activation, ActivationConfig};
use ruda_model::config::Config;
use ruda_model::module::Initializer;
use ruda_model::module::Module;
use ruda_model::module::{Content, DisplaySettings, ModuleDisplay};
use ruda_model::tensor::Tensor;
use ruda_model::tensor::backend::Backend;

/// Configuration to create a [gru](Gru) module using the [init function](GruConfig::init).
#[derive(Config, Debug)]
pub struct GruConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the Gru transformation.
    pub bias: bool,
    /// If reset gate should be applied after weight multiplication.
    ///
    /// This configuration option controls how the reset gate is applied to the hidden state.
    /// * `true` - (Default) Match the initial arXiv version of the paper [Learning Phrase Representations using RNN Encoder-Decoder for
    ///   Statistical Machine Translation (v1)](https://arxiv.org/abs/1406.1078v1) and apply the reset gate after multiplication by
    ///   the weights. This matches the behavior of [PyTorch GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU).
    /// * `false` - Match the most recent revision of [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine
    ///   Translation (v3)](https://arxiv.org/abs/1406.1078) and apply the reset gate before the weight multiplication.
    ///
    /// The differing implementations can give slightly different numerical results and have different efficiencies. For more
    /// motivation for why the `true` can be more efficient see [Optimizing RNNs with Differentiable Graphs](https://svail.github.io/diff_graphs).
    ///
    /// To set this field to `false` use [`with_reset_after`](`GruConfig::with_reset_after`).
    #[config(default = "true")]
    pub reset_after: bool,
    /// Gru initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
    /// Activation function for the update and reset gates.
    /// Default is Sigmoid, which is standard for GRU gates.
    #[config(default = "ActivationConfig::Sigmoid")]
    pub gate_activation: ActivationConfig,
    /// Activation function for the new/candidate gate.
    /// Default is Tanh, which is standard for GRU.
    #[config(default = "ActivationConfig::Tanh")]
    pub hidden_activation: ActivationConfig,
    /// Optional hidden state clip threshold. If provided, hidden state values are clipped
    /// to the range `[-clip, +clip]` after each timestep. This can help prevent
    /// exploding values during inference.
    pub clip: Option<f64>,
}

/// The Gru (Gated recurrent unit) module. This implementation is for a unidirectional, stateless, Gru.
///
/// Introduced in the paper: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078).
///
/// Should be created with [GruConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Gru<B: Backend> {
    /// The update gate controller.
    pub update_gate: GateController<B>,
    /// The reset gate controller.
    pub reset_gate: GateController<B>,
    /// The new gate controller.
    pub new_gate: GateController<B>,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If reset gate should be applied after weight multiplication.
    pub reset_after: bool,
    /// Activation function for gates (update, reset).
    pub gate_activation: Activation<B>,
    /// Activation function for new/candidate gate.
    pub hidden_activation: Activation<B>,
    /// Optional hidden state clip threshold.
    pub clip: Option<f64>,
}

impl<B: Backend> ModuleDisplay for Gru<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, _] = self.update_gate.input_transform.weight.shape().dims();
        let bias = self.update_gate.input_transform.bias.is_some();

        content
            .add("d_input", &d_input)
            .add("d_hidden", &self.d_hidden)
            .add("bias", &bias)
            .add("reset_after", &self.reset_after)
            .optional()
    }
}

impl GruConfig {
    /// Initialize a new [gru](Gru) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gru<B> {
        let d_output = self.d_hidden;

        let update_gate = GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
            device,
        );
        let reset_gate = GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
            device,
        );
        let new_gate = GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
            device,
        );

        Gru {
            update_gate,
            reset_gate,
            new_gate,
            d_hidden: self.d_hidden,
            reset_after: self.reset_after,
            gate_activation: self.gate_activation.init(device),
            hidden_activation: self.hidden_activation.init(device),
            clip: self.clip,
        }
    }
}

impl<B: Backend> Gru<B> {
    /// Applies the forward pass on the input tensor. This GRU implementation
    /// returns a state tensor with dimensions `[batch_size, sequence_length, hidden_size]`.
    ///
    /// # Parameters
    /// - batched_input: `[batch_size, sequence_length, input_size]`.
    /// - state: An optional tensor representing an initial cell state with dimensions
    ///   `[batch_size, hidden_size]`. If none is provided, an empty state will be used.
    ///
    /// # Returns
    /// - output: `[batch_size, sequence_length, hidden_size]`
    pub fn forward(
        &self,
        batched_input: Tensor<B, 3>,
        state: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 3> {
        let device = batched_input.device();
        let [batch_size, seq_length, _] = batched_input.shape().dims();

        self.forward_iter(
            batched_input.iter_dim(1).zip(0..seq_length),
            state,
            batch_size,
            seq_length,
            &device,
        )
        .0
    }

    /// Forward pass variant that accepts an iterator over timesteps.
    /// Used by BiGru to process sequences in either direction.
    ///
    /// # Parameters
    /// - input_timestep_iter: Iterator yielding (input_tensor, timestep_index) pairs.
    ///   The timestep_index determines where in the output tensor to store results.
    /// - state: Optional initial hidden state with shape `[batch_size, hidden_size]`.
    /// - batch_size: Batch size of the input.
    /// - seq_length: Sequence length of the input.
    /// - device: Device to create tensors on.
    ///
    /// # Returns
    /// - output: `[batch_size, sequence_length, hidden_size]`
    /// - final_hidden: Final hidden state `[batch_size, hidden_size]`
    pub(crate) fn forward_iter<I: Iterator<Item = (Tensor<B, 3>, usize)>>(
        &self,
        input_timestep_iter: I,
        state: Option<Tensor<B, 2>>,
        batch_size: usize,
        seq_length: usize,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let mut batched_hidden_state =
            Tensor::empty([batch_size, seq_length, self.d_hidden], device);

        let mut hidden_t = match state {
            Some(state) => state,
            None => Tensor::zeros([batch_size, self.d_hidden], device),
        };

        for (input_t, t) in input_timestep_iter {
            let input_t = input_t.squeeze_dim(1);

            // u(pdate)g(ate) tensors
            let biased_ug_input_sum =
                self.gate_product(&input_t, &hidden_t, None, &self.update_gate);
            let update_values = self.gate_activation.forward(biased_ug_input_sum);

            // r(eset)g(ate) tensors
            let biased_rg_input_sum =
                self.gate_product(&input_t, &hidden_t, None, &self.reset_gate);
            let reset_values = self.gate_activation.forward(biased_rg_input_sum);

            // n(ew)g(ate) tensor
            let biased_ng_input_sum = if self.reset_after {
                self.gate_product(&input_t, &hidden_t, Some(&reset_values), &self.new_gate)
            } else {
                let reset_t = hidden_t.clone().mul(reset_values);
                self.gate_product(&input_t, &reset_t, None, &self.new_gate)
            };
            let candidate_state = self.hidden_activation.forward(biased_ng_input_sum);

            // calculate linear interpolation between previous hidden state and candidate state:
            // h_t = (1 - z_t) * g_t + z_t * h_{t-1}
            let one_minus_z = update_values.clone().neg().add_scalar(1.0);
            hidden_t = candidate_state.mul(one_minus_z) + update_values.mul(hidden_t);

            // Apply hidden state clipping if configured
            if let Some(clip) = self.clip {
                hidden_t = hidden_t.clamp(-clip, clip);
            }

            let unsqueezed_hidden_state = hidden_t.clone().unsqueeze_dim(1);

            batched_hidden_state = batched_hidden_state.slice_assign(
                [0..batch_size, t..(t + 1), 0..self.d_hidden],
                unsqueezed_hidden_state,
            );
        }

        (batched_hidden_state, hidden_t)
    }

    /// Helper function for performing weighted matrix product for a gate and adds
    /// bias, if any, and optionally applies reset to hidden state.
    ///
    ///  Mathematically, performs `Wx*X + r .* (Wh*H + b)`, where:
    ///     Wx = weight matrix for the connection to input vector X
    ///     Wh = weight matrix for the connection to hidden state H
    ///     X = input vector
    ///     H = hidden state
    ///     b = bias terms
    ///     r = reset state
    fn gate_product(
        &self,
        input: &Tensor<B, 2>,
        hidden: &Tensor<B, 2>,
        reset: Option<&Tensor<B, 2>>,
        gate: &GateController<B>,
    ) -> Tensor<B, 2> {
        let input_product = input.clone().matmul(gate.input_transform.weight.val());
        let hidden_product = hidden.clone().matmul(gate.hidden_transform.weight.val());

        let input_part = match &gate.input_transform.bias {
            Some(bias) => input_product + bias.val().unsqueeze(),
            None => input_product,
        };

        let hidden_part = match &gate.hidden_transform.bias {
            Some(bias) => hidden_product + bias.val().unsqueeze(),
            None => hidden_product,
        };

        match reset {
            Some(r) => input_part + r.clone().mul(hidden_part),
            None => input_part + hidden_part,
        }
    }
}

/// Configuration to create a [BiGru](BiGru) module using the [init function](BiGruConfig::init).
#[derive(Config, Debug)]
pub struct BiGruConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the BiGru transformation.
    pub bias: bool,
    /// If reset gate should be applied after weight multiplication.
    #[config(default = "true")]
    pub reset_after: bool,
    /// BiGru initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
    /// If true, the input tensor is expected to be `[batch_size, seq_length, input_size]`.
    /// If false, the input tensor is expected to be `[seq_length, batch_size, input_size]`.
    #[config(default = true)]
    pub batch_first: bool,
    /// Activation function for the update and reset gates.
    #[config(default = "ActivationConfig::Sigmoid")]
    pub gate_activation: ActivationConfig,
    /// Activation function for the new/candidate gate.
    #[config(default = "ActivationConfig::Tanh")]
    pub hidden_activation: ActivationConfig,
    /// Optional hidden state clip threshold.
    pub clip: Option<f64>,
}

/// The BiGru module. This implementation is for Bidirectional GRU.
///
/// Based on the paper: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078).
///
/// Should be created with [BiGruConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct BiGru<B: Backend> {
    /// GRU for the forward direction.
    pub forward: Gru<B>,
    /// GRU for the reverse direction.
    pub reverse: Gru<B>,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If true, input is `[batch_size, seq_length, input_size]`.
    /// If false, input is `[seq_length, batch_size, input_size]`.
    pub batch_first: bool,
}

impl<B: Backend> ModuleDisplay for BiGru<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, _] = self
            .forward
            .update_gate
            .input_transform
            .weight
            .shape()
            .dims();
        let bias = self.forward.update_gate.input_transform.bias.is_some();

        content
            .add("d_input", &d_input)
            .add("d_hidden", &self.d_hidden)
            .add("bias", &bias)
            .optional()
    }
}

impl BiGruConfig {
    /// Initialize a new [Bidirectional GRU](BiGru) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BiGru<B> {
        // Internal GRUs always use batch_first=true; BiGru handles layout conversion
        let base_config = GruConfig::new(self.d_input, self.d_hidden, self.bias)
            .with_initializer(self.initializer.clone())
            .with_reset_after(self.reset_after)
            .with_gate_activation(self.gate_activation.clone())
            .with_hidden_activation(self.hidden_activation.clone())
            .with_clip(self.clip);

        BiGru {
            forward: base_config.clone().init(device),
            reverse: base_config.init(device),
            d_hidden: self.d_hidden,
            batch_first: self.batch_first,
        }
    }
}

impl<B: Backend> BiGru<B> {
    /// Applies the forward pass on the input tensor. This Bidirectional GRU implementation
    /// returns the state for each element in a sequence (i.e., across seq_length) and a final state.
    ///
    /// ## Parameters:
    /// - batched_input: The input tensor of shape:
    ///   - `[batch_size, sequence_length, input_size]` if `batch_first` is true (default)
    ///   - `[sequence_length, batch_size, input_size]` if `batch_first` is false
    /// - state: An optional tensor representing the initial hidden state with shape
    ///   `[2, batch_size, hidden_size]`. If no initial state is provided, it is initialized to zeros.
    ///
    /// ## Returns:
    /// - output: A tensor representing the output features. Shape:
    ///   - `[batch_size, sequence_length, hidden_size * 2]` if `batch_first` is true
    ///   - `[sequence_length, batch_size, hidden_size * 2]` if `batch_first` is false
    /// - state: The final forward and reverse hidden states stacked along dimension 0
    ///   with shape `[2, batch_size, hidden_size]`.
    pub fn forward(
        &self,
        batched_input: Tensor<B, 3>,
        state: Option<Tensor<B, 3>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // Convert to batch-first layout internally if needed
        let batched_input = if self.batch_first {
            batched_input
        } else {
            batched_input.swap_dims(0, 1)
        };

        let device = batched_input.clone().device();
        let [batch_size, seq_length, _] = batched_input.shape().dims();

        let [init_state_forward, init_state_reverse] = match state {
            Some(state) => {
                let hidden_state_forward = state
                    .clone()
                    .slice([0..1, 0..batch_size, 0..self.d_hidden])
                    .squeeze_dim(0);
                let hidden_state_reverse = state
                    .slice([1..2, 0..batch_size, 0..self.d_hidden])
                    .squeeze_dim(0);

                [Some(hidden_state_forward), Some(hidden_state_reverse)]
            }
            None => [None, None],
        };

        // forward direction
        let (batched_hidden_state_forward, final_state_forward) = self.forward.forward_iter(
            batched_input.clone().iter_dim(1).zip(0..seq_length),
            init_state_forward,
            batch_size,
            seq_length,
            &device,
        );

        // reverse direction
        let (batched_hidden_state_reverse, final_state_reverse) = self.reverse.forward_iter(
            batched_input.iter_dim(1).rev().zip((0..seq_length).rev()),
            init_state_reverse,
            batch_size,
            seq_length,
            &device,
        );

        let output = Tensor::cat(
            [batched_hidden_state_forward, batched_hidden_state_reverse].to_vec(),
            2,
        );

        // Convert output back to seq-first layout if needed
        let output = if self.batch_first {
            output
        } else {
            output.swap_dims(0, 1)
        };

        let state = Tensor::stack([final_state_forward, final_state_reverse].to_vec(), 0);

        (output, state)
    }
}

#[cfg(test)]
mod tests;
