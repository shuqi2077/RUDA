#![allow(clippy::excessive_precision)]

use super::Reduction;
use ruda_model::config::Config;
use ruda_model::module::Module;
use ruda_model::tensor::{Int, Tensor, backend::Backend};

/// Configuration for the [CTC Loss](CTCLoss) module.
#[derive(Config, Debug)]
pub struct CTCLossConfig {
    /// The index number used to represent the blank label. Default value is `0`.
    #[config(default = 0)]
    pub blank: usize,
    /// Whether to zero infinite losses and the associated gradients. Default value is `false`.
    #[config(default = false)]
    pub zero_infinity: bool,
}

impl CTCLossConfig {
    /// Initialize a new [CTC Loss](CTCLoss) module
    pub fn init(&self) -> CTCLoss {
        CTCLoss {
            blank: self.blank,
            zero_infinity: self.zero_infinity,
        }
    }
}

/// Computes the Connectionist Temporal Classification (CTC) loss.
///
/// Calculates the loss between a continuous (unsegmented) time series and a target sequence.
/// CTC sums over the probability of all possible alignments of the input to the target,
/// producing a loss value that is differentiable with respect to each input node.
///
/// The input to this loss is expected to be **log-probabilities** (e.g,, via `log_softmax`),
/// not raw logits.
///
/// # References
///
/// - [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
///
/// # Example
///
/// ```rust,ignore
/// use ruda_tensor::api::{Tensor, Int};
/// use ruda_tensor::api::activation::log_softmax;
/// use ruda_nn::loss::{CTCLossConfig, CTCLoss};
///
/// let device = Default::default();
///
/// // Initialize CTC Loss with default configuration
/// let ctc_loss = CTCLossConfig::new().init();
///
/// // Initialize CTC Loss with custom configuration
/// let ctc_loss = CTCLossConfig::new()
///     .with_blank(1)
///     .with_zero_infinity(true)
///     .init();
///
/// // Prepare inputs (Logits shape: [Time, Batch, Class])
/// // In your actual code, the logits would be the output of your model
/// let logits = Tensor::<B, 3>::ones([10, 2, 5], &device);
/// let log_probs = log_softmax(logits, 2);
///
/// // Targets shape: [Batch, Max_Target_Len]
/// // Note: Targets should not contain the blank index (1).
/// let targets = Tensor::<B, 2, Int>::from_data([[0, 2], [3, 4]], &device);
///
/// // Lengths shape: [Batch]
/// let input_lengths = Tensor::<B, 1, Int>::from_data([10, 8], &device);
/// let target_lengths = Tensor::<B, 1, Int>::from_data([2, 2], &device);
///
/// // Compute loss
/// let loss = ctc_loss.forward(log_probs, targets, input_lengths, target_lengths);
/// ```
#[derive(Module, Clone, Debug)]
pub struct CTCLoss {
    blank: usize,
    zero_infinity: bool,
}

impl CTCLoss {
    /// Computes the CTC loss for the input log-probabilities and targets with no reduction applied.
    ///
    /// # Arguments
    ///
    /// - `log_probs`: The log-probabilities of the outputs (e.g., from `log_softmax`).
    /// - `targets`: A 2D tensor containing the target class indices. These indices should not
    ///   include the blank index used in CTC loss. The targets are padded to the length of the longest sequence.
    /// - `input_lengths`: A 1D tensor containing the actual length of the input sequence for each batch. This
    ///   allows retrieving the actual sequence of log-probabilities from `log_probs` if the batch contains
    ///   sequences of varying lengths.
    /// - `target_lengths`: A 1D tensor containing the actual length of the target sequence for each target
    ///   sequence in `targets`.
    ///
    /// # Returns
    ///
    /// - A 1D tensor of shape `[batch_size]` containing the loss for each sample.
    ///
    /// # Shapes
    ///
    /// - `log_probs`: `[time_steps, batch_size, num_classes]` where `num_classes` includes blank.
    /// - `targets`: `[batch_size, max_target_length]`
    /// - `input_lengths`: `[batch_size]`
    /// - `target_lengths`: `[batch_size]`
    pub fn forward<B: Backend>(
        &self,
        log_probs: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let [max_input_length, batch_size, num_classes] = log_probs.dims();
        let max_target_len = targets.dims()[1];
        let input_lengths_len = input_lengths.dims()[0];
        let target_lengths_len = target_lengths.dims()[0];
        self.assertions(
            batch_size,
            num_classes,
            targets.clone(),
            input_lengths_len,
            target_lengths_len,
        );
        self.length_assertions(
            input_lengths.clone(),
            target_lengths.clone(),
            max_target_len,
            max_input_length,
        );

        let mut loss = ruda_model::tensor::module::ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            self.blank,
        );

        if self.zero_infinity {
            let inf_mask = loss.clone().is_inf();
            loss = loss.clone().mask_where(inf_mask, loss.clone().zeros_like());
        }

        loss
    }

    /// Computes the CTC loss for the input log-probabilities and targets with reduction.
    ///
    /// # Arguments
    ///
    /// - `log_probs`: The log-probabilities of the outputs (e.g., from `log_softmax`).
    /// - `targets`: A 2D tensor containing the target class indices. These indices should not
    ///   include the blank index used in CTC loss. The targets are padded to the length of the longest sequence.
    /// - `input_lengths`: A 1D tensor containing the actual length of the input sequence for each batch. This
    ///   allows retrieving the actual sequence of log-probabilities from `log_probs` if the batch contains
    ///   sequences of varying lengths.
    /// - `target_lengths`: A 1D tensor containing the actual length of the target sequence for each target
    ///   sequence in `targets`.
    /// - `reduction`: The reduction stratey to apply to the loss tensor containing the CTC loss values for
    ///   each sample (e.g., mean, sum). For the mean reduction strategy, the output losses will be divided
    ///   by the target lengths and then the mean over the batch is taken. This follows PyTorch's behavior.
    ///
    /// # Returns
    ///
    /// - A 1D tensor of shape `[1]` containing the reduced loss value.
    ///
    /// # Shapes
    ///
    /// - `log_probs`: `[time_steps, batch_size, num_classes]` where `num_classes` includes blank.
    /// - `targets`: `[batch_size, max_target_length]`
    /// - `input_lengths`: `[batch_size]`
    /// - `target_lengths`: `[batch_size]`
    ///
    /// # Panics
    /// - If `reduction` is not one of `Reduction::Auto`, `Reduction::Mean`, and `Reduction::Sum`.
    /// - If `blank` index is greater than or equal to `num_classes`.
    /// - If the batch dimension of `log_probs`, `targets`, `input_lengths`, and `target_lengths` do not match.
    pub fn forward_with_reduction<B: Backend>(
        &self,
        log_probs: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let ctc_loss_tensor =
            self.forward(log_probs, targets, input_lengths, target_lengths.clone());

        match reduction {
            Reduction::Auto | Reduction::Mean => {
                // Following PyTorch's behavior where the output losses are divided
                // by the target lengths and then the mean over the batch is taken
                let target_lengths_float = target_lengths.float();
                ctc_loss_tensor.div(target_lengths_float).mean()
            }
            Reduction::Sum => ctc_loss_tensor.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Checks the per-element length invariants required by the alpha
    /// recursion. These require reading the length tensors from the device,
    /// so the checks are gated behind `cfg(debug_assertions)` to avoid the
    /// device-to-host sync in release builds.
    ///
    /// Validated:
    /// - `target_lengths[i] >= 0`
    /// - `target_lengths[i] <= max_target_len`
    /// - `input_lengths[i] >= target_lengths[i]`
    /// - `input_lengths[i] <= max_input_length`
    #[allow(unused_variables)]
    fn length_assertions<B: Backend>(
        &self,
        input_lengths: Tensor<B, 1, Int>,
        target_lengths: Tensor<B, 1, Int>,
        max_target_len: usize,
        max_input_length: usize,
    ) {
        #[cfg(debug_assertions)]
        {
            let target_lengths_data = target_lengths.into_data();
            let input_lengths_data = input_lengths.into_data();
            let target_iter = target_lengths_data.iter::<i64>();
            let input_iter = input_lengths_data.iter::<i64>();
            for (i, (tl, il)) in target_iter.zip(input_iter).enumerate() {
                assert!(tl >= 0, "target_lengths[{i}] = {tl} must be non-negative");
                assert!(
                    tl as usize <= max_target_len,
                    "target_lengths[{i}] = {tl} exceeds the targets tensor width {max_target_len}"
                );
                assert!(
                    il >= tl,
                    "input_lengths[{i}] = {il} must be >= target_lengths[{i}] = {tl} \
                     (no valid CTC alignment otherwise)"
                );
                assert!(
                    il as usize <= max_input_length,
                    "input_lengths[{i}] = {il} exceeds the log_probs time dimension \
                     {max_input_length}"
                );
            }
        }
    }

    fn assertions<B: Backend>(
        &self,
        batch_size: usize,
        num_classes: usize,
        targets: Tensor<B, 2, Int>,
        input_lengths_len: usize,
        target_lengths_len: usize,
    ) {
        assert!(
            self.blank < num_classes,
            "blank index {} must be less than num_classes {}",
            self.blank,
            num_classes
        );
        assert_eq!(
            targets.dims()[0],
            batch_size,
            "targets batch dimension {} must equal batch_size {}",
            targets.dims()[0],
            batch_size
        );
        assert_eq!(
            input_lengths_len, batch_size,
            "input_lengths length {} must equal batch_size {}",
            input_lengths_len, batch_size
        );
        assert_eq!(
            target_lengths_len, batch_size,
            "target_lengths length {} must equal batch_size {}",
            target_lengths_len, batch_size
        );
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod pytorch_comparison_tests;
