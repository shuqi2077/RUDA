use super::{GenerationError, SamplingConfig};

/// Request-local, reusable storage. It has no effect on the random stream.
#[derive(Debug, Default)]
pub(super) struct SamplingWorkspace {
    probabilities: Vec<f64>,
    order: Vec<usize>,
}

impl SamplingWorkspace {
    pub(super) fn fill(
        &mut self,
        logits: &[f32],
        config: SamplingConfig,
    ) -> Result<&[f64], GenerationError> {
        config.validate()?;
        if logits.is_empty() || logits.len() - 1 > i32::MAX as usize {
            return Err(GenerationError(
                "sampling requires a nonempty i32-indexed vocabulary".into(),
            ));
        }
        let mut maximum = f32::NEG_INFINITY;
        for &value in logits {
            if value.is_nan() || value == f32::INFINITY {
                return Err(GenerationError(
                    "sampling logits contain NaN or positive infinity".into(),
                ));
            }
            maximum = maximum.max(value);
        }
        if maximum == f32::NEG_INFINITY {
            return Err(GenerationError("all sampling logits are masked".into()));
        }

        let top_k = config.top_k > 0 && config.top_k < logits.len();
        let nucleus = config.top_p < 1.0;
        // Numeric equality deliberately treats +0 and -0 as tied. The token
        // index gives a total, deterministic order after NaNs were rejected.
        let compare = |&left: &usize, &right: &usize| {
            logits[left]
                .partial_cmp(&logits[right])
                .expect("NaN logits were rejected")
                .then(left.cmp(&right))
        };
        self.order.clear();
        if top_k || nucleus {
            self.order.extend(0..logits.len());
        }
        let threshold = if top_k {
            // Select a boundary in linear time instead of sorting V tokens.
            let rank = logits.len() - config.top_k;
            self.order.select_nth_unstable_by(rank, compare);
            logits[self.order[rank]]
        } else {
            f32::NEG_INFINITY
        };

        self.probabilities.resize(logits.len(), 0.0);
        for (probability, &value) in self.probabilities.iter_mut().zip(logits) {
            *probability = if value < threshold {
                0.0
            } else {
                ((value as f64 - maximum as f64) / config.temperature).exp()
            };
        }
        // Retain the original vocabulary-order sums and CDF, so introducing
        // selection/scratch reuse does not change seeded draws.
        let total: f64 = self.probabilities.iter().sum();
        if nucleus {
            // Top-k threshold ties are all retained, not arbitrarily truncated.
            self.order.retain(|&token| logits[token] >= threshold);
            self.order.sort_unstable_by(compare);
            let mut cumulative = 0.0;
            for &token in self.order.iter().take(self.order.len() - 1) {
                cumulative += self.probabilities[token] / total;
                if cumulative <= 1.0 - config.top_p {
                    self.probabilities[token] = 0.0;
                }
            }
        }
        let retained: f64 = self.probabilities.iter().sum();
        for probability in &mut self.probabilities {
            *probability /= retained;
        }
        Ok(&self.probabilities)
    }
}
