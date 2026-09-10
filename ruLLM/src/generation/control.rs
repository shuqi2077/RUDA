use super::{GenerationError, TokenGenerationOutput};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

/// A one-shot, shareable request cancellation flag. Cancellation is observed
/// between token steps; it does not interrupt an already-running device kernel.
#[derive(Debug, Clone, Default)]
pub struct GenerationCancellation {
    cancelled: Arc<AtomicBool>,
}

impl GenerationCancellation {
    pub fn new() -> Self { Self::default() }

    pub fn cancel(&self) { self.cancelled.store(true, Ordering::Release); }

    pub fn is_cancelled(&self) -> bool { self.cancelled.load(Ordering::Acquire) }
}

/// Additive controls: existing generation configuration structs remain valid.
#[derive(Debug, Clone, Default)]
pub struct GenerationControl {
    /// Match only the generated suffix, never across the prompt boundary.
    /// Matching tokens are retained in the callback and final output. Empty
    /// patterns are rejected. Simultaneous matches choose the first pattern.
    pub stop_token_sequences: Vec<Vec<i32>>,
    pub cancellation: Option<GenerationCancellation>,
}

impl GenerationControl {
    pub fn validate(&self, vocab_size: usize) -> Result<(), GenerationError> {
        for (index, pattern) in self.stop_token_sequences.iter().enumerate() {
            if pattern.is_empty() {
                return Err(GenerationError(format!("stop sequence {index} must not be empty")));
            }
            for &token in pattern {
                if token < 0 || token as usize >= vocab_size {
                    return Err(GenerationError(format!(
                        "stop sequence {index} token {token} is outside vocabulary [0, {vocab_size})"
                    )));
                }
            }
        }
        Ok(())
    }

    pub(super) fn is_cancelled(&self) -> bool {
        self.cancellation.as_ref().is_some_and(GenerationCancellation::is_cancelled)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationFinishReason {
    MaxNewTokens,
    EosToken(i32),
    /// Zero-based index in GenerationControl::stop_token_sequences.
    StopSequence(usize),
    Cancelled,
}

/// A borrowed view, valid only for the duration of a token callback. Receiving
/// it does not copy the generated history. Token IDs are not UTF-8 fragments.
#[derive(Debug, Clone, Copy)]
pub struct GenerationEvent<'a> {
    pub token_id: i32,
    pub generated_token_ids: &'a [i32],
    /// EOS / stop sequence / length termination known before invoking callback.
    pub finish_reason: Option<GenerationFinishReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlledGenerationOutput {
    pub output: TokenGenerationOutput,
    pub finish_reason: GenerationFinishReason,
}

/// One prefix-function matcher per stop pattern. Incremental state handles
/// overlapping patterns without rescanning the entire generated history.
pub(super) struct StopSequenceMatcher<'a> {
    patterns: &'a [Vec<i32>],
    prefixes: Vec<Vec<usize>>,
    matched: Vec<usize>,
}

impl<'a> StopSequenceMatcher<'a> {
    // Call only after GenerationControl::validate has rejected empty patterns.
    pub(super) fn new(patterns: &'a [Vec<i32>]) -> Self {
        let prefixes = patterns.iter().map(|pattern| {
            let mut prefix = vec![0; pattern.len()];
            for index in 1..pattern.len() {
                let mut length = prefix[index - 1];
                while length > 0 && pattern[index] != pattern[length] {
                    length = prefix[length - 1];
                }
                if pattern[index] == pattern[length] { length += 1; }
                prefix[index] = length;
            }
            prefix
        }).collect();
        Self { patterns, prefixes, matched: vec![0; patterns.len()] }
    }

    pub(super) fn push(&mut self, token: i32) -> Option<usize> {
        let mut first_match = None;
        for (index, pattern) in self.patterns.iter().enumerate() {
            let prefix = &self.prefixes[index];
            let mut length = self.matched[index];
            while length > 0 && pattern[length] != token {
                length = prefix[length - 1];
            }
            if pattern[length] == token { length += 1; }
            if length == pattern.len() {
                first_match.get_or_insert(index);
                length = prefix[length - 1];
            }
            self.matched[index] = length;
        }
        first_match
    }
}

#[cfg(test)]
mod tests;
