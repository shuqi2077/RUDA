use std::fmt::{Display, Formatter};

/// New structured errors are separate from the existing public tuple errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeErrorKind {
    InvalidInput,
    Unsupported,
    DriverIncompatible,
    OutOfMemory,
    QueueFull,
    Cancelled,
    Numerical,
    DeviceLost,
    Synchronization,
    WorkerPanicked,
    Closed,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeError {
    kind: RuntimeErrorKind,
    message: String,
}

impl RuntimeError {
    pub fn new(kind: RuntimeErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
    pub const fn kind(&self) -> RuntimeErrorKind { self.kind }
    pub fn message(&self) -> &str { &self.message }
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::new(RuntimeErrorKind::InvalidInput, message)
    }
    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(RuntimeErrorKind::Internal, message)
    }
}
impl Display for RuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for RuntimeError {}

/// Only the adapter that owns all affected streams can establish quiescence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkState {
    NotSubmitted,
    Quiescent,
    /// Includes timeouts and failed synchronization; NOT evidence of completion.
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryAction {
    FailRequest,
    RetryWithTokenLimit(usize),
    RetryPortable,
    QuarantineDevice,
}

#[derive(Debug, Clone, Copy)]
pub struct RecoveryPolicy {
    pub max_retries: usize,
    pub allow_portable: bool,
}
impl Default for RecoveryPolicy {
    fn default() -> Self { Self { max_retries: 2, allow_portable: true } }
}
impl RecoveryPolicy {
    /// `attempts` counts completed retries, not the initial attempt. This only
    /// returns a decision: it never resets a context or replays emitted tokens.
    /// `committed_cache_intact` must be false when earlier KV entries may have
    /// been overwritten. An append-only failed tail may be safely recomputed.
    pub fn decide(
        self,
        error: RuntimeErrorKind,
        work: WorkState,
        committed_cache_intact: bool,
        attempts: usize,
        token_limit: usize,
        minimum_tokens: usize,
        already_portable: bool,
    ) -> RecoveryAction {
        if work == WorkState::Unknown || !committed_cache_intact || matches!(
            error,
            RuntimeErrorKind::DeviceLost | RuntimeErrorKind::Synchronization
                | RuntimeErrorKind::WorkerPanicked | RuntimeErrorKind::Internal
        ) {
            return RecoveryAction::QuarantineDevice;
        }
        if attempts >= self.max_retries { return RecoveryAction::FailRequest; }
        match error {
            RuntimeErrorKind::OutOfMemory
                if minimum_tokens > 0 && token_limit > minimum_tokens => {
                    RecoveryAction::RetryWithTokenLimit((token_limit / 2).max(minimum_tokens))
                }
            RuntimeErrorKind::Unsupported if self.allow_portable && !already_portable => {
                RecoveryAction::RetryPortable
            }
            _ => RecoveryAction::FailRequest,
        }
    }
}
