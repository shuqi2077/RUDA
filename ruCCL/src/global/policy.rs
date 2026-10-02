//! Bounded control/data deadlines for the WebSocket collective API.
use std::time::Duration;

/// Startup retry is safe only before a request is submitted. Failed collectives
/// are never replayed: discard the communicator and register a fresh group.
#[derive(Debug, Clone, Copy)]
pub struct GlobalFailurePolicy {
    /// Maximum connection attempts per control channel, including the first.
    pub connect_attempts: u32,
    /// Per-attempt connection and handshake limit.
    pub connect_timeout: Duration,
    /// Initial retry backoff, doubled with a five-second cap.
    pub retry_backoff: Duration,
    /// Total bound for a queued control request and its reply.
    pub request_timeout: Duration,
    /// Total bound for a collective, including local serialization and sync.
    pub collective_timeout: Duration,
}
impl Default for GlobalFailurePolicy {
    fn default() -> Self {
        Self { connect_attempts: 5, connect_timeout: Duration::from_secs(5),
            retry_backoff: Duration::from_millis(100), request_timeout: Duration::from_secs(30),
            collective_timeout: Duration::from_secs(300) }
    }
}
impl GlobalFailurePolicy {
    /// Environment overrides are validated instead of enabling infinite waits.
    pub fn from_environment() -> Result<Self, String> {
        fn number(name: &str, default: u64) -> Result<u64, String> {
            match std::env::var(name) {
                Ok(value) => value.parse::<u64>().map_err(|_| format!("{name} must be a positive integer")),
                Err(std::env::VarError::NotPresent) => Ok(default),
                Err(_) => Err(format!("{name} must be Unicode")),
            }
        }
        let defaults = Self::default();
        let attempts = number("RUCCL_CONNECT_ATTEMPTS", defaults.connect_attempts as u64)?;
        if !(1..=64).contains(&attempts) { return Err("RUCCL_CONNECT_ATTEMPTS must be 1..64".into()); }
        let duration = |name, default: Duration| -> Result<Duration, String> {
            let value = number(name, default.as_millis() as u64)?;
            if value == 0 || value > 86_400_000 { return Err(format!("{name} must be 1..86400000 ms")); }
            Ok(Duration::from_millis(value))
        };
        Ok(Self {
            connect_attempts: attempts as u32,
            connect_timeout: duration("RUCCL_CONNECT_TIMEOUT_MS", defaults.connect_timeout)?,
            retry_backoff: duration("RUCCL_RETRY_BACKOFF_MS", defaults.retry_backoff)?,
            request_timeout: duration("RUCCL_REQUEST_TIMEOUT_MS", defaults.request_timeout)?,
            collective_timeout: duration("RUCCL_COLLECTIVE_TIMEOUT_MS", defaults.collective_timeout)?,
        })
    }
    pub(crate) fn backoff(self, attempt: u32) -> Duration {
        self.retry_backoff.saturating_mul(1u32 << attempt.min(16)).min(Duration::from_secs(5))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn backoff_is_bounded() {
        let p = GlobalFailurePolicy::default();
        assert_eq!(p.backoff(0), Duration::from_millis(100));
        assert_eq!(p.backoff(1), Duration::from_millis(200));
        assert_eq!(p.backoff(u32::MAX), Duration::from_secs(5));
    }
}
