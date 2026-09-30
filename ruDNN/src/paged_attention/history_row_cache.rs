//! Configuration for register-local reuse in ordered history backward.
//! No tensor data or GPU state is accessed here.
use super::PagedAttentionError;

pub(super) const ENV: &str = "RUDA_PAGED_ORDERED_CACHE_ROWS";

pub(super) fn parse(value: Option<&str>) -> Result<bool, PagedAttentionError> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err(PagedAttentionError("RUDA_PAGED_ORDERED_CACHE_ROWS must be 0 or 1")),
    }
}

/// Read once, before allocating a new ordered workspace. Existing workspaces
/// keep their setting; changing the process environment cannot mutate them.
pub(super) fn from_env() -> Result<bool, PagedAttentionError> {
    let value = std::env::var_os(ENV);
    match value.as_ref() {
        None => parse(None),
        Some(value) => parse(Some(value.to_str().ok_or(PagedAttentionError(
            "RUDA_PAGED_ORDERED_CACHE_ROWS must be valid UTF-8",
        ))?)),
    }
}

/// Local array lengths for a compile-time kernel specialization. A disabled
/// array gets one element only; no history loads may access it in that mode.
/// Count is a source-level local-storage estimate, NOT hardware register usage.
pub(super) fn slots(
    enabled: bool, mla: bool, score_gradient: bool,
    key: usize, value: usize, position: usize, lanes: usize,
) -> (usize, usize, usize) {
    debug_assert!(lanes == 32 || lanes == 64);
    (
        if enabled { key.div_ceil(lanes) } else { 1 },
        if enabled && !mla && score_gradient { value.div_ceil(lanes) } else { 1 },
        if enabled && mla { position.div_ceil(lanes) } else { 1 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn default_off() { assert_eq!(parse(None), Ok(false)); }
    #[test] fn explicit_off() { assert_eq!(parse(Some("0")), Ok(false)); }
    #[test] fn explicit_on() { assert_eq!(parse(Some("1")), Ok(true)); }
    #[test] fn invalid_values_fail() {
        for v in ["", "true", "yes", "01", " 1", "1 ", "-1", "2"] {
            assert!(parse(Some(v)).is_err(), "{v:?}");
        }
    }
    #[test] fn disabled_arrays_are_placeholders() {
        assert_eq!(slots(false, false, true, 1024, 1024, 0, 32), (1, 1, 1));
    }
    #[test] fn gqa_caches_key_and_value() {
        assert_eq!(slots(true, false, true, 128, 128, 0, 32), (4, 4, 1));
    }
    #[test] fn value_only_needs_no_value_load() {
        assert_eq!(slots(true, false, false, 129, 1024, 0, 32), (5, 1, 1));
    }
    #[test] fn mla_reuses_latent_for_key_and_value() {
        assert_eq!(slots(true, true, true, 512, 512, 64, 32), (16, 1, 2));
    }
    #[test] fn tails_and_wave64() {
        assert_eq!(slots(true, true, true, 513, 513, 65, 64), (9, 1, 2));
    }
}
