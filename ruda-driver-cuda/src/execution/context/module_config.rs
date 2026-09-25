//! Pure host-side validation before handing cached PTX to the driver.
use std::ffi::{CString, c_char};

pub(super) fn validate_module(
    ptx: &[c_char], entry: &str, shared: usize, device_limit: usize,
) -> Result<(CString, i32), String> {
    if ptx.len() < 2 || ptx.last().copied() != Some(0) {
        return Err("PTX must contain nonempty text followed by a NUL terminator".into());
    }
    if ptx[..ptx.len() - 1].contains(&0) {
        return Err("PTX contains an interior NUL; refusing a truncated cached module".into());
    }
    if entry.is_empty() { return Err("PTX entrypoint cannot be empty".into()); }
    let entry = CString::new(entry).map_err(|_| "PTX entrypoint contains an interior NUL".to_string())?;
    if shared > device_limit {
        return Err(format!("PTX requests {shared} dynamic shared-memory bytes, device limit is {device_limit}"));
    }
    let shared = i32::try_from(shared).map_err(|_| "dynamic shared-memory size exceeds the driver ABI".to_string())?;
    Ok((entry, shared))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn text(bytes: &[u8]) -> Vec<c_char> { bytes.iter().map(|&v| v as c_char).collect() }
    #[test]
    fn accepts_a_terminated_module_and_zero_shared_memory() {
        let (name, size) = validate_module(&text(b".version 8.0\n\0"), "kernel", 0, 49152).unwrap();
        assert_eq!(name.to_bytes(), b"kernel"); assert_eq!(size, 0);
    }
    #[test]
    fn rejects_bad_cache_without_entering_driver() {
        for ptx in [b"".as_slice(), b"\0", b".version 8.0", b"x\0junk\0"] {
            assert!(validate_module(&text(ptx), "kernel", 0, 49152).is_err());
        }
        for entry in ["", "entry\0junk"] {
            assert!(validate_module(&text(b"x\0"), entry, 0, 49152).is_err());
        }
        assert!(validate_module(&text(b"x\0"), "kernel", 49153, 49152).is_err());
        assert!(validate_module(&text(b"x\0"), "kernel", i32::MAX as usize + 1, usize::MAX).is_err());
        assert!(validate_module(&text(b"x\0"), "kernel", 49152, 49152).is_ok());
    }
}
