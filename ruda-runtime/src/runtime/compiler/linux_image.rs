//! Identify the image containing the runtime without walking unrelated ELF
//! program headers. Some injected GPU/shared objects use load biases whose
//! addition overflows buildid's debug-mode dl_iterate_phdr callback; unwinding
//! across that C callback aborts the entire process.
use core::hash::Hasher;
use ruda_core::hash::{StableHash, StableHasher};
use std::{
    ffi::{CStr, OsStr},
    io::Read,
    os::unix::ffi::OsStrExt,
    path::PathBuf,
    sync::OnceLock,
};

pub(super) fn build_hash() -> StableHash {
    static HASH: OnceLock<StableHash> = OnceLock::new();
    *HASH.get_or_init(|| {
        let result=module_path().ok_or_else(||std::io::Error::other("dladdr did not identify the runtime image"))
            .and_then(std::fs::File::open).and_then(hash_image);
        result.unwrap_or_else(|error| {
            // An unreadable/deleted image must not cause cache hits from a
            // different executable. Keep an instance-local identity instead.
            log::warn!("Unable to fingerprint runtime image; using process-local compilation cache identity: {error}");
            StableHasher::hash_one(&("ruda-unreadable-image-v1",std::process::id(),
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos(),
                build_hash as *const () as usize))
        })
    })
}

fn module_path() -> Option<PathBuf> {
    let mut info = core::mem::MaybeUninit::<libc::Dl_info>::uninit();
    // SAFETY: dladdr initializes Dl_info on success. The address belongs to
    // this live module, and the loader-owned filename is copied before return.
    let result = unsafe {
        libc::dladdr(
            build_hash as *const () as *const libc::c_void,
            info.as_mut_ptr(),
        )
    };
    if result == 0 {
        return None;
    }
    let info = unsafe { info.assume_init() };
    if info.dli_fname.is_null() {
        return None;
    }
    let bytes = unsafe { CStr::from_ptr(info.dli_fname) }.to_bytes();
    Some(PathBuf::from(OsStr::from_bytes(bytes)))
}

fn hash_image(mut reader: impl Read) -> std::io::Result<StableHash> {
    let mut hasher = StableHasher::new();
    hasher.write(b"ruda-linux-image-v1\0");
    let mut buffer = [0u8; 64 * 1024];
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => return Ok(hasher.finalize()),
            Ok(n) => hasher.write(&buffer[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identifies_the_own_image_and_memoizes_its_contents() {
        let path = module_path().expect("own runtime image");
        let expected = hash_image(std::fs::File::open(path).unwrap()).unwrap();
        assert_eq!(build_hash(), expected);
        assert_eq!(build_hash(), expected);
    }
    #[test]
    fn changed_image_bytes_invalidate_the_identity() {
        let a = hash_image(&b"kernel body A"[..]).unwrap();
        let b = hash_image(&b"kernel body B"[..]).unwrap();
        assert_ne!(a, b);
        assert_eq!(a, hash_image(&b"kernel body A"[..]).unwrap());
    }
}
