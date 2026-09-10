//! Ownership of an NVRTC program and the source storage it may reference.

use cudarc::nvrtc::{result, sys};
use std::ffi::CString;

type DestroyProgram = unsafe fn(*mut sys::nvrtcProgram) -> sys::nvrtcResult;

/// Owns a program from successful creation until all PTX/log extraction is done.
/// Not Clone: exactly one owner is responsible for attempting destruction.
pub(super) struct NvrtcProgram {
    handle: sys::nvrtcProgram,
    // NVRTC may retain references into this allocation. Fields are dropped only
    // after Drop::drop, so the source remains alive during nvrtcDestroyProgram.
    _source: CString,
    destroy: DestroyProgram,
}

unsafe fn destroy_program(program: *mut sys::nvrtcProgram) -> sys::nvrtcResult {
    // SAFETY: The owner calls this once, with its live program handle.
    unsafe { sys::nvrtcDestroyProgram(program) }
}

impl NvrtcProgram {
    pub(super) fn new(source: CString) -> Result<Self, result::NvrtcError> {
        let handle = result::create_program(source.as_c_str(), None)?;
        Ok(Self { handle, _source: source, destroy: destroy_program })
    }

    /// The borrowed raw handle must not be destroyed or retained after this owner.
    pub(super) fn raw(&self) -> sys::nvrtcProgram {
        self.handle
    }
}

impl Drop for NvrtcProgram {
    fn drop(&mut self) {
        // SAFETY: The constructor creates exactly one owned handle. The source
        // is still alive and no manual destroy is exposed. This also runs on
        // early Result returns and unwinding (but not process abort).
        let status = unsafe { (self.destroy)(&mut self.handle) };
        if status != sys::nvrtcResult::NVRTC_SUCCESS {
            // Never mask the original compilation error with a Drop panic.
            log::warn!("Unable to destroy NVRTC program: {status:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, ffi::CStr};

    std::thread_local! {
        static DESTROYS: Cell<usize> = const { Cell::new(0) };
    }

    unsafe fn mock_destroy(program: *mut sys::nvrtcProgram) -> sys::nvrtcResult {
        // SAFETY: mock_program uses the source's live CString pointer as the
        // mock opaque handle. This test intentionally checks its Drop lifetime.
        unsafe {
            assert_eq!(CStr::from_ptr((*program).cast()).to_bytes(), b"mock source");
            *program = core::ptr::null_mut();
        }
        DESTROYS.with(|count| count.set(count.get() + 1));
        sys::nvrtcResult::NVRTC_SUCCESS
    }

    fn mock_program() -> NvrtcProgram {
        let source = CString::new("mock source").unwrap();
        NvrtcProgram {
            handle: source.as_ptr().cast_mut().cast(),
            _source: source,
            destroy: mock_destroy,
        }
    }

    #[test]
    fn owner_destroys_once_after_move() {
        DESTROYS.with(|count| count.set(0));
        let program = mock_program();
        let moved = Some(program);
        drop(moved);
        DESTROYS.with(|count| assert_eq!(count.get(), 1));
    }

    #[test]
    fn owner_destroys_on_early_error() {
        fn fail_after_creation() -> Result<(), &'static str> {
            let _program = mock_program();
            Err("compile/log/PTX extraction failed")
        }
        DESTROYS.with(|count| count.set(0));
        assert!(fail_after_creation().is_err());
        DESTROYS.with(|count| assert_eq!(count.get(), 1));
    }

    #[test]
    #[cfg(panic = "unwind")]
    fn owner_destroys_during_unwind() {
        DESTROYS.with(|count| count.set(0));
        let result = std::panic::catch_unwind(|| {
            let _program = mock_program();
            panic!("error after creation");
        });
        assert!(result.is_err());
        DESTROYS.with(|count| assert_eq!(count.get(), 1));
    }

    #[test]
    #[ignore = "requires the NVRTC shared library; not part of CPU-only tests"]
    fn nvrtc_program_smoke_success_and_failure() {
        let options: &[&str] = &["--std=c++17"];
        for _ in 0..8 {
            let program = NvrtcProgram::new(
                CString::new("extern \"C\" __global__ void ruda_test() {}").unwrap(),
            ).unwrap();
            // SAFETY: program owns a valid NVRTC handle throughout both calls.
            unsafe {
                result::compile_program(program.raw(), options).unwrap();
                assert!(!result::get_ptx(program.raw()).unwrap().is_empty());
            }
            drop(program);

            let invalid = NvrtcProgram::new(CString::new("not valid CUDA;").unwrap()).unwrap();
            // SAFETY: The invalid source still belongs to a valid program.
            unsafe {
                assert!(result::compile_program(invalid.raw(), options).is_err());
                assert!(!result::get_program_log(invalid.raw()).unwrap().is_empty());
            }
        }
    }
}
