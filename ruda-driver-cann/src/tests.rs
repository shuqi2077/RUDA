use super::*;
use std::ffi::{c_int, c_void};
use std::mem::{align_of, size_of};

#[test]
fn device_ids_preserve_the_acl_signed_range() {
    for id in [0, 1, i32::MAX as u32] {
        let device = CannDevice::new(id).unwrap();
        assert_eq!(device.ordinal(), id);
        assert_eq!(device.acl_id(), id as i32);
    }
    for id in [i32::MAX as u32 + 1, u32::MAX] {
        assert_eq!(CannDevice::new(id), Err(CannError::InvalidDevice(id)));
    }
}

#[test]
fn unknown_error_codes_are_preserved() {
    assert_eq!(check_status("aclInit", 0), Ok(()));
    for code in [-1, 1, 100000, i32::MIN, i32::MAX] {
        let error = check_status("aclInit", code).unwrap_err();
        assert_eq!(
            error,
            CannError::Status {
                operation: "aclInit",
                code
            }
        );
        assert!(error.to_string().contains(&code.to_string()));
    }
}

#[test]
fn raw_types_match_c_abi_storage() {
    assert_eq!(size_of::<sys::AclError>(), size_of::<c_int>());
    assert_eq!(size_of::<sys::AclContext>(), size_of::<*mut c_void>());
    assert_eq!(size_of::<sys::AclStream>(), size_of::<*mut c_void>());
    assert_eq!(align_of::<sys::AclEvent>(), align_of::<*mut c_void>());
    assert_eq!(size_of::<sys::AclEventRecordedStatus>(), size_of::<c_int>());
}

#[test]
fn recorded_event_status_is_not_the_legacy_event_status() {
    assert_eq!(sys::ACL_EVENT_RECORDED_STATUS_NOT_READY, 0);
    assert_eq!(sys::ACL_EVENT_RECORDED_STATUS_COMPLETE, 1);
}

#[test]
fn missing_library_is_an_error_without_sdk_or_device() {
    let path = std::env::temp_dir()
        .join(format!("ruda-cann-missing-{}", std::process::id()))
        .join(libloading::library_filename("ascendcl"));
    assert!(!path.exists());
    // SAFETY: a nonexistent absolute path cannot execute library hooks.
    let result = unsafe { CannApi::load_from(&path) };
    assert!(matches!(result, Err(CannError::Library { .. })));
}
