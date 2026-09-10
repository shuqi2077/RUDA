use crate::dsl::prelude::*;
use crate::dsl::{debug_print};

#[ruda]
fn helper_fn<F: Float>(num: F) -> F {
    num * num
}

#[ruda(launch)]
fn simple_call_kernel<F: Float>(out: &mut Array<F>) {
    if UNIT_POS == 0 {
        out[0] = helper_fn::<F>(out[0]);
    }
}

pub fn test_simple_call<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(f32::as_bytes(&[10.0, 1.0]));

    simple_call_kernel::launch::<f32, R>(
        &client,
        RudaCount::Static(1, 1, 1),
        RudaDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 100.0);
}

#[ruda]
fn nested_helper<F: Float>(num: F) -> F {
    helper_fn::<F>(num) * num
}

#[ruda(launch)]
fn nested_call_kernel<F: Float>(out: &mut Array<F>) {
    if UNIT_POS == 0 {
        out[0] = nested_helper::<F>(out[0]);
    }
}

pub fn test_nested_call<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(f32::as_bytes(&[10.0, 1.0]));

    nested_call_kernel::launch::<f32, R>(
        &client,
        RudaCount::Static(1, 1, 1),
        RudaDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], 1000.0);
}

#[ruda(launch)]
fn debug_print_kernel<F: Float>(out: &mut Array<F>) {
    if UNIT_POS == 0 {
        let val = out[0];
        debug_print!("Test value: %f\n", val);
        out[0] = helper_fn::<F>(val);
    }
}

#[cfg(not(all(target_os = "macos")))]
pub fn test_debug_print<R: Runtime>(client: ComputeClient<R>) {
    //let logger = MemoryLogger::setup(log::Level::Info);
    let handle = client.create_from_slice(f32::as_bytes(&[10.0, 1.0]));

    debug_print_kernel::launch::<f32, R>(
        &client,
        RudaCount::Static(1, 1, 1),
        RudaDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);

    // No way to assert the log is happening right now because CUDA prints to stdout, which can't be
    // easily captured
    assert_eq!(actual[0], 100.0);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_debug {
    () => {
        use super::*;

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_simple_call_debug() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::debug::test_simple_call::<TestRuntime>(client);
        }

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_nested_call_debug() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::debug::test_nested_call::<TestRuntime>(client);
        }

        #[cfg(not(all(target_os = "macos")))]
        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_debug_print() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::debug::test_debug_print::<TestRuntime>(client);
        }
    };
}
