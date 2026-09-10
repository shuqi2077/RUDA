use crate::dsl::{as_bytes, as_type};
use crate::dsl::prelude::*;

#[ruda(launch)]
pub fn slice_select<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if UNIT_POS == 0 {
        let slice = input.slice(2, 3);
        output[0] = slice[0];
    }
}

#[ruda(launch)]
pub fn slice_len<F: Float>(input: &Array<F>, output: &mut Array<u32>) {
    if UNIT_POS == 0 {
        let slice = input.slice(2, 4);
        output[0] = slice.len() as u32;
    }
}

#[ruda(launch)]
pub fn slice_for<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if UNIT_POS == 0 {
        let mut sum = F::new(0.0);

        for item in input.slice(2, 4) {
            sum += item;
        }

        output[0] = sum;
    }
}

#[ruda(launch)]
pub fn slice_mut_assign<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if UNIT_POS == 0 {
        let slice_1 = &mut output.slice_mut(2, 3);
        slice_1[0] = input[0];
    }
}

#[ruda(launch)]
pub fn slice_mut_len(output: &mut Array<u32>) {
    if UNIT_POS == 0 {
        let slice = output.slice_mut(0, 2).into_vectorized();
        output[0] = slice.len() as u32;
    }
}

pub fn test_slice_select<R: Runtime, F: Float + RudaElement>(client: ComputeClient<R>) {
    let input = client.create_from_slice(as_bytes![F: 0.0, 1.0, 2.0, 3.0, 4.0]);
    let output = client.empty(core::mem::size_of::<F>());

    unsafe {
        slice_select::launch::<F, R>(
            &client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(1),
            ArrayArg::from_raw_parts(input, 5),
            ArrayArg::from_raw_parts(output.clone(), 1),
        )
    };

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(2.0));
}

pub fn test_slice_len<R: Runtime, F: Float + RudaElement>(client: ComputeClient<R>) {
    let input = client.create_from_slice(as_bytes![F: 0.0, 1.0, 2.0, 3.0, 4.0]);
    let output = client.empty(core::mem::size_of::<u32>());

    unsafe {
        slice_len::launch::<F, R>(
            &client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(1),
            ArrayArg::from_raw_parts(input, 5),
            ArrayArg::from_raw_parts(output.clone(), 1),
        )
    };

    let actual = client.read_one_unchecked(output);
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual, &[2]);
}

pub fn test_slice_for<R: Runtime, F: Float + RudaElement>(client: ComputeClient<R>) {
    let input = client.create_from_slice(as_bytes![F: 0.0, 1.0, 2.0, 3.0, 4.0]);
    let output = client.create_from_slice(as_bytes![F: 0.0]);

    unsafe {
        slice_for::launch::<F, R>(
            &client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(1),
            ArrayArg::from_raw_parts(input, 5),
            ArrayArg::from_raw_parts(output.clone(), 1),
        )
    };

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_slice_mut_assign<R: Runtime, F: Float + RudaElement>(client: ComputeClient<R>) {
    let input = client.create_from_slice(as_bytes![F: 15.0]);
    let output = client.create_from_slice(as_bytes![F: 0.0, 1.0, 2.0, 3.0, 4.0]);

    unsafe {
        slice_mut_assign::launch::<F, R>(
            &client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(1),
            ArrayArg::from_raw_parts(input, 5),
            ArrayArg::from_raw_parts(output.clone(), 1),
        )
    };

    let actual = client.read_one_unchecked(output);
    let actual = F::from_bytes(&actual);

    assert_eq!(&actual[0..5], as_type![F: 0.0, 1.0, 15.0, 3.0, 4.0]);
}

pub fn test_slice_mut_len<R: Runtime>(client: ComputeClient<R>) {
    let output = client.empty(core::mem::size_of::<u32>() * 4);

    unsafe {
        slice_mut_len::launch(
            &client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(1),
            ArrayArg::from_raw_parts(output.clone(), 4),
        )
    };

    let actual = client.read_one_unchecked(output);
    let actual = u32::from_bytes(&actual);

    assert_eq!(actual[0], 2);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_slice {
    () => {
        use super::*;

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_slice_select() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::slice::test_slice_select::<TestRuntime, FloatType>(client);
        }

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_slice_len() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::slice::test_slice_len::<TestRuntime, FloatType>(client);
        }

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_slice_for() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::slice::test_slice_for::<TestRuntime, FloatType>(client);
        }

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_slice_mut_assign() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::slice::test_slice_mut_assign::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_slice_mut_len() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::slice::test_slice_mut_len::<TestRuntime>(client);
        }
    };
}
