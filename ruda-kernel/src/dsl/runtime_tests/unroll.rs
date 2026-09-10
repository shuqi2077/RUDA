use crate::dsl::{as_bytes, as_type};
use crate::dsl::prelude::*;

#[ruda(launch)]
pub fn unroll_add<F: Float, N: Size>(output: &mut Array<Vector<F, N>>) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let a = Vector::<u32, Const<16>>::new(0u32);
    let b = Vector::<u32, Const<16>>::new(3u32);

    let c = a + b;

    let mut out = Vector::<u32, N>::empty();
    #[unroll]
    for i in 0..N::value() {
        out[i] = c[i];
    }

    output[0] = Vector::cast_from(out);
}

#[ruda(launch)]
pub fn unroll_load_store<F: Float, N: Size>(output: &mut Array<Vector<F, N>>) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let a = output[0];
    let b = Vector::<F, N>::new(F::from_int(3));

    let c = a + b;

    output[0] = c;
}

pub fn test_unroll_add<R: Runtime, F: Float + RudaElement>(client: ComputeClient<R>) {
    let handle = client.empty(4 * size_of::<F>());

    unroll_add::launch::<F, R>(
        &client,
        RudaCount::Static(1, 1, 1),
        RudaDim::new_1d(1),
        4,
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 4) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(3.0));
}

pub fn test_unroll_load_store<R: Runtime, F: Float + RudaElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(as_bytes!(F: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0));

    unroll_load_store::launch::<F, R>(
        &client,
        RudaCount::Static(1, 1, 1),
        RudaDim::new_1d(1),
        8,
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 8) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual, as_type!(F: 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_unroll {
    () => {
        use super::*;

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_unroll_add() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::unroll::test_unroll_add::<TestRuntime, FloatType>(client);
        }

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_unroll_load_store() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::unroll::test_unroll_load_store::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}
