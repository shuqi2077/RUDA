use crate::dsl::{as_bytes};
use crate::dsl::prelude::*;

#[ruda(launch)]
pub fn sequence_for_loop<F: Float>(output: &mut Array<F>) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let mut sequence = Sequence::<F>::new();
    sequence.push(F::new(1.0));
    sequence.push(F::new(4.0));

    for value in sequence {
        output[0] += value;
    }
}

#[ruda(launch)]
pub fn sequence_index<F: Float>(output: &mut Array<F>) {
    if UNIT_POS != 0 {
        terminate!();
    }

    let mut sequence = Sequence::<F>::new();
    sequence.push(F::new(2.0));
    sequence.push(F::new(4.0));

    output[0] += sequence[0];
    output[0] += *Sequence::<F>::index(&sequence, 1usize);
}

pub fn test_sequence_for_loop<R: Runtime, F: Float + RudaElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(as_bytes![F: 0.0]);

    sequence_for_loop::launch::<F, R>(
        &client,
        RudaCount::Static(1, 1, 1),
        RudaDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(5.0));
}

pub fn test_sequence_index<R: Runtime, F: Float + RudaElement>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(as_bytes![F: 0.0]);

    sequence_index::launch::<F, R>(
        &client,
        RudaCount::Static(1, 1, 1),
        RudaDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 2) },
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[0], F::new(6.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_sequence {
    () => {
        use super::*;

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_sequence_for_loop() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::sequence::test_sequence_for_loop::<TestRuntime, FloatType>(
                client,
            );
        }

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_sequence_index() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::sequence::test_sequence_index::<TestRuntime, FloatType>(
                client,
            );
        }
    };
}
