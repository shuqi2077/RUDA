use core::fmt::Debug;
use core::hash;

use crate::dsl::{as_bytes};

use crate::dsl::prelude::*;

#[derive(RudaType, Clone, Hash, PartialEq, Eq, Debug)]
pub enum Operation<U: Int + hash::Hash + Eq + Debug> {
    IndexAssign(usize, U),
}

#[ruda(launch)]
pub fn kernel_const_match_simple<F: Float, U: Int + hash::Hash + Eq + Debug>(
    output: &mut Array<F>,
    #[comptime] op: Operation<U>,
) {
    match op {
        Operation::IndexAssign(index, value) => {
            output[index.runtime()] = F::cast_from(value.runtime());
        }
    };
}

pub fn test_kernel_const_match<
    R: Runtime,
    F: Float + RudaElement,
    U: Int + hash::Hash + Eq + Debug,
>(
    client: ComputeClient<R>,
) {
    // Workaround for Naga bug, remove in future wgpu version to test again
    if U::BITS == 64 {
        return;
    }

    let handle = client.create_from_slice(as_bytes![F: 0.0, 1.0]);

    let index = 1;
    let value = 5.0;

    kernel_const_match_simple::launch::<F, U, R>(
        &client,
        RudaCount::Static(1, 1, 1),
        RudaDim::new_1d(1),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 2) },
        Operation::IndexAssign(index, U::new(value as i64)),
    );

    let actual = client.read_one_unchecked(handle);
    let actual = F::from_bytes(&actual);

    assert_eq!(actual[index], F::new(value));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_const_match {
    () => {
        use super::*;

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_const_match() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::const_match::test_kernel_const_match::<
                TestRuntime,
                FloatType,
                UintType,
            >(client);
        }
    };
}
