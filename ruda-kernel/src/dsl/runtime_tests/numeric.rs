use crate::dsl::prelude::*;
use ruda_core::ir::{ElemType, FloatKind, UIntKind};

#[ruda(launch)]
pub fn kernel_define<N: Numeric>(array: &mut Array<N>, #[define(N)] _elem: ElemType) {
    array[UNIT_POS as usize] += N::cast_from(5.0f32);
}

#[ruda(launch)]
pub fn kernel_define_many<N: Numeric, N2: Numeric>(
    array: &mut Array<N>,
    second: Array<N2>,
    #[define(N, N2)] _defines: [ElemType; 2],
) {
    array[UNIT_POS as usize] += N::cast_from(second[UNIT_POS as usize]);
}

pub fn test_kernel_define<R: Runtime>(client: ComputeClient<R>) {
    let handle = client.create_from_slice(f32::as_bytes(&[f32::new(0.0), f32::new(1.0)]));

    let elem = ElemType::Float(FloatKind::F32);

    kernel_define::launch(
        &client,
        RudaCount::Static(1, 1, 1),
        RudaDim::new_1d(2),
        unsafe { ArrayArg::from_raw_parts(handle.clone(), 2) },
        elem,
    );

    let actual = client.read_one_unchecked(handle);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(5.0));
    assert_eq!(actual[1], f32::new(6.0));
}

pub fn test_kernel_define_many<R: Runtime>(client: ComputeClient<R>) {
    let first = client.create_from_slice(f32::as_bytes(&[f32::new(0.0), f32::new(1.0)]));
    let second = client.create_from_slice(u32::as_bytes(&[u32::new(5), u32::new(6)]));

    let elem_first = ElemType::Float(FloatKind::F32);
    let elem_second = ElemType::UInt(UIntKind::U32);

    kernel_define_many::launch(
        &client,
        RudaCount::Static(1, 1, 1),
        RudaDim::new_1d(2),
        unsafe { ArrayArg::from_raw_parts(first.clone(), 2) },
        unsafe { ArrayArg::from_raw_parts(second.clone(), 2) },
        [elem_first, elem_second],
    );

    let actual = client.read_one_unchecked(first);
    let actual = f32::from_bytes(&actual);

    assert_eq!(actual[0], f32::new(5.0));
    assert_eq!(actual[1], f32::new(7.0));
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_numeric {
    () => {
        use super::*;
        use ruda_kernel::dsl::prelude::*;

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_kernel_define() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::numeric::test_kernel_define::<TestRuntime>(client);
        }

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_kernel_define_many() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::numeric::test_kernel_define_many::<TestRuntime>(client);
        }
    };
}
