use alloc::vec::Vec;

use crate::dsl::prelude::*;

#[ruda(launch, address_type = "dynamic")]
pub fn kernel_absolute_pos(output1: &mut Array<u32>) {
    if ABSOLUTE_POS >= output1.len() {
        terminate!();
    }

    output1[ABSOLUTE_POS] = ABSOLUTE_POS as u32;
}

pub fn test_kernel_topology_absolute_pos<R: Runtime>(
    client: ComputeClient<R>,
    addr_type: AddressType,
) {
    if !client.properties().supports_address(addr_type) {
        return;
    }

    let ruda_count = (3, 5, 7);
    let ruda_dim = (16, 16, 1);

    let length = ruda_count.0 * ruda_count.1 * ruda_count.2 * ruda_dim.0 * ruda_dim.1 * ruda_dim.2;
    let handle1 = client.empty(length as usize * core::mem::size_of::<u32>());

    unsafe {
        kernel_absolute_pos::launch(
            &client,
            RudaCount::Static(ruda_count.0, ruda_count.1, ruda_count.2),
            RudaDim {
                x: ruda_dim.0,
                y: ruda_dim.1,
                z: ruda_dim.2,
            },
            addr_type,
            ArrayArg::from_raw_parts(handle1.clone(), length as usize),
        )
    };

    let actual = client.read_one_unchecked(handle1);
    let actual = u32::from_bytes(&actual);
    let expect: Vec<u32> = (0..length).collect();

    assert_eq!(actual, &expect);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_topology {
    () => {
        use super::*;

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_topology_scalar() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::topology::test_kernel_topology_absolute_pos::<TestRuntime>(
                client.clone(),
                AddressType::U32,
            );
            ruda_kernel::dsl::runtime_tests::topology::test_kernel_topology_absolute_pos::<TestRuntime>(
                client,
                AddressType::U64,
            );
        }
    };
}
