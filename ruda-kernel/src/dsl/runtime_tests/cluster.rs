use alloc::{vec, vec::Vec};

use crate::dsl::prelude::*;

#[ruda(launch, cluster_dim = RudaDim::new_3d(1, 2, 3))]
fn cluster_meta_kernel(out: &mut Array<u32>) {
    if UNIT_POS == 0 {
        if RUDA_POS == 0 {
            out[0] = RUDA_CLUSTER_DIM;
            out[1] = RUDA_CLUSTER_DIM_X;
            out[2] = RUDA_CLUSTER_DIM_Y;
            out[3] = RUDA_CLUSTER_DIM_Z;
        }

        let offset = RUDA_POS * 4 + 4;

        out[offset] = RUDA_POS_CLUSTER;
        out[offset + 1] = RUDA_POS_CLUSTER_X;
        out[offset + 2] = RUDA_POS_CLUSTER_Y;
        out[offset + 3] = RUDA_POS_CLUSTER_Z;
    }
}

pub fn test_cluster_meta<R: Runtime>(client: ComputeClient<R>) {
    if !client.features().ruda_cluster {
        return;
    }

    let cluster_dim_x = 1;
    let cluster_dim_y = 2;
    let cluster_dim_z = 3;

    let ruda_count_x = 2;
    let ruda_count_y = 2;
    let ruda_count_z = 6;
    let ruda_count = RudaCount::new_3d(ruda_count_x, ruda_count_y, ruda_count_z);
    let num_rudas = ruda_count_x * ruda_count_y * ruda_count_z;

    let handle = client.empty((num_rudas as usize * 4 + 4) * size_of::<u32>());

    cluster_meta_kernel::launch(&client, ruda_count, RudaDim::new_single(), unsafe {
        ArrayArg::from_raw_parts(handle.clone(), num_rudas as usize * 8)
    });

    let actual = client.read_one_unchecked(handle);
    let actual = u32::from_bytes(&actual);

    let mut expected: Vec<u32> = vec![6, 1, 2, 3];
    for z in 0..ruda_count_z {
        for y in 0..ruda_count_y {
            for x in 0..ruda_count_x {
                let rank_x = x % cluster_dim_x;
                let rank_y = y % cluster_dim_y;
                let rank_z = z % cluster_dim_z;
                let rank_abs = rank_z * cluster_dim_y + rank_y * cluster_dim_x + rank_x;
                expected.extend([rank_abs, rank_x, rank_y, rank_z]);
            }
        }
    }

    assert_eq!(actual, &expected);
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_cluster {
    () => {
        use super::*;

        #[$crate::dsl::runtime_tests::test_log::test]
        fn test_cluster_meta() {
            let client = TestRuntime::client(&Default::default());
            ruda_kernel::dsl::runtime_tests::cluster::test_cluster_meta::<TestRuntime>(client);
        }
    };
}
