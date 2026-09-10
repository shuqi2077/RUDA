pub type TestRuntime = crate::CpuRuntime;

pub use half::f16;

use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

ruda_kernel::dsl::testgen_all!(f32: [f16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
ruda_kernel::library::testgen!();
ruda_kernel::library::testgen_tensor_identity!([f16, f32, u32]);
ruda_kernel::library::testgen_quantized_view!(f32);

#[ruda(launch)]
fn barrier_smoke(out: &mut Array<f32>) {
    let barrier = barrier::Barrier::local();
    barrier.arrive_and_wait();
    if UNIT_POS == 0 {
        out[0] = 1.0;
    }
}

#[ruda(launch)]
fn sync_ruda_magic(out: &mut Array<u32>) {
    let mut mem = SharedMemory::<u32>::new(1usize);
    if UNIT_POS == 0 {
        mem[0] = 0xDEADBEEFu32;
    }
    sync_ruda();
    out[UNIT_POS as usize] = mem[0];
}

#[ruda(launch)]
fn sync_ruda_two_phase(out: &mut Array<u32>) {
    let mut mem = SharedMemory::<u32>::new(4usize);
    let idx = UNIT_POS as usize;
    mem[idx] = (idx as u32) + 1;
    sync_ruda();

    if UNIT_POS == 0 {
        let mut sum = 0u32;
        for i in 0..4 {
            sum += mem[i];
        }
        mem[0] = sum;
    }
    sync_ruda();

    out[idx] = mem[0];
}

#[ruda(launch)]
fn sync_ruda_all_reduce(out: &mut Array<u32>) {
    let mut mem = SharedMemory::<u32>::new(8usize);
    let idx = UNIT_POS as usize;
    mem[idx] = idx as u32;
    sync_ruda();

    let mut sum = 0u32;
    for i in 0..8 {
        sum += mem[i];
    }
    out[idx] = sum;
}

#[test]
fn test_barrier_smoke_cpu() {
    let client = TestRuntime::client(&Default::default());
    let out = client.empty(core::mem::size_of::<f32>());

    unsafe {
        barrier_smoke::launch::<TestRuntime>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(1),
            ArrayArg::from_raw_parts(out.clone(), 1),
        )
    }

    let bytes = client.read_one_unchecked(out);
    let actual = f32::from_bytes(&bytes);
    assert_eq!(actual[0], 1.0);
}

#[test]
fn test_sync_ruda_magic_cpu() {
    let client = TestRuntime::client(&Default::default());
    let out = client.empty(4 * core::mem::size_of::<u32>());

    unsafe {
        sync_ruda_magic::launch::<TestRuntime>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(4),
            ArrayArg::from_raw_parts(out.clone(), 4),
        )
    }

    let bytes = client.read_one_unchecked(out);
    let actual = u32::from_bytes(&bytes);
    assert_eq!(actual, &[0xDEADBEEF; 4]);
}

#[test]
fn test_sync_ruda_two_phase_cpu() {
    let client = TestRuntime::client(&Default::default());
    let out = client.empty(4 * core::mem::size_of::<u32>());

    unsafe {
        sync_ruda_two_phase::launch::<TestRuntime>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(4),
            ArrayArg::from_raw_parts(out.clone(), 4),
        )
    }

    let bytes = client.read_one_unchecked(out);
    let actual = u32::from_bytes(&bytes);
    assert_eq!(actual, &[10u32; 4]);
}

#[test]
fn test_sync_ruda_all_reduce_cpu() {
    let client = TestRuntime::client(&Default::default());
    let out = client.empty(8 * core::mem::size_of::<u32>());

    unsafe {
        sync_ruda_all_reduce::launch::<TestRuntime>(
            &client,
            RudaCount::new_single(),
            RudaDim::new_1d(8),
            ArrayArg::from_raw_parts(out.clone(), 8),
        )
    }

    let bytes = client.read_one_unchecked(out);
    let actual = u32::from_bytes(&bytes);
    assert_eq!(actual, &[28u32; 8]);
}
