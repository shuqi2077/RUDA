use ruda_driver_cuda::CudaRuntime;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch, address_type = "dynamic")]
fn exchange(
    input_f: &Array<f32>,
    input_u: &Array<u64>,
    output_f: &mut Array<f32>,
    output_u: &mut Array<u64>,
    #[comptime] size: usize,
) {
    let tid = UNIT_POS as usize;
    let base = RUDA_POS as usize * size;
    let mut a = SharedMemory::<f32>::new_aligned(size, 16usize);
    let mut b = SharedMemory::<u64>::new_aligned(size, 128usize);
    a[tid] = input_f[base + tid];
    b[tid] = input_u[base + tid];
    sync_ruda();
    let f = a[size - tid - 1];
    let u = b[(tid + 33) % size];
    sync_ruda();
    unsafe {
        a.free();
        b.free();
    }
    let mut c = SharedMemory::<u64>::new_aligned(size, 128usize);
    c[tid] = u + 7;
    sync_ruda();
    if base + tid < output_f.len() {
        output_f[base + tid] = f;
        output_u[base + tid] = c[size - tid - 1];
    }
}

#[ruda(launch, address_type = "dynamic")]
fn maximum_storage(output: &mut Array<u32>, #[comptime] first: usize, #[comptime] second: usize) {
    let mut a = SharedMemory::<u32>::new(first);
    let mut b = SharedMemory::<u32>::new(second);
    let mut index = UNIT_POS as usize;
    while index < first {
        a[index] = index as u32 + 1;
        index += RUDA_DIM as usize;
    }
    index = UNIT_POS as usize;
    while index < second {
        b[index] = index as u32 + 100;
        index += RUDA_DIM as usize;
    }
    sync_ruda();
    index = UNIT_POS as usize;
    while index < first + second {
        if index < first {
            output[index] = a[first - index - 1];
        } else {
            output[index] = b[second - (index - first) - 1];
        }
        index += RUDA_DIM as usize;
    }
}

fn check_maximum(client: &ComputeClient<CudaRuntime>, backend: &str, address: AddressType) {
    let bytes = client.properties().hardware.max_shared_memory_size;
    assert_eq!(bytes % 4, 0);
    let first = 24576 / 4;
    let second = bytes / 4 - first;
    let output = client.create_from_slice(u32::as_bytes(&vec![u32::MAX; bytes / 4 + 16]));
    // SAFETY: Both allocations are fully initialized before the uniform barrier;
    // the kernel accesses exactly the allocated ranges and output length.
    unsafe {
        maximum_storage::launch::<CudaRuntime>(
            client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(64),
            address,
            ArrayArg::from_raw_parts(output.clone(), bytes / 4),
            first,
            second,
        );
    }
    let result = client.read_one(output).unwrap();
    let actual = u32::from_bytes(&result);
    for i in 0..first + second {
        let expected = if i < first {
            (first - i) as u32
        } else {
            (second - (i - first) - 1) as u32 + 100
        };
        assert_eq!(actual[i], expected, "maximum shared word {i}");
    }
    assert!(actual[first + second..].iter().all(|&x| x == u32::MAX));
    println!(
        "PASS {backend} maximum shared {address:?} bytes={bytes} words={}",
        bytes / 4
    );
}

pub fn run(client: &ComputeClient<CudaRuntime>, backend: &str) {
    for address in [AddressType::U32, AddressType::U64] {
        for dim in [RudaDim::new_1d(65), RudaDim::new_3d(4, 4, 4)] {
            let size = (dim.x * dim.y * dim.z) as usize;
            for tail in [0, 3, size * 12] {
                let count = size * 12 - tail;
                let patterns = [
                    0u32, 0x80000000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc12345,
                ];
                let f: Vec<f32> = (0..count)
                    .map(|i| f32::from_bits(patterns[i % patterns.len()]))
                    .collect();
                let u: Vec<u64> = (0..count).map(|i| (1u64 << 40) + i as u64).collect();
                let fi = client.create_from_slice(f32::as_bytes(&if f.is_empty() {
                    vec![0.0]
                } else {
                    f.clone()
                }));
                let ui = client.create_from_slice(u64::as_bytes(&if u.is_empty() {
                    vec![0]
                } else {
                    u.clone()
                }));
                let fo = client.create_from_slice(f32::as_bytes(&vec![247.0; count + 16]));
                let uo = client.create_from_slice(u64::as_bytes(&vec![247; count + 16]));
                // SAFETY: Every block has exactly `size` threads, all shared accesses
                // are in range and all threads reach the barriers before storage reuse.
                unsafe {
                    exchange::launch::<CudaRuntime>(
                        client,
                        RudaCount::Static(3, 2, 2),
                        dim,
                        address,
                        ArrayArg::from_raw_parts(fi, count),
                        ArrayArg::from_raw_parts(ui, count),
                        ArrayArg::from_raw_parts(fo.clone(), count),
                        ArrayArg::from_raw_parts(uo.clone(), count),
                        size,
                    );
                }
                let fbytes = client.read_one(fo).unwrap();
                let ubytes = client.read_one(uo).unwrap();
                let actual_f = f32::from_bytes(&fbytes);
                let actual_u = u64::from_bytes(&ubytes);
                for i in 0..count {
                    let base = i / size * size;
                    let reverse = size - i % size - 1;
                    let expected_f = f.get(base + reverse).copied().unwrap_or(0.0);
                    let expected_u = u.get(base + (reverse + 33) % size).copied().unwrap_or(0) + 7;
                    assert_eq!(actual_f[i].to_bits(), expected_f.to_bits(), "f32 at {i}");
                    assert_eq!(actual_u[i], expected_u, "u64 at {i}");
                }
                assert!(actual_f[count..].iter().all(|&x| x == 247.0));
                assert!(actual_u[count..].iter().all(|&x| x == 247));
                println!("PASS {backend} shared {address:?} dim={dim:?} count={count}");
            }
        }
    }
    for address in [AddressType::U32, AddressType::U64] {
        check_maximum(client, backend, address);
    }
}

pub fn over_limit(client: &ComputeClient<CudaRuntime>, backend: &str) {
    use ruda::runtime::server::{LaunchError, ResourceLimitError, ServerError};
    let max = client.properties().hardware.max_shared_memory_size;
    let first = 24576 / 4;
    let second = max / 4 + 1 - first;
    let output = client.empty((first + second) * 4);
    // SAFETY: Output covers every word; allocation must be rejected before launch.
    unsafe {
        maximum_storage::launch::<CudaRuntime>(
            client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(64),
            AddressType::U32,
            ArrayArg::from_raw_parts(output.clone(), first + second),
            first,
            second,
        );
    }
    let error = client
        .read_one(output)
        .expect_err("over-limit shared memory must fail");
    let ServerError::ServerUnhealthy { errors, .. } = error else {
        panic!("{error:?}")
    };
    assert!(
        errors.iter().any(|error| matches!(error,
        ServerError::Launch(LaunchError::TooManyResources(ResourceLimitError::SharedMemory {
            requested, max: actual_max, ..
        })) if *requested == (first + second) * 4 && *actual_max == max)),
        "{errors:?}"
    );
    println!(
        "PASS {backend} shared limit requested={} max={max}",
        (first + second) * 4
    );
}
