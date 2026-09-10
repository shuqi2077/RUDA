use ruda_driver_cuda::CudaRuntime;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch, address_type = "dynamic")]
fn tensor_metadata_and_gather(
    first: &Array<f32>,
    a: &Tensor<f32>,
    second: &Array<f32>,
    b: &Tensor<f32>,
    metadata: &mut Array<u64>,
    data: &mut Array<f32>,
    bias: f32,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos == 0 {
        metadata[0] = a.rank() as u64;
        metadata[1] = b.rank() as u64;
        metadata[2] = a.len() as u64;
        metadata[3] = b.len() as u64;
        metadata[4] = a.buffer_len() as u64;
        metadata[5] = b.buffer_len() as u64;
    }
    if pos < a.rank() {
        metadata[8 + pos * 4] = a.shape(pos) as u64;
        metadata[9 + pos * 4] = a.stride(pos) as u64;
    }
    if pos < b.rank() {
        metadata[10 + pos * 4] = b.shape(pos) as u64;
        metadata[11 + pos * 4] = b.stride(pos) as u64;
    }
    if pos < a.len() {
        let mut offset = 0usize;
        let mut remaining = pos;
        let mut dim = a.rank();
        while dim > 0 {
            dim -= 1;
            offset += (remaining % a.shape(dim)) * a.stride(dim);
            remaining /= a.shape(dim);
        }
        data[pos] = a[offset] + bias + first[0] + second[0];
    }
}

fn buffer_len(shape: &[usize], strides: &[usize]) -> usize {
    if shape.contains(&0) {
        return 1;
    }
    1 + shape
        .iter()
        .zip(strides)
        .map(|(&size, &stride)| (size - 1) * stride)
        .sum::<usize>()
}

fn check_layout(
    client: &ComputeClient<CudaRuntime>,
    backend: &str,
    address: AddressType,
    shape_a: &[usize],
    strides_a: &[usize],
    shape_b: &[usize],
    strides_b: &[usize],
    offset_view: bool,
) {
    let len_a = shape_a.iter().product::<usize>();
    let len_b = shape_b.iter().product::<usize>();
    let physical_a = buffer_len(shape_a, strides_a);
    let physical_b = buffer_len(shape_b, strides_b);
    let input: Vec<f32> = (0..physical_a).map(|i| i as f32 * 0.5).collect();
    let a = if offset_view {
        let mut padded = vec![-999.0; 4];
        padded.extend_from_slice(&input);
        padded.extend_from_slice(&[-999.0; 4]);
        client
            .create_from_slice(f32::as_bytes(&padded))
            .offset_start(16)
            .offset_end(16)
    } else {
        client.create_from_slice(f32::as_bytes(&input))
    };
    let b = client.create_from_slice(f32::as_bytes(&vec![0.0; physical_b]));
    let first = client.create_from_slice(f32::as_bytes(&[2.0]));
    let second = client.create_from_slice(f32::as_bytes(&[3.0]));
    let meta_len = 8 + 4 * shape_a.len().max(shape_b.len());
    let meta = client.create_from_slice(u64::as_bytes(&vec![u64::MAX; meta_len + 16]));
    let data = client.create_from_slice(f32::as_bytes(&vec![-247.0; len_a.max(1) + 16]));
    let count = len_a.max(shape_a.len()).max(shape_b.len()).max(1);
    // SAFETY: The backing buffers cover every address of each strided layout,
    // including the offset view. Only in-rank metadata and in-length data are read.
    unsafe {
        tensor_metadata_and_gather::launch::<CudaRuntime>(
            client,
            RudaCount::Static(count.div_ceil(64) as u32, 1, 1),
            RudaDim::new_1d(64),
            address,
            ArrayArg::from_raw_parts(first, 1),
            TensorArg::from_raw_parts(a, strides_a.to_vec().into(), shape_a.to_vec().into()),
            ArrayArg::from_raw_parts(second, 1),
            TensorArg::from_raw_parts(b, strides_b.to_vec().into(), shape_b.to_vec().into()),
            ArrayArg::from_raw_parts(meta.clone(), meta_len),
            ArrayArg::from_raw_parts(data.clone(), len_a.max(1)),
            1.25f32,
        );
    }
    let bytes = client.read_one(meta).unwrap();
    let actual = u64::from_bytes(&bytes);
    let mut expected = vec![u64::MAX; meta_len + 16];
    expected[..6].copy_from_slice(&[
        shape_a.len() as u64,
        shape_b.len() as u64,
        len_a as u64,
        len_b as u64,
        physical_a as u64,
        physical_b as u64,
    ]);
    for (dim, (&shape, &stride)) in shape_a.iter().zip(strides_a).enumerate() {
        expected[8 + dim * 4] = shape as u64;
        expected[9 + dim * 4] = stride as u64;
    }
    for (dim, (&shape, &stride)) in shape_b.iter().zip(strides_b).enumerate() {
        expected[10 + dim * 4] = shape as u64;
        expected[11 + dim * 4] = stride as u64;
    }
    assert_eq!(actual, expected);
    let bytes = client.read_one(data).unwrap();
    let actual = f32::from_bytes(&bytes);
    for index in 0..len_a {
        let mut rest = index;
        let mut offset = 0;
        for (&size, &stride) in shape_a.iter().zip(strides_a).rev() {
            offset += (rest % size) * stride;
            rest /= size;
        }
        assert_eq!(actual[index].to_bits(), (input[offset] + 6.25).to_bits());
    }
    assert!(
        actual[len_a..len_a.max(1) + 16]
            .iter()
            .all(|&value| value == -247.0)
    );
    println!(
        "PASS {backend} tensor {address:?} a={shape_a:?}/{strides_a:?} b={shape_b:?}/{strides_b:?} offset={offset_view}"
    );
}

pub fn run(client: &ComputeClient<CudaRuntime>, backend: &str) {
    use ruda_kernel::dsl::runtime_tests::metadata;
    for address in [AddressType::U32, AddressType::U64] {
        assert!(client.properties().supports_address(address));
        metadata::test_shape_dim_4::<CudaRuntime>(client.clone(), address);
        metadata::test_shape_different_ranks::<CudaRuntime>(client.clone(), address);
        metadata::test_stride_different_ranks::<CudaRuntime>(client.clone(), address);
        metadata::test_len_different_ranks::<CudaRuntime>(client.clone(), address);
        metadata::test_buffer_len_discontiguous::<CudaRuntime>(client.clone(), address);
        println!("PASS {backend} 5 existing scalar metadata tests {address:?}");
        check_layout(
            client,
            backend,
            address,
            &[3, 5],
            &[5, 1],
            &[2, 3, 4],
            &[12, 4, 1],
            false,
        );
        check_layout(client, backend, address, &[5, 3], &[1, 5], &[4], &[2], true);
        check_layout(
            client,
            backend,
            address,
            &[1, 7],
            &[0, 1],
            &[2, 1, 3],
            &[3, 0, 1],
            false,
        );
        check_layout(
            client,
            backend,
            address,
            &[2, 0, 3],
            &[0, 3, 1],
            &[1],
            &[1],
            false,
        );
        check_layout(
            client,
            backend,
            address,
            &[2, 3],
            &[0, 1],
            &[3],
            &[1],
            false,
        );
        if backend == "ptx" {
            check_layout(client, backend, address, &[], &[], &[], &[], false);
        }
    }
    check_layout(
        client,
        backend,
        AddressType::U64,
        &[1, 3],
        &[1usize << 33, 1],
        &[1, 1],
        &[1usize << 34, 1],
        false,
    );
}
