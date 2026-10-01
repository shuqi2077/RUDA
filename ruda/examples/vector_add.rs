#![allow(unsafe_code)]

use ruda::prelude::*;

#[ruda(launch)]
fn add(a: &Array<f32>, b: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

fn run<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let a = client.create_from_slice(f32::as_bytes(&[1.0, 2.0, 3.0, 4.0]));
    let b = client.create_from_slice(f32::as_bytes(&[10.0, 20.0, 30.0, 40.0]));
    let output = client.empty(4 * core::mem::size_of::<f32>());
    // SAFETY: All buffers contain four f32 elements; the kernel bounds-checks each access.
    unsafe {
        add::launch::<R>(
            &client,
            RudaCount::Static(1, 1, 1),
            RudaDim::new_1d(32),
            ArrayArg::from_raw_parts(a, 4),
            ArrayArg::from_raw_parts(b, 4),
            ArrayArg::from_raw_parts(output.clone(), 4),
        );
    }
    let bytes = client.read_one(output).expect("GPU readback failed");
    let values = f32::from_bytes(&bytes);
    assert_eq!(values, &[11.0, 22.0, 33.0, 44.0]);
    println!("{}: {values:?}", R::name(&client));
}

fn main() {
    let enabled = [
        ("cuda", cfg!(feature = "cuda")),
        ("hip", cfg!(feature = "hip")),
        ("wgpu", cfg!(feature = "wgpu")),
    ];
    let backends: Vec<_> = enabled.into_iter().filter_map(|(name, on)| on.then_some(name)).collect();
    let backend = std::env::args().nth(1).unwrap_or_else(|| {
        assert_eq!(backends.len(), 1, "Select one enabled backend: {backends:?}");
        backends[0].to_owned()
    });
    match backend.as_str() {
        #[cfg(feature = "cuda")]
        "cuda" => run::<ruda::cuda::CudaRuntime>(&Default::default()),
        #[cfg(feature = "hip")]
        "hip" => run::<ruda::hip::HipRuntime>(&Default::default()),
        #[cfg(feature = "wgpu")]
        "wgpu" => run::<ruda::wgpu::WgpuRuntime>(&Default::default()),
        _ => panic!("Backend {backend:?} is not enabled; enabled backends: {backends:?}"),
    }
}
