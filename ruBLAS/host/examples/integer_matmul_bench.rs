//! Compare the public i64 operator with its previous strided implementation.
//! Includes tensor cloning, contiguous conversion and output allocation on both sides.
//! Run with `cargo run --release -p rublas-host --example integer_matmul_bench`.

use ruda_core::{
    bytes::Bytes,
    tensor::{DType, Shape, host::{HostTensor, Layout}},
};
use rublas_host::int_matmul;
use std::{hint::black_box, time::Instant};

// The 2-D i64 implementation from ac817b115e50b7e3ba26f61149647bdafdfd97cd.
fn baseline(lhs: HostTensor, rhs: HostTensor) -> HostTensor {
    assert_eq!(lhs.dtype(), rhs.dtype());
    assert_eq!(lhs.layout().shape().num_dims(), 2);
    assert_eq!(rhs.layout().shape().num_dims(), 2);
    assert_eq!(lhs.layout().shape()[1], rhs.layout().shape()[0]);
    let lhs = lhs.to_contiguous();
    let rhs = rhs.to_contiguous();
    let m = lhs.layout().shape()[0];
    let k = lhs.layout().shape()[1];
    let n = rhs.layout().shape()[1];
    let left: &[i64] = lhs.storage();
    let right: &[i64] = rhs.storage();
    let mut output = vec![0i64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0i64;
            for l in 0..k {
                sum = sum.wrapping_add(left[i * k + l].wrapping_mul(right[l * n + j]));
            }
            output[i * n + j] = sum;
        }
    }
    HostTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(Shape::new([m, n])),
        DType::I64,
    )
}

fn input(rows: usize, columns: usize, seed: u64) -> HostTensor {
    let mut state = seed;
    let values: Vec<i64> = (0..rows * columns)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state as i64
        })
        .collect();
    HostTensor::new(
        Bytes::from_elems(values),
        Layout::contiguous(Shape::new([rows, columns])),
        DType::I64,
    )
}

fn measure(mut operation: impl FnMut() -> HostTensor, iterations: usize) -> f64 {
    let started = Instant::now();
    for _ in 0..iterations {
        black_box(operation());
    }
    started.elapsed().as_nanos() as f64 / iterations as f64
}

fn main() {
    eprintln!("CPU public i64 matmul; same inputs; 3 warmups, 9 alternating samples; allocation included");
    eprintln!("arch={}; os={}; simd={}; rayon={}",
        std::env::consts::ARCH, std::env::consts::OS,
        cfg!(feature = "simd"), cfg!(feature = "rayon"));
    println!("m,k,n,iterations,baseline_ns_median,optimized_ns_median,speedup");
    for (m, k, n) in [
        (1, 256, 1), (8, 8, 8), (16, 64, 32), (64, 64, 64),
        (128, 128, 128), (256, 256, 256), (32, 257, 129),
    ] {
        let lhs = input(m, k, 42);
        let rhs = input(k, n, 123);
        let before = baseline(lhs.clone(), rhs.clone());
        let after = int_matmul(lhs.clone(), rhs.clone());
        assert_eq!(before.layout().shape(), after.layout().shape());
        assert_eq!(before.storage::<i64>(), after.storage::<i64>(),
            "incorrect output for [{m}, {k}] x [{k}, {n}]");
        let baseline_call = || baseline(black_box(lhs.clone()), black_box(rhs.clone()));
        let optimized_call = || int_matmul(black_box(lhs.clone()), black_box(rhs.clone()));
        for _ in 0..3 {
            black_box(baseline_call());
            black_box(optimized_call());
        }
        let iterations = (2_000_000 / (m * k * n)).clamp(1, 1000);
        let mut baseline_times = [0.0; 9];
        let mut optimized_times = [0.0; 9];
        for sample in 0..9 {
            if sample % 2 == 0 {
                baseline_times[sample] = measure(baseline_call, iterations);
                optimized_times[sample] = measure(optimized_call, iterations);
            } else {
                optimized_times[sample] = measure(optimized_call, iterations);
                baseline_times[sample] = measure(baseline_call, iterations);
            }
        }
        baseline_times.sort_by(f64::total_cmp);
        optimized_times.sort_by(f64::total_cmp);
        println!("{m},{k},{n},{iterations},{:.0},{:.0},{:.3}",
            baseline_times[4], optimized_times[4], baseline_times[4] / optimized_times[4]);
    }
}
