//! Compare compact attention mask/bias preparation with the former full expansion.
//! Run: cargo run --release -p ruDNN-host --example attention_broadcast_bench
//!
//! Both sides use the same current public attention kernel. The baseline performs
//! full mask/bias expansion inside the timed call to reproduce the previous
//! preparation cost. This is a preparation comparison, not a historical kernel
//! benchmark; it includes allocations and small differences in wrapper overhead.

use ruda_core::{
    bytes::Bytes,
    tensor::{BoolStore, DType, Shape, host::{HostTensor, Layout}},
};
use rudnn_host::attention::{attention_flash, attention_naive};
use std::{hint::black_box, time::Instant};

fn f32_tensor(shape: [usize; 4], offset: usize) -> HostTensor {
    let len: usize = shape.iter().product();
    HostTensor::new(
        Bytes::from_elems((0..len).map(|i| ((i + offset) % 29) as f32 / 29.0 - 0.5).collect::<Vec<_>>()),
        Layout::contiguous(Shape::new(shape)),
        DType::F32,
    )
}

fn expand_full(tensor: HostTensor, target: [usize; 4]) -> HostTensor {
    ruprim_host::expand::expand(tensor, Shape::new(target)).to_contiguous()
}

fn measure(mut operation: impl FnMut() -> HostTensor, iterations: usize) -> f64 {
    let started = Instant::now();
    for _ in 0..iterations {
        black_box(operation());
    }
    started.elapsed().as_nanos() as f64 / iterations as f64
}

fn main() {
    eprintln!("CPU f32 attention; same public kernel, dense versus compact mask/bias preparation");
    eprintln!("Full expansion is timed; 3 warmups, 9 alternating samples; no measured speedup is assumed");
    eprintln!("arch={}; os={}; simd={}; rayon={}",
        std::env::consts::ARCH, std::env::consts::OS,
        cfg!(feature = "simd"), cfg!(feature = "rayon"));
    println!("strategy,batch,heads,seq_q,seq_kv,head_dim,iterations,dense_mask_bias_bytes,compact_mask_bias_bytes,dense_ns_median,compact_ns_median,ratio");
    for (batch, heads, seq_q, seq_kv, dim) in [
        (1, 1, 1, 65, 16),
        (2, 4, 8, 65, 16),
        (2, 8, 32, 257, 32),
    ] {
        let target = [batch, heads, seq_q, seq_kv];
        let query = f32_tensor([batch, heads, seq_q, dim], 0);
        let key = f32_tensor([batch, heads, seq_kv, dim], 7);
        let value = f32_tensor([batch, heads, seq_kv, dim], 13);
        let bias = f32_tensor([1, 1, 1, seq_kv], 3);
        let mask = HostTensor::new(
            Bytes::from_elems((0..seq_kv).map(|i| u8::from(i % 11 == 0)).collect::<Vec<_>>()),
            Layout::contiguous(Shape::new([1, 1, 1, seq_kv])),
            DType::Bool(BoolStore::Native),
        );
        for (strategy, attention_fn) in [("naive", attention_naive), ("flash", attention_flash)] {
            let dense_call = || attention_fn(
                black_box(query.clone()), black_box(key.clone()), black_box(value.clone()),
                Some(expand_full(black_box(mask.clone()), target)),
                Some(expand_full(black_box(bias.clone()), target)), Default::default(),
            );
            let compact_call = || attention_fn(
                black_box(query.clone()), black_box(key.clone()), black_box(value.clone()),
                Some(black_box(mask.clone())), Some(black_box(bias.clone())), Default::default(),
            );
            let dense = dense_call();
            let compact = compact_call();
            assert_eq!(dense.layout().shape(), compact.layout().shape());
            assert_eq!(dense.storage::<f32>(), compact.storage::<f32>(), "{strategy}: {target:?}");
            for _ in 0..3 {
                black_box(dense_call());
                black_box(compact_call());
            }
            let iterations = (1_000_000 / (batch * heads * seq_q * seq_kv * dim)).clamp(1, 100);
            let mut dense_times = [0.0; 9];
            let mut compact_times = [0.0; 9];
            for sample in 0..9 {
                if sample % 2 == 0 {
                    dense_times[sample] = measure(dense_call, iterations);
                    compact_times[sample] = measure(compact_call, iterations);
                } else {
                    compact_times[sample] = measure(compact_call, iterations);
                    dense_times[sample] = measure(dense_call, iterations);
                }
            }
            dense_times.sort_by(f64::total_cmp);
            compact_times.sort_by(f64::total_cmp);
            // Four bytes for f32 bias plus one byte for the bool mask. These are
            // the prepared buffers only, not total attention or peak memory.
            let dense_bytes = batch * heads * seq_q * seq_kv * 5;
            let compact_bytes = seq_q * seq_kv * 5;
            println!("{strategy},{batch},{heads},{seq_q},{seq_kv},{dim},{iterations},{dense_bytes},{compact_bytes},{:.0},{:.0},{:.3}",
                dense_times[4], compact_times[4], dense_times[4] / compact_times[4]);
        }
    }
}
