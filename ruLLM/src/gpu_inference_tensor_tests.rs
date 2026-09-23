//! Tensor-level regression checks. These tests use the existing Host test
//! dependency; they are NOT CUDA execution or performance measurements.
use super::*;
use ruda_tensor::api::activation::softmax;
use ruda_tensor_host::{Host, HostDevice};

fn sample(shape: [usize; 4], seed: usize, transposed: bool, dtype: DType) -> Tensor<Host, 4> {
    let [batch, heads, length, dim] = shape;
    let physical = if transposed { [batch, length, heads, dim] } else { shape };
    let count = batch * heads * length * dim;
    let values = (0..count)
        .map(|i| (((i * (seed * 2 + 1) + seed) % 67) as f32 - 33.0) / 64.0)
        .collect::<Vec<_>>();
    let tensor = Tensor::<Host, 4>::from_data(
        TensorData::new(values, physical), (&HostDevice, dtype),
    );
    if transposed { tensor.swap_dims(1, 2) } else { tensor }
}

fn close(actual: Tensor<Host, 4>, expected: Tensor<Host, 4>, tolerance: f32) {
    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(actual.dtype(), expected.dtype());
    let actual = actual.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
    let expected = expected.cast(DType::F32).into_data().to_vec::<f32>().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (index, (a, b)) in actual.into_iter().zip(expected).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "nonfinite result at {index}");
        assert!((a - b).abs() <= tolerance + tolerance * b.abs(),
            "element {index}: actual={a}, expected={b}, tolerance={tolerance}");
    }
}

#[test]
fn grouped_query_and_value_products_match_existing_repeat_heads() {
    // Independent head values also detect an incorrect interleaved head mapping.
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let tolerance = match dtype { DType::F32 => 2e-5, _ => 0.04 };
        for (batch, kv_heads, groups, sequence, length) in [
            (1, 2, 1, 1, 17), (1, 2, 4, 1, 33), (2, 2, 3, 5, 9),
            (1, 1, 8, 7, 7), (2, 3, 2, 3, 31),
        ] {
            for transposed in [false, true] {
                let heads = kv_heads * groups;
                let query = sample([batch, heads, sequence, 16], 3, transposed, dtype);
                let key = sample([batch, kv_heads, length, 16], 5, transposed, dtype);
                let value = sample([batch, kv_heads, length, 16], 11, transposed, dtype);
                let ops = shared_tensor_ops::<Host>();
                let expected_scores = query.clone()
                    .matmul(repeat_heads(key.clone(), groups).swap_dims(2, 3)) * 0.25;
                let actual_scores = gqa_scores(&ops, query, key) * 0.25;
                close(actual_scores.clone(), expected_scores.clone(), tolerance);

                // The softmax is deliberately kept over the original final axis.
                let expected_probs = softmax(expected_scores.cast(DType::F32), 3).cast(dtype);
                let actual_probs = softmax(actual_scores.cast(DType::F32), 3).cast(dtype);
                let expected = expected_probs.matmul(repeat_heads(value.clone(), groups));
                let actual = crate::gpu_inference::gqa_value_product(&ops, actual_probs, value);
                close(actual, expected, tolerance);
            }
        }
    }
}

#[test]
fn compressed_mla_matches_expanded_history_f32() {
    let ops = shared_tensor_ops::<Host>();
    for (batch, heads, sequence, length) in [(1, 4, 1, 17), (2, 3, 5, 11)] {
        for transposed in [false, true] {
            let q = sample([batch, heads, sequence, 8], 3, transposed, DType::F32);
            let qp = sample([batch, heads, sequence, 4], 5, transposed, DType::F32);
            let c = sample([batch, 1, length, 12], 7, transposed, DType::F32);
            let kp = sample([batch, 1, length, 4], 11, transposed, DType::F32);
            let wk = sample([1, heads, 8, 12], 13, false, DType::F32);
            let wv = sample([1, heads, 12, 6], 17, false, DType::F32);
            let expanded_key = c.clone().matmul(wk.clone().swap_dims(2, 3));
            let expanded_value = c.clone().matmul(wv.clone());
            let expected_scores = (q.clone().matmul(expanded_key.swap_dims(2, 3))
                + qp.clone().matmul(kp.clone().swap_dims(2, 3))) * 0.31;
            let absorbed = crate::gpu_inference::mla_absorb_query(&ops, q, wk);
            let actual_scores = crate::gpu_inference::mla_scores(
                &ops, absorbed, qp, c.clone(), kp,
            ) * 0.31;
            close(actual_scores.clone(), expected_scores.clone(), 2e-4);
            let expected = softmax(expected_scores, 3).matmul(expanded_value);
            let actual = crate::gpu_inference::mla_value_product(
                &ops, softmax(actual_scores, 3), c, wv,
            );
            close(actual, expected, 2e-4);
        }
    }
}
