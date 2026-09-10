use super::*;
use half::{bf16, f16};
use rublas::tensor_grouped::grouped_matmul_nt;
use ruda_core::tensor::data::TensorData;
use ruda_kernel::tensor::{permutation::swap_dims, readback::into_data_sync, transfer::from_data};
use ruda_test_runtime::TestRuntime;

type Tensor = RudaTensor<TestRuntime>;

fn tensor(values: &[f32], shape: impl Into<Shape>, dtype: DType) -> Tensor {
    let shape = shape.into();
    let data = match dtype {
        DType::F32 => TensorData::new(values.to_vec(), shape),
        DType::F16 => TensorData::new(
            values
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
            shape,
        ),
        DType::BF16 => TensorData::new(
            values
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            shape,
        ),
        _ => panic!("test float dtype"),
    };
    from_data(data, &Default::default())
}

fn floats(tensor: Tensor) -> Vec<f32> {
    let dtype = tensor.dtype;
    let data = into_data_sync(tensor);
    match dtype {
        DType::F32 => data.to_vec::<f32>().unwrap(),
        DType::F16 => data
            .to_vec::<f16>()
            .unwrap()
            .into_iter()
            .map(f16::to_f32)
            .collect(),
        DType::BF16 => data
            .to_vec::<bf16>()
            .unwrap()
            .into_iter()
            .map(bf16::to_f32)
            .collect(),
        _ => panic!("test float dtype"),
    }
}

fn integers(tensor: Tensor) -> Vec<u32> {
    into_data_sync(tensor).to_vec::<u32>().unwrap()
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= tolerance * b.abs().max(1.0),
            "{index}: {a} != {b}"
        );
    }
}

fn round(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::F16 => f16::from_f32(value).to_f32(),
        DType::BF16 => bf16::from_f32(value).to_f32(),
        _ => value,
    }
}

#[test]
fn dimensions_reject_overflow_even_for_empty_tensors() {
    assert_eq!(elements(&[0, 8]).unwrap(), 0);
    assert!(elements(&[u32::MAX as usize, 2]).is_err());
    if usize::BITS > 32 {
        assert!(elements(&[0, u32::MAX as usize + 1]).is_err());
    }
}

#[test]
fn route_matches_known_softmax_and_optional_normalization() {
    let logits = [0.0, 2.0f32.ln(), 4.0f32.ln(), 8.0f32.ln()];
    for renormalize in [false, true] {
        let plan = route(
            tensor(&logits, [1, 4], DType::F32),
            RoutingOptions {
                top_k: 2,
                renormalize,
            },
        )
        .unwrap();
        assert_eq!(integers(plan.indices), [3, 2]);
        let divisor = if renormalize { 12.0 } else { 15.0 };
        close(&floats(plan.weights), &[8.0 / divisor, 4.0 / divisor], 2e-6);
    }
}

#[test]
fn route_ties_masks_and_nonfinite_rows_have_explicit_behavior() {
    let values = [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        f32::NEG_INFINITY,
        0.0,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
        1.0,
        2.0,
        f32::INFINITY,
        0.0,
        1.0,
        2.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ];
    let plan = route(
        tensor(&values, [5, 4], DType::F32),
        RoutingOptions {
            top_k: 4,
            renormalize: false,
        },
    )
    .unwrap();
    assert_eq!(
        integers(plan.indices),
        [0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    );
    let weights = floats(plan.weights);
    close(&weights[..4], &[0.25; 4], 1e-6);
    assert_eq!(&weights[6..8], &[0.0, 0.0]);
    assert!(weights[8..].iter().all(|weight| weight.is_nan()));
}

#[test]
fn compact_dispatch_preserves_all_routes_and_empty_experts() {
    let input = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let logits = [
        3.0, 2.0, -8.0, -9.0, 2.0, 3.0, -8.0, -9.0, 4.0, 1.0, -8.0, -9.0,
    ];
    let plan = route(
        tensor(&logits, [3, 4], DType::F32),
        RoutingOptions {
            top_k: 2,
            renormalize: true,
        },
    )
    .unwrap();
    let dispatched = plan.dispatch(tensor(&input, [3, 2], DType::F32)).unwrap();
    assert_eq!(integers(dispatched.offsets.clone()), [0, 3, 6, 6, 6]);
    assert_eq!(integers(dispatched.row_experts.clone()), [0, 0, 0, 1, 1, 1]);
    let rows = integers(dispatched.slot_rows.clone());
    let mut sorted = rows.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, [0, 1, 2, 3, 4, 5]);
    let packed = floats(dispatched.values.clone());
    for (slot, row) in rows.into_iter().enumerate() {
        assert_eq!(
            &packed[row as usize * 2..row as usize * 2 + 2],
            &input[slot / 2 * 2..slot / 2 * 2 + 2]
        );
    }
    let retained = dispatched.clone();
    drop(dispatched);
    close(
        &floats(retained.combine(retained.values.clone()).unwrap()),
        &input,
        2e-6,
    );
}

#[test]
fn grouped_matmul_handles_strides_padding_and_float_dtypes() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let input = swap_dims(tensor(&[1.0, 3.0, 5.0, 2.0, 4.0, 6.0], [2, 3], dtype), 0, 1);
        let weights = swap_dims(
            tensor(&[1.0, 2.0, 3.0, 4.0, 2.0, 1.0, -1.0, 3.0], [2, 2, 2], dtype),
            1,
            2,
        );
        let groups = from_data(
            TensorData::new(vec![0u32, 1, u32::MAX], [3]),
            &Default::default(),
        );
        let output = grouped_matmul_nt(input, weights, groups).unwrap();
        close(&floats(output), &[7.0, 10.0, 2.0, 15.0, 0.0, 0.0], 0.0);
    }
}

#[test]
fn contended_dispatch_spans_workgroups_without_losing_assignments() {
    let tokens = 257;
    let input = (0..tokens * 3).map(|i| i as f32 / 16.0).collect::<Vec<_>>();
    let logits = [3.0, 2.0, -8.0, -9.0].repeat(tokens);
    for _ in 0..3 {
        let plan = route(
            tensor(&logits, [tokens, 4], DType::F32),
            RoutingOptions {
                top_k: 2,
                renormalize: true,
            },
        )
        .unwrap();
        let dispatched = plan
            .dispatch(tensor(&input, [tokens, 3], DType::F32))
            .unwrap();
        assert_eq!(
            integers(dispatched.offsets.clone()),
            [0, 257, 514, 514, 514]
        );
        let mut assignments = integers(dispatched.slot_rows.clone());
        assignments.sort_unstable();
        assert_eq!(assignments, (0..514).collect::<Vec<u32>>());
        close(
            &floats(dispatched.combine(dispatched.values.clone()).unwrap()),
            &input,
            2e-6,
        );
    }
}

fn reference(
    input: &[f32],
    logits: &[f32],
    gate: &[f32],
    up: &[f32],
    down: &[f32],
    dtype: DType,
    options: RoutingOptions,
) -> Vec<f32> {
    let mut output = vec![0.0; 6];
    for token in 0..3 {
        let scores = &logits[token * 3..token * 3 + 3];
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut probabilities = scores
            .iter()
            .map(|&x| (x as f64 - maximum).exp())
            .collect::<Vec<_>>();
        let sum: f64 = probabilities.iter().sum();
        probabilities.iter_mut().for_each(|x| *x /= sum);
        let mut selected = [0usize, 1, 2];
        selected.sort_by(|&a, &b| {
            probabilities[b]
                .total_cmp(&probabilities[a])
                .then(a.cmp(&b))
        });
        let denominator = if options.renormalize {
            selected[..options.top_k]
                .iter()
                .map(|&e| probabilities[e])
                .sum()
        } else {
            1.0
        };
        for expert in 0..3 {
            if !selected[..options.top_k].contains(&expert) {
                continue;
            }
            let weight = round((probabilities[expert] / denominator) as f32, dtype);
            let mut activation = [0.0; 2];
            for i in 0..2 {
                let mut g = 0.0f64;
                let mut u = 0.0f64;
                for h in 0..2 {
                    g += input[token * 2 + h] as f64 * gate[expert * 4 + i * 2 + h] as f64;
                    u += input[token * 2 + h] as f64 * up[expert * 4 + i * 2 + h] as f64;
                }
                let g = round(g as f32, dtype);
                let u = round(u as f32, dtype);
                let silu = round((g as f64 / (1.0 + (-(g as f64)).exp())) as f32, dtype);
                activation[i] = round(silu * u, dtype);
            }
            for h in 0..2 {
                let dot: f64 = (0..2)
                    .map(|i| activation[i] as f64 * down[expert * 4 + h * 2 + i] as f64)
                    .sum();
                let contribution = round(round(dot as f32, dtype) * weight, dtype);
                output[token * 2 + h] = round(output[token * 2 + h] + contribution, dtype);
            }
        }
    }
    output
}

#[test]
fn full_swiglu_matches_independent_expert_loop() {
    let input = [0.5, -1.0, 1.5, 0.25, -0.5, 2.0];
    let logits = [2.0, 1.0, -1.0, -2.0, 0.0, 1.0, 1.0, -3.0, 2.0];
    let gate = [
        1.0, 0.5, -0.25, 1.0, 0.5, -1.0, 1.0, 0.25, -0.5, 1.0, 0.25, -0.5,
    ];
    let up = [
        0.5, 0.0, 1.0, -0.5, 1.0, 0.5, -1.0, 0.25, 0.25, -0.5, 0.5, 1.0,
    ];
    let down = [
        1.0, -0.5, 0.25, 0.5, -0.5, 1.0, 0.5, 0.25, 0.5, -0.25, 1.0, 0.5,
    ];
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let experts = SwiGluExperts::new(
            tensor(&gate, [3, 2, 2], dtype),
            tensor(&up, [3, 2, 2], dtype),
            tensor(&down, [3, 2, 2], dtype),
        )
        .unwrap();
        for top_k in [1, 2, 3] {
            for renormalize in [false, true] {
                let options = RoutingOptions { top_k, renormalize };
                let result = experts
                    .forward(
                        tensor(&input, [3, 2], dtype),
                        tensor(&logits, [3, 3], dtype),
                        options,
                    )
                    .unwrap();
                let tolerance = match dtype {
                    DType::F16 => 0.004,
                    DType::BF16 => 0.03,
                    _ => 2e-5,
                };
                close(
                    &floats(result),
                    &reference(&input, &logits, &gate, &up, &down, dtype, options),
                    tolerance,
                );
            }
        }
    }
}

#[test]
fn empty_batches_keep_zero_offsets_and_output_shapes() {
    let options = RoutingOptions {
        top_k: 2,
        renormalize: true,
    };
    let dispatched = route(tensor(&[], [0, 3], DType::F32), options)
        .unwrap()
        .dispatch(tensor(&[], [0, 2], DType::F32))
        .unwrap();
    assert_eq!(integers(dispatched.offsets.clone()), [0, 0, 0, 0]);
    let experts = SwiGluExperts::new(
        tensor(&[1.0; 12], [3, 2, 2], DType::F32),
        tensor(&[1.0; 12], [3, 2, 2], DType::F32),
        tensor(&[1.0; 12], [3, 2, 2], DType::F32),
    )
    .unwrap();
    let output = dispatched
        .combine(experts.forward_dispatched(&dispatched).unwrap())
        .unwrap();
    assert_eq!(output.meta.shape(), &Shape::from([0, 2]));
}

#[test]
fn invalid_contracts_are_rejected() {
    for top_k in [0, 4] {
        assert!(
            route(
                tensor(&[0.0; 3], [1, 3], DType::F32),
                RoutingOptions {
                    top_k,
                    renormalize: true
                }
            )
            .is_err()
        );
    }
    let plan = route(
        tensor(&[0.0; 3], [1, 3], DType::F32),
        RoutingOptions {
            top_k: 2,
            renormalize: true,
        },
    )
    .unwrap();
    assert!(
        plan.clone()
            .dispatch(tensor(&[0.0; 4], [2, 2], DType::F32))
            .is_err()
    );
    let dispatched = plan
        .dispatch(tensor(&[1.0; 2], [1, 2], DType::F32))
        .unwrap();
    assert!(
        dispatched
            .combine(tensor(&[0.0; 2], [1, 2], DType::F32))
            .is_err()
    );
    assert!(
        dispatched
            .combine(tensor(&[0.0; 4], [2, 2], DType::F16))
            .is_err()
    );
    assert!(
        SwiGluExperts::new(
            tensor(&[0.0; 12], [3, 2, 2], DType::F32),
            tensor(&[0.0; 8], [2, 2, 2], DType::F32),
            tensor(&[0.0; 12], [3, 2, 2], DType::F32)
        )
        .is_err()
    );
}

#[test]
fn mixed_router_dtype_and_noncontiguous_logits_preserve_combine() {
    let logits = swap_dims(
        tensor(&[2.0, 1.0, 1.0, 2.0, -8.0, -8.0], [3, 2], DType::F32),
        0,
        1,
    );
    let plan = route(
        logits,
        RoutingOptions {
            top_k: 2,
            renormalize: true,
        },
    )
    .unwrap();
    let dispatched = plan
        .dispatch(tensor(&[1.0, -2.0], [2, 1], DType::BF16))
        .unwrap();
    close(
        &floats(dispatched.combine(dispatched.values.clone()).unwrap()),
        &[1.0, -2.0],
        0.016,
    );
}
