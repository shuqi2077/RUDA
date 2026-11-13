//! Hardware differential tests. Not part of CPU-only compilation validation.
use ruda_core::tensor::{DType, Shape};
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::{
    dsl::Runtime,
    tensor::{RudaTensor, readback::into_data_sync},
};
use rudnn::{
    inference::{
        cascade::{self, AttentionState, LogBase},
        paged::{AttentionOptions, CacheLayout, Mask, PagedPlan},
    },
    paged_attention::HostPlan,
};
use rurand::tensor::sampling::{
    FilterOrder, SamplingFilter, SamplingPlan, chain_speculative_sampling,
};

fn upload<T: bytemuck::Pod>(
    values: &[T],
    shape: impl Into<Shape>,
    dtype: DType,
) -> RudaTensor<CudaRuntime> {
    let device = CudaDevice::default();
    let client = CudaRuntime::client(&device);
    RudaTensor::new_contiguous(
        client.clone(),
        device,
        shape.into(),
        client.create_from_slice(bytemuck::cast_slice(values)),
        dtype,
    )
}
fn f32s(t: RudaTensor<CudaRuntime>) -> Vec<f32> {
    into_data_sync(t).convert::<f32>().to_vec().unwrap()
}

#[test]
fn low_precision_packed_mask_alibi_and_base_two_lse() {
    for dtype in [DType::F16, DType::BF16] {
        let make = |x: &[f32], shape: Shape| {
            if dtype == DType::F16 {
                upload(
                    &x.iter()
                        .copied()
                        .map(half::f16::from_f32)
                        .collect::<Vec<_>>(),
                    shape,
                    dtype,
                )
            } else {
                upload(
                    &x.iter()
                        .copied()
                        .map(half::bf16::from_f32)
                        .collect::<Vec<_>>(),
                    shape,
                    dtype,
                )
            }
        };
        let q = make(&[1.0; 34], [1, 2, 17].into());
        let k = make(&[0.0; 34], [1, 2, 1, 17].into());
        let mut values = vec![2.0; 19];
        values.extend([6.0; 19]);
        let v = make(&values, [1, 2, 1, 19].into());
        let host = HostPlan::new(2, 1, &[vec![0]], &[2], &[0], &[1]).unwrap();
        let plan = PagedPlan::upload(host, &q).unwrap();
        let mask = upload(&[0b10u8], [1, 1], DType::U8);
        let slopes = upload(&[0.5f32, 1.0], [2], DType::F32);
        let out = plan
            .run(
                &q,
                &k,
                &v,
                CacheLayout::Nhd,
                AttentionOptions {
                    scale: 1.0,
                    causal: true,
                    window_left: -1,
                    logits_soft_cap: 0.0,
                    lse_base: LogBase::Two,
                },
                Some(Mask::Packed(&mask)),
                Some(&slopes),
            )
            .unwrap();
        close(&f32s(out.values), &[6.0; 38]);
        close(&f32s(out.lse), &[0.0, 0.0]);
        let out = plan
            .run(
                &q,
                &k,
                &v,
                CacheLayout::Nhd,
                AttentionOptions {
                    scale: 1.0,
                    causal: false,
                    window_left: -1,
                    logits_soft_cap: 0.0,
                    lse_base: LogBase::Two,
                },
                None,
                Some(&slopes),
            )
            .unwrap();
        let actual = f32s(out.values);
        let lse = f32s(out.lse);
        for (h, slope) in [0.5f32, 1.0].iter().enumerate() {
            let w = (-slope).exp();
            let expected = (2.0 * w + 6.0) / (w + 1.0);
            for &a in &actual[h * 19..(h + 1) * 19] {
                assert!((a - expected).abs() < 0.04);
            }
            close(&[lse[h]], &[(w + 1.0).log2()]);
        }
    }
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&a, &b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 3e-5 * (1.0 + b.abs()), "{a} != {b}");
    }
}

#[test]
fn indexed_sampling_matches_scalar_filter_reference() {
    let (sources, rows, width) = (20, 70, 37);
    let logits: Vec<f32> = (0..sources * width)
        .map(|i| {
            if i % 13 == 0 {
                f32::NEG_INFINITY
            } else {
                ((i * 17 % 43) as f32 - 21.0) * 0.17
            }
        })
        .collect();
    let x = upload(&logits, [sources, width], DType::F32);
    let indices: Vec<u32> = (0..rows).map(|i| (i * 7 % sources) as u32).collect();
    let filters: Vec<SamplingFilter> = (0..rows)
        .map(|i| SamplingFilter {
            temperature: 0.7 + (i % 4) as f32 * 0.2,
            top_k: [0, 1, 7, 16, 80][i % 5],
            top_p: [0.43, 0.79, 0.97, 1.0][i % 4],
            min_p: [0.0, 0.2, 1.0][i % 3],
        })
        .collect();
    for order in [FilterOrder::TopKFirst, FilterOrder::Joint] {
        let plan = SamplingPlan::new(&x, &filters, Some(&indices), true, order).unwrap();
        let distribution = plan.prepare(&x).unwrap();
        let actual = f32s(distribution.probabilities.clone());
        for row in 0..rows {
            let filter = filters[row];
            let source =
                &logits[indices[row] as usize * width..(indices[row] as usize + 1) * width];
            let maximum = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = source
                .iter()
                .map(|&x| ((x - maximum) / filter.temperature).exp())
                .collect();
            let mut sorted = weights.clone();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let cutoff = if filter.top_k == 0 || filter.top_k as usize >= width {
                0.0
            } else {
                sorted[filter.top_k as usize - 1]
            };
            let eligible: Vec<f32> = sorted
                .iter()
                .copied()
                .filter(|&w| order == FilterOrder::Joint || w >= cutoff)
                .collect();
            let total: f32 = eligible.iter().sum();
            let mut nucleus = 0.0;
            if filter.top_p < 1.0 {
                let mut cumulative = 0.0;
                for w in eligible {
                    cumulative += w;
                    if cumulative >= filter.top_p * total {
                        nucleus = w;
                        break;
                    }
                }
            }
            let boundary = cutoff.max(nucleus);
            let mut expected: Vec<f32> = weights
                .iter()
                .map(|&w| {
                    if w >= boundary && w >= filter.min_p {
                        w
                    } else {
                        0.0
                    }
                })
                .collect();
            let total: f32 = expected.iter().sum();
            for w in &mut expected {
                *w /= total;
            }
            close(&actual[row * width..(row + 1) * width], &expected);
        }
        let draws: Vec<f32> = (0..rows)
            .map(|i| ((i * 31 % 997) as f32 + 0.37) / 997.0)
            .collect();
        let uniforms = upload(&draws, [rows], DType::F32);
        let ids = into_data_sync(distribution.sample(&uniforms).unwrap().token_ids)
            .to_vec::<i32>()
            .unwrap();
        for row in 0..rows {
            let p = &actual[row * width..(row + 1) * width];
            let total: f64 = p.iter().map(|&x| x as f64).sum();
            let mut cumulative = 0.0;
            let mut expected = 0;
            for (j, &w) in p.iter().enumerate() {
                cumulative += w as f64;
                if cumulative > draws[row] as f64 * total {
                    expected = j;
                    break;
                }
            }
            assert_eq!(ids[row], expected as i32, "row {row}, {order:?}");
        }
    }
}
#[test]
fn sampling_filters_ties_invalid_rows_and_chain_rejection() {
    let input = upload(&[0.1f32, 0.4, 0.3, 0.2], [1, 4], DType::F32);
    let filter = SamplingFilter {
        top_k: 2,
        ..Default::default()
    };
    let plan = SamplingPlan::new(&input, &[filter], None, false, FilterOrder::TopKFirst).unwrap();
    close(
        &f32s(plan.prepare(&input).unwrap().probabilities),
        &[0.0, 4.0 / 7.0, 3.0 / 7.0, 0.0],
    );
    let signed_zero = upload(&[-0.0f32, 0.1, 0.9], [1, 3], DType::F32);
    let single = SamplingPlan::new(
        &signed_zero,
        &[SamplingFilter {
            top_k: 1,
            ..Default::default()
        }],
        None,
        false,
        FilterOrder::TopKFirst,
    )
    .unwrap();
    close(
        &f32s(single.prepare(&signed_zero).unwrap().probabilities),
        &[0.0, 0.0, 1.0],
    );
    let u = upload(&[0.99f32], [1], DType::F32);
    assert_eq!(
        into_data_sync(plan.sample(&input, &u).unwrap().token_ids)
            .to_vec::<i32>()
            .unwrap(),
        vec![2]
    );
    let tied = upload(&[0.2f32; 5], [1, 5], DType::F32);
    let plan = SamplingPlan::new(
        &tied,
        &[SamplingFilter {
            top_k: 2,
            top_p: 0.5,
            ..Default::default()
        }],
        None,
        false,
        FilterOrder::TopKFirst,
    )
    .unwrap();
    close(&f32s(plan.prepare(&tied).unwrap().probabilities), &[0.2; 5]);
    let invalid = upload(&[f32::NEG_INFINITY; 5], [1, 5], DType::F32);
    let plan = SamplingPlan::new(
        &invalid,
        &[SamplingFilter::default()],
        None,
        true,
        FilterOrder::TopKFirst,
    )
    .unwrap();
    assert_eq!(
        into_data_sync(plan.sample(&invalid, &u).unwrap().token_ids)
            .to_vec::<i32>()
            .unwrap(),
        vec![-1]
    );
    let draft = upload(&[0.9f32, 0.1], [1, 1, 2], DType::F32);
    let ids = upload(&[0i32], [1, 1], DType::I32);
    let target = upload(&[0.1f32, 0.9, 0.5, 0.5], [1, 2, 2], DType::F32);
    let draws = upload(&[0.5f32, 0.5], [1, 2], DType::F32);
    let out = chain_speculative_sampling(&draft, &ids, &target, &draws, &draws).unwrap();
    assert_eq!(
        into_data_sync(out.token_ids).to_vec::<i32>().unwrap(),
        vec![1, -1]
    );
    assert_eq!(
        into_data_sync(out.accepted).to_vec::<u32>().unwrap(),
        vec![0]
    );
    assert_eq!(
        into_data_sync(out.emitted).to_vec::<u32>().unwrap(),
        vec![1]
    );
    let draft = upload(&[0.5f32, 0.5, 0.4, 0.6], [1, 2, 2], DType::F32);
    let target = upload(&[0.5f32, 0.5, 0.4, 0.6, 0.1, 0.9], [1, 3, 2], DType::F32);
    let ids = upload(&[0i32, 1], [1, 2], DType::I32);
    let draws = upload(&[0.9f32, 0.9, 0.5], [1, 3], DType::F32);
    let out = chain_speculative_sampling(&draft, &ids, &target, &draws, &draws).unwrap();
    assert_eq!(
        into_data_sync(out.token_ids).to_vec::<i32>().unwrap(),
        vec![0, 1, 1]
    );
    assert_eq!(
        into_data_sync(out.accepted).to_vec::<u32>().unwrap(),
        vec![2]
    );
    assert_eq!(
        into_data_sync(out.emitted).to_vec::<u32>().unwrap(),
        vec![3]
    );
    let draft = upload(&[0.9f32, 0.1, 0.5, 0.5], [1, 2, 2], DType::F32);
    let target = upload(&[0.1f32, 0.9, 0.5, 0.5, 0.1, 0.9], [1, 3, 2], DType::F32);
    let draws = upload(&[0.5f32, 0.1, 0.5], [1, 3], DType::F32);
    let out = chain_speculative_sampling(&draft, &ids, &target, &draws, &draws).unwrap();
    assert_eq!(
        into_data_sync(out.accepted).to_vec::<u32>().unwrap(),
        vec![1]
    );
    assert_eq!(
        into_data_sync(out.emitted_draft).to_vec::<u32>().unwrap(),
        vec![0]
    );
    assert_eq!(
        into_data_sync(out.emitted).to_vec::<u32>().unwrap(),
        vec![1]
    );
}
#[test]
fn segmented_bitpacking_orders_and_padding() {
    let x = upload(&[1u8, 0, 1, 1, 0, 0, 0, 1, 1, 1], [10], DType::U8);
    let big = ruprim::bitpack::segment_packbits(&x, &[0, 3, 3, 10], ruprim::bitpack::BitOrder::Big)
        .unwrap();
    assert_eq!(big.host_indptr, vec![0, 1, 1, 2]);
    assert_eq!(
        into_data_sync(big.data).to_vec::<u8>().unwrap(),
        vec![0b10100000, 0b10001110]
    );
    let little = ruprim::bitpack::packbits(&x, ruprim::bitpack::BitOrder::Little).unwrap();
    assert_eq!(
        into_data_sync(little).to_vec::<u8>().unwrap(),
        vec![0b10001101, 0b00000011]
    );
    let words = upload(
        &[1u32, 0, 1, 1, 0, 0, 0, 1, 1, 1],
        [10],
        DType::Bool(ruda_core::tensor::BoolStore::U32),
    );
    let little = ruprim::bitpack::packbits(&words, ruprim::bitpack::BitOrder::Little).unwrap();
    assert_eq!(
        into_data_sync(little).to_vec::<u8>().unwrap(),
        vec![0b10001101, 0b00000011]
    );
}
#[test]
fn cascade_empty_states_and_large_lse() {
    for base in [LogBase::Natural, LogBase::Two] {
        let a = AttentionState {
            values: upload(&[2f32, 4.0, f32::NAN, f32::NAN], [2, 1, 2], DType::F32),
            lse: upload(&[1000f32, f32::NEG_INFINITY], [2, 1], DType::F32),
        };
        let b = AttentionState {
            values: upload(&[8f32, 10.0, f32::NAN, f32::NAN], [2, 1, 2], DType::F32),
            lse: upload(&[-1000f32, f32::NEG_INFINITY], [2, 1], DType::F32),
        };
        let merged = cascade::merge_state(&a, &b, base).unwrap();
        close(&f32s(merged.values), &[2.0, 4.0, 0.0, 0.0]);
        let lse = f32s(merged.lse);
        assert!((lse[0] - 1000.0).abs() < 0.001);
        assert_eq!(lse[1], f32::NEG_INFINITY);
    }
}
#[test]
fn paged_layouts_masks_lse_and_snapshot_append() {
    let (pages, page_size, heads, kv_heads, d, dv) = (4, 2, 4, 2, 3, 5);
    let q: Vec<f32> = (0..3 * heads * d)
        .map(|i| (i as f32 * 0.23).sin())
        .collect();
    let k: Vec<f32> = (0..pages * page_size * kv_heads * d)
        .map(|i| (i as f32 * 0.13).cos())
        .collect();
    let v: Vec<f32> = (0..pages * page_size * kv_heads * dv)
        .map(|i| i as f32 * 0.01 - 0.5)
        .collect();
    let query = upload(&q, [3, heads, d], DType::F32);
    let tables = [vec![2, 0], vec![3]];
    let ids = [0usize, 0, 1];
    let positions = [1usize, 2, 0];
    let lengths = [3usize, 1];
    let host = HostPlan::new(page_size, pages, &tables, &[3, 1], &[0, 0, 1], &[1, 2, 0]).unwrap();
    let plan = PagedPlan::upload(host, &query).unwrap();
    let reorder = |x: &[f32], width: usize| {
        let mut out = vec![0.0; x.len()];
        for p in 0..pages {
            for n in 0..page_size {
                for h in 0..kv_heads {
                    for c in 0..width {
                        out[((p * kv_heads + h) * page_size + n) * width + c] =
                            x[((p * page_size + n) * kv_heads + h) * width + c];
                    }
                }
            }
        }
        out
    };
    let bytes = [1u8, 0, 1, 1, 1, 0, 0, 0, 0];
    let mask = upload(&bytes, [3, 3], DType::U8);
    for layout in [CacheLayout::Nhd, CacheLayout::Hnd] {
        let key = if layout == CacheLayout::Nhd {
            upload(&k, [pages, page_size, kv_heads, d], DType::F32)
        } else {
            upload(&reorder(&k, d), [pages, kv_heads, page_size, d], DType::F32)
        };
        let value = if layout == CacheLayout::Nhd {
            upload(&v, [pages, page_size, kv_heads, dv], DType::F32)
        } else {
            upload(
                &reorder(&v, dv),
                [pages, kv_heads, page_size, dv],
                DType::F32,
            )
        };
        let options = AttentionOptions {
            scale: 0.6,
            causal: true,
            window_left: 1,
            logits_soft_cap: 0.7,
            lse_base: LogBase::Natural,
        };
        let result = plan
            .run(
                &query,
                &key,
                &value,
                layout,
                options,
                Some(Mask::Bytes(&mask)),
                None,
            )
            .unwrap();
        let mut expected = vec![0.0; 3 * heads * dv];
        let mut expected_lse = vec![f32::NEG_INFINITY; 3 * heads];
        for row in 0..3 {
            for h in 0..heads {
                let seq = ids[row];
                let pos = positions[row];
                let mut scores = Vec::new();
                for t in pos.saturating_sub(1)..lengths[seq].min(pos + 1) {
                    if bytes[row * 3 + t] == 0 {
                        continue;
                    }
                    let slot = (tables[seq][t / page_size] as usize * page_size + t % page_size)
                        * kv_heads
                        + h / (heads / kv_heads);
                    let dot: f32 = (0..d)
                        .map(|c| q[(row * heads + h) * d + c] * k[slot * d + c])
                        .sum();
                    scores.push((slot, 0.7 * (dot * 0.6 / 0.7).tanh()));
                }
                if scores.is_empty() {
                    continue;
                }
                let m = scores.iter().map(|s| s.1).fold(f32::NEG_INFINITY, f32::max);
                let total: f32 = scores.iter().map(|s| (s.1 - m).exp()).sum();
                expected_lse[row * heads + h] = m + total.ln();
                for (slot, s) in scores {
                    for c in 0..dv {
                        expected[(row * heads + h) * dv + c] +=
                            (s - m).exp() / total * v[slot * dv + c];
                    }
                }
            }
        }
        close(&f32s(result.values), &expected);
        let lse = f32s(result.lse);
        for (&a, &b) in lse.iter().zip(&expected_lse) {
            if b == f32::NEG_INFINITY {
                assert_eq!(a, b);
            } else {
                close(&[a], &[b]);
            }
        }
        let snapshot = key.clone();
        let legacy_key = key.clone();
        let old = f32s(snapshot.clone());
        let tokens = upload(&vec![42f32; 3 * kv_heads * d], [3, kv_heads, d], DType::F32);
        let updated = plan.append(&tokens, key, layout).unwrap();
        close(&f32s(snapshot), &old);
        close(
            &f32s(plan.gather(&updated, layout).unwrap()),
            &vec![42.0; 3 * kv_heads * d],
        );
        if layout == CacheLayout::Nhd {
            let legacy = rudnn::paged_attention::DevicePlan::upload(plan.host().clone(), &query);
            let new_v = upload(
                &vec![11.0f32; 3 * kv_heads * dv],
                [3, kv_heads, dv],
                DType::F32,
            );
            let (new_k, new_v) = legacy
                .append(tokens.clone(), new_v, legacy_key.clone(), value.clone())
                .unwrap();
            assert!(new_k.is_contiguous() && new_v.is_contiguous());
            close(
                &f32s(plan.gather(&new_k, layout).unwrap()),
                &vec![42.0; 3 * kv_heads * d],
            );
            close(
                &f32s(plan.gather(&new_v, layout).unwrap()),
                &vec![11.0; 3 * kv_heads * dv],
            );
            close(&f32s(legacy_key), &old);
        }
    }
}
