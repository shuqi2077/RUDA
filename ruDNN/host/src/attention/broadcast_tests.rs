//! Verify broadcast preparation separately from the attention arithmetic.

use super::*;
use ruda_core::tensor::{BoolStore, Shape};

fn tensor_f32(data: Vec<f32>, shape: [usize; 4]) -> HostTensor {
    HostTensor::new(
        Bytes::from_elems(data),
        Layout::contiguous(Shape::new(shape)),
        DType::F32,
    )
}

fn expanded(tensor: HostTensor, target: [usize; 4]) -> HostTensor {
    ruprim_host::expand::expand(tensor, Shape::new(target)).to_contiguous()
}

#[test]
fn broadcast_preparation_copies_each_distinct_tile_once() {
    let target = [2, 3, 4, 5];
    let tile_len = target[2] * target[3];
    // All combinations of leading and trailing broadcasting, including both
    // query-row broadcasting and key-column broadcasting.
    for bits in 0..16 {
        let src: [usize; 4] = core::array::from_fn(|d| {
            if bits & (1 << d) == 0 { target[d] } else { 1 }
        });
        let len: usize = src.iter().product();
        let source = tensor_f32((0..len).map(|i| i as f32 - 7.0).collect(), src);
        let full = expanded(source.clone(), target);
        let prepared = broadcast_attn_mask_bias(source, target, "bias");
        let data: &[f32] = prepared.tensor.storage();
        assert_eq!(data.len(), src[0] * src[1] * tile_len, "shape {src:?}");
        for b in 0..target[0] {
            for h in 0..target[1] {
                let compact_offset = b * prepared.batch_step + h * prepared.head_step;
                let full_offset = (b * target[1] + h) * tile_len;
                assert_eq!(
                    &data[compact_offset..compact_offset + tile_len],
                    &full.storage::<f32>()[full_offset..full_offset + tile_len],
                    "shape {src:?}, batch {b}, head {h}",
                );
            }
        }
    }
}

#[test]
fn trailing_broadcast_preserves_transposed_offset_and_negative_stride_views() {
    let target = [2, 3, 4, 5];
    // Narrow leaves a nonzero offset and excess backing storage. Transpose and
    // flip exercise strides that cannot be treated as a contiguous source tile.
    let source = tensor_f32((0..45).map(|i| i as f32).collect(), [3, 3, 5, 1])
        .narrow(0, 1, 1)
        .transpose(2, 3);
    let source = HostTensor::from_arc(source.data_arc(), source.layout().flip(&[1, 3]), source.dtype());
    let full = expanded(source.clone(), target);
    let prepared = broadcast_attn_mask_bias(source, target, "bias");
    assert_eq!(prepared.tensor.storage::<f32>().len(), 3 * 4 * 5);
    assert_eq!(prepared.batch_step, 0);
    for b in 0..2 {
        for h in 0..3 {
            let offset = h * prepared.head_step;
            let expected = (b * 3 + h) * 20;
            assert_eq!(
                &prepared.tensor.storage::<f32>()[offset..offset + 20],
                &full.storage::<f32>()[expected..expected + 20],
            );
        }
    }
}

#[test]
fn trailing_broadcast_empty_targets_do_not_allocate_tiles() {
    for target in [[0, 3, 4, 5], [2, 0, 4, 5], [2, 3, 0, 5], [2, 3, 4, 0]] {
        let prepared = broadcast_attn_mask_bias(
            tensor_f32(vec![1.0], [1, 1, 1, 1]), target, "bias",
        );
        assert!(prepared.tensor.bytes().is_empty(), "target {target:?}");
    }
}

#[test]
#[should_panic(expected = "bias dim 1 must be 3 or 1, got 2")]
fn trailing_broadcast_still_validates_empty_targets() {
    broadcast_attn_mask_bias(tensor_f32(vec![0.0; 2], [1, 2, 1, 1]), [0, 3, 4, 5], "bias");
}

#[test]
fn trailing_broadcast_matches_dense_inputs_for_naive_and_flash() {
    let target = [2, 2, 3, TILE_KV + 3];
    let q_shape = [target[0], target[1], target[2], 4];
    let k_shape = [target[0], target[1], target[3], 4];
    let v_shape = [target[0], target[1], target[3], 3];
    let input = |shape: [usize; 4]| {
        tensor_f32((0..shape.iter().product::<usize>()).map(|i| (i % 29) as f32 / 29.0 - 0.5).collect(), shape)
    };
    let query = input(q_shape);
    let key = input(k_shape);
    let value = input(v_shape);
    let options = AttentionModuleOptions { scale: Some(0.75), softcap: Some(2.0), is_causal: true };

    for bits in 0..16 {
        let src: [usize; 4] = core::array::from_fn(|d| {
            if bits & (1 << d) == 0 { target[d] } else { 1 }
        });
        let len: usize = src.iter().product();
        let mask = HostTensor::new(
            Bytes::from_elems((0..len).map(|i| u8::from(i % 7 == 0)).collect::<Vec<_>>()),
            Layout::contiguous(Shape::new(src)),
            DType::Bool(BoolStore::Native),
        );
        let bias = tensor_f32((0..len).map(|i| (i % 13) as f32 / 17.0).collect(), src);
        let full_mask = expanded(mask.clone(), target);
        let full_bias = expanded(bias.clone(), target);
        for attention_fn in [attention_naive, attention_flash] {
            let compact = attention_fn(
                query.clone(), key.clone(), value.clone(),
                Some(mask.clone()), Some(bias.clone()), options,
            );
            let full = attention_fn(
                query.clone(), key.clone(), value.clone(),
                Some(full_mask.clone()), Some(full_bias.clone()), options,
            );
            // Same strategy, same values and operation order: require exact bits.
            let compact_bits: Vec<_> = compact.storage::<f32>().iter().map(|x| x.to_bits()).collect();
            let full_bits: Vec<_> = full.storage::<f32>().iter().map(|x| x.to_bits()).collect();
            assert_eq!(compact_bits, full_bits, "shape {src:?}");
        }
    }
}

#[test]
fn trailing_broadcast_keeps_f64_bias_precision() {
    let target = [2, 2, 2, 3];
    let tensor = |data: Vec<f64>, shape: [usize; 4]| HostTensor::new(
        Bytes::from_elems(data), Layout::contiguous(Shape::new(shape)), DType::F64,
    );
    let query = tensor(vec![1.0; 8], [2, 2, 2, 1]);
    let key = tensor(vec![1.0; 12], [2, 2, 3, 1]);
    let value = tensor((0..12).map(|i| i as f64).collect(), [2, 2, 3, 1]);
    let bias = tensor(vec![0.0, 1e-10, -1e-10, 0.5, 0.5 + 1e-10, 0.5 - 1e-10], [1, 2, 1, 3]);
    let dense = expanded(bias.clone(), target);
    for attention_fn in [attention_naive, attention_flash] {
        let compact = attention_fn(query.clone(), key.clone(), value.clone(), None, Some(bias.clone()), Default::default());
        let full = attention_fn(query.clone(), key.clone(), value.clone(), None, Some(dense.clone()), Default::default());
        assert_eq!(compact.storage::<f64>(), full.storage::<f64>());
    }
}
