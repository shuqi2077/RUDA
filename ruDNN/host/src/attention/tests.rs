    use alloc::{vec, vec::Vec};
    use ruda_core::tensor::DType;
    use ruda_core::tensor::spatial::AttentionModuleOptions;
    use ruda_core::{bytes::Bytes, tensor::{BoolStore, Shape}};
use half::f16;

    use ruda_core::tensor::host::{HostTensor, Layout};

    fn flex_f32(data: Vec<f32>, shape: &[usize]) -> HostTensor {
        HostTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            DType::F32,
        )
    }

    fn flex_f64(data: Vec<f64>, shape: &[usize]) -> HostTensor {
        HostTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            DType::F64,
        )
    }

    fn flex_f16(data: Vec<f16>, shape: &[usize]) -> HostTensor {
        HostTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            DType::F16,
        )
    }

    fn flex_bool(data: Vec<u8>, shape: &[usize]) -> HostTensor {
        HostTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            DType::Bool(BoolStore::Native),
        )
    }

    /// Assert two attention-output slices are elementwise close. Used by the
    /// flash broadcast tests below.
    fn assert_attention_outputs_close(bcast: &[f32], full: &[f32], label: &str) {
        assert_eq!(bcast.len(), full.len(), "{label}: length mismatch");
        for (i, (&a, &b)) in bcast.iter().zip(full).enumerate() {
            assert!((a - b).abs() < 1e-5, "{label} mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_flash_bias_broadcast_across_batch_and_heads() {
        // Exercises the flash path for a `[1, 1, seq_q, seq_kv]` broadcast
        // bias. The dispatcher in `attention()` routes to naive for
        // `seq_q * seq_kv <= NAIVE_SCORE_BUDGET` (and the backend-tests
        // attention suite uses small shapes), so the flash entry needs a
        // direct call to stay covered. General broadcast semantics for the
        // main `attention()` path live in
        // crates/ruda-backend-tests/tests/tensor/float/module/attention.rs.
        let batch = 2;
        let heads = 2;
        let seq_q = 3;
        let seq_kv = 5;
        let head_dim = 4;

        let mk = |shape: &[usize], g: &dyn Fn(usize) -> f32| -> HostTensor {
            let len: usize = shape.iter().product();
            flex_f32((0..len).map(g).collect(), shape)
        };

        let q = mk(&[batch, heads, seq_q, head_dim], &|i| {
            (i as f32 * 0.1).sin()
        });
        let k = mk(&[batch, heads, seq_kv, head_dim], &|i| {
            (i as f32 * 0.1 + 1.0).sin()
        });
        let v = mk(&[batch, heads, seq_kv, head_dim], &|i| {
            (i as f32 * 0.1 + 2.0).sin()
        });

        let bias_tile: Vec<f32> = (0..seq_q * seq_kv)
            .map(|i| (i as f32 * 0.4).sin())
            .collect();
        let bias_bcast = flex_f32(bias_tile.clone(), &[1, 1, seq_q, seq_kv]);
        let bias_full_vec: Vec<f32> = bias_tile
            .iter()
            .cloned()
            .cycle()
            .take(batch * heads * seq_q * seq_kv)
            .collect();
        let bias_full = flex_f32(bias_full_vec, &[batch, heads, seq_q, seq_kv]);

        let out_bcast = super::attention_flash(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            Some(bias_bcast),
            Default::default(),
        );
        let out_full = super::attention_flash(q, k, v, None, Some(bias_full), Default::default());

        let bcast: &[f32] = out_bcast.storage();
        let full: &[f32] = out_full.storage();
        assert_attention_outputs_close(bcast, full, "flash bias[1,1,sq,skv]");
    }

    #[test]
    fn test_flash_bool_mask_broadcast_across_batch_and_heads() {
        // Flash-path counterpart to the bias broadcast test, but for a
        // `[1, 1, seq_q, seq_kv]` bool mask. The mask (u8) slicing path with
        // stride-0 batch/head steps is a different branch from the bias
        // (f32) path, so both need direct flash coverage.
        let batch = 2;
        let heads = 2;
        let seq_q = 3;
        let seq_kv = 5;
        let head_dim = 4;

        let mk = |shape: &[usize], g: &dyn Fn(usize) -> f32| -> HostTensor {
            let len: usize = shape.iter().product();
            flex_f32((0..len).map(g).collect(), shape)
        };

        let q = mk(&[batch, heads, seq_q, head_dim], &|i| {
            (i as f32 * 0.1).sin()
        });
        let k = mk(&[batch, heads, seq_kv, head_dim], &|i| {
            (i as f32 * 0.1 + 1.0).sin()
        });
        let v = mk(&[batch, heads, seq_kv, head_dim], &|i| {
            (i as f32 * 0.1 + 2.0).sin()
        });

        // Mask out columns 3 and 4 (1 == masked out).
        let mask_tile: Vec<u8> = (0..seq_q * seq_kv)
            .map(|i| if (i % seq_kv) >= 3 { 1u8 } else { 0u8 })
            .collect();
        let mask_bcast = flex_bool(mask_tile.clone(), &[1, 1, seq_q, seq_kv]);
        let mask_full_vec: Vec<u8> = mask_tile
            .iter()
            .copied()
            .cycle()
            .take(batch * heads * seq_q * seq_kv)
            .collect();
        let mask_full = flex_bool(mask_full_vec, &[batch, heads, seq_q, seq_kv]);

        let out_bcast = super::attention_flash(
            q.clone(),
            k.clone(),
            v.clone(),
            Some(mask_bcast),
            None,
            Default::default(),
        );
        let out_full = super::attention_flash(q, k, v, Some(mask_full), None, Default::default());

        let bcast: &[f32] = out_bcast.storage();
        let full: &[f32] = out_full.storage();
        assert_attention_outputs_close(bcast, full, "flash bool mask[1,1,sq,skv]");
    }

    #[test]
    #[should_panic(expected = "must be 4D")]
    fn test_mask_wrong_rank_panics() {
        // Contract: the broadcast helper rejects non-4D mask/bias up-front
        // with a clearer message than `expand`'s rank prepending would produce.
        let mask = flex_bool(vec![0u8; 6], &[2, 3]);
        super::broadcast_attn_mask_bias(mask, [1, 1, 2, 3], "mask");
    }

    #[test]
    #[should_panic(expected = "bias dim 1 must be 3 or 1, got 2")]
    fn test_bias_incompatible_dim_panics() {
        // Contract: a dim that is neither equal to target nor `1` is a hard
        // error. Here heads=2 in the bias but target heads=3.
        let bias = flex_f32(vec![0.0f32; 2 * 2 * 4 * 5], &[2, 2, 4, 5]);
        super::broadcast_attn_mask_bias(bias, [2, 3, 4, 5], "bias");
    }

    #[test]
    fn test_multi_tile_seq_kv() {
        // seq_kv > TILE_KV (64) exercises tiling with online softmax
        // correction. 128 keys = exactly 2 tiles.
        let seq_q = 2;
        let seq_kv = 128;
        let head_dim = 4;
        let val_dim = 2;

        // Q: first query selects dim 0, second selects dim 1.
        let mut q_data = vec![0.0f32; seq_q * head_dim];
        q_data[0] = 1.0;
        q_data[head_dim + 1] = 1.0;

        // K: all keys identical so all scores are equal.
        let k_data = vec![0.1f32; seq_kv * head_dim];

        // V: linearly increasing values.
        let mut v_data = vec![0.0f32; seq_kv * val_dim];
        for i in 0..seq_kv {
            v_data[i * val_dim] = i as f32;
            v_data[i * val_dim + 1] = (seq_kv - 1 - i) as f32;
        }

        let q = flex_f32(q_data, &[1, 1, seq_q, head_dim]);
        let k = flex_f32(k_data, &[1, 1, seq_kv, head_dim]);
        let v = flex_f32(v_data, &[1, 1, seq_kv, val_dim]);

        let result = super::attention(q, k, v, None, None, Default::default());
        let data: &[f32] = result.storage();

        // All scores equal -> output = mean of 0..127 = 63.5 for both cols.
        assert_eq!(data.len(), seq_q * val_dim);
        assert!(
            (data[0] - 63.5).abs() < 0.1,
            "expected ~63.5, got {}",
            data[0]
        );
        assert!(
            (data[1] - 63.5).abs() < 0.1,
            "expected ~63.5, got {}",
            data[1]
        );
    }

    #[test]
    fn test_multi_tile_causal() {
        // seq_kv=128 with causal mask: each query sees only up to
        // causal_offset + its own index worth of keys.
        let seq_q = 4;
        let seq_kv = 128;
        let head_dim = 2;
        let val_dim = 1;

        // All queries/keys [1, 0] so all visible scores equal.
        let mut q_data = vec![0.0f32; seq_q * head_dim];
        for i in 0..seq_q {
            q_data[i * head_dim] = 1.0;
        }
        let mut k_data = vec![0.0f32; seq_kv * head_dim];
        for i in 0..seq_kv {
            k_data[i * head_dim] = 1.0;
        }
        let v_data: Vec<f32> = (0..seq_kv).map(|i| i as f32).collect();

        let q = flex_f32(q_data, &[1, 1, seq_q, head_dim]);
        let k = flex_f32(k_data, &[1, 1, seq_kv, head_dim]);
        let v = flex_f32(v_data, &[1, 1, seq_kv, val_dim]);

        let opts = AttentionModuleOptions {
            is_causal: true,
            ..Default::default()
        };
        let result = super::attention(q, k, v, None, None, opts);
        let data: &[f32] = result.storage();

        // causal_offset = seq_kv - seq_q = 124
        // q[0] sees 0..=124 -> mean 62.0; q[1] 62.5; q[2] 63.0; q[3] 63.5.
        assert_eq!(data.len(), seq_q);
        assert!((data[0] - 62.0).abs() < 0.1, "q0: got {}", data[0]);
        assert!((data[1] - 62.5).abs() < 0.1, "q1: got {}", data[1]);
        assert!((data[2] - 63.0).abs() < 0.1, "q2: got {}", data[2]);
        assert!((data[3] - 63.5).abs() < 0.1, "q3: got {}", data[3]);
    }

    #[test]
    fn test_tile_boundary_mask() {
        // Mask falls exactly on a tile boundary: first 64 keys masked,
        // next 64 visible.
        let seq_q = 1;
        let seq_kv = 128;
        let head_dim = 2;
        let val_dim = 1;

        let q_data = vec![1.0f32, 0.0];
        let k_data = vec![1.0f32, 0.0].repeat(seq_kv);
        let v_data: Vec<f32> = (0..seq_kv).map(|i| i as f32).collect();
        // true == masked out.
        let mask_data: Vec<u8> = (0..seq_kv).map(|i| (i < 64) as u8).collect();

        let q = flex_f32(q_data, &[1, 1, seq_q, head_dim]);
        let k = flex_f32(k_data, &[1, 1, seq_kv, head_dim]);
        let v = flex_f32(v_data, &[1, 1, seq_kv, val_dim]);
        let mask = flex_bool(mask_data, &[1, 1, seq_q, seq_kv]);

        let result = super::attention(q, k, v, Some(mask), None, Default::default());
        let data: &[f32] = result.storage();

        // Visible = 64..128 -> mean of 64..=127 = 95.5.
        assert!(
            (data[0] - 95.5).abs() < 0.1,
            "expected ~95.5, got {}",
            data[0]
        );
    }

    #[test]
    fn test_non_uniform_scores_across_tiles() {
        // Forces the online softmax correction path: tile 1 has much
        // larger scores than tile 0, so the correction factor
        // exp(old_max - new_max) < 1.
        let seq_q = 1;
        let seq_kv = 128;
        let head_dim = 1;
        let val_dim = 1;

        let q_data = vec![1.0f32];
        let mut k_data = vec![0.0f32; seq_kv];
        for k in k_data.iter_mut().take(64) {
            *k = 0.1;
        }
        for k in k_data.iter_mut().skip(64) {
            *k = 5.0;
        }
        let mut v_data = vec![0.0f32; seq_kv];
        for v in v_data.iter_mut().skip(64) {
            *v = 1.0;
        }

        let q = flex_f32(q_data, &[1, 1, seq_q, head_dim]);
        let k = flex_f32(k_data, &[1, 1, seq_kv, head_dim]);
        let v = flex_f32(v_data, &[1, 1, seq_kv, val_dim]);

        let result = super::attention(q, k, v, None, None, Default::default());
        let data: &[f32] = result.storage();

        // Large-score keys (tile 2) dominate softmax -> output ~ 1.0.
        assert!(data[0] > 0.99, "expected ~1.0, got {}", data[0]);
    }

    #[test]
    fn test_partial_last_tile() {
        // seq_kv=100 is not a multiple of TILE_KV=64, so the last tile has
        // 36 elements -> exercises partial-tile gemm and score buffer
        // indexing.
        let seq_q = 2;
        let seq_kv = 100;
        let head_dim = 2;
        let val_dim = 1;

        let q_data = vec![0.1f32, 0.1].repeat(seq_q);
        let k_data = vec![0.1f32, 0.1].repeat(seq_kv);
        let v_data: Vec<f32> = (0..seq_kv).map(|i| i as f32).collect();

        let q = flex_f32(q_data, &[1, 1, seq_q, head_dim]);
        let k = flex_f32(k_data, &[1, 1, seq_kv, head_dim]);
        let v = flex_f32(v_data, &[1, 1, seq_kv, val_dim]);

        let result = super::attention(q, k, v, None, None, Default::default());
        let data: &[f32] = result.storage();

        // Uniform attention -> mean of 0..99 = 49.5.
        assert_eq!(data.len(), seq_q);
        assert!(
            (data[0] - 49.5).abs() < 0.1,
            "expected ~49.5, got {}",
            data[0]
        );
        assert!(
            (data[1] - 49.5).abs() < 0.1,
            "expected ~49.5, got {}",
            data[1]
        );
    }

    /// Verify naive attention produces the same results as flash attention
    /// across various configurations.
    #[test]
    fn test_naive_matches_flash() {
        /// Deterministic tensor for cross-validation tests.
        ///
        /// Uses `i * 997 % N` as a cheap hash: 997 is prime and coprime to
        /// any power-of-two length, so the sequence visits all residues
        /// before repeating. For f32 this gives values in [-0.5, 0.5]. For
        /// bool masks it gives ~30% density with an irregular pattern that
        /// varies across rows (unlike a regular `i % 3` stride).
        fn make_tensor(shape: &[usize], dtype: DType) -> HostTensor {
            let len: usize = shape.iter().product();
            match dtype {
                DType::F32 => {
                    let data: Vec<f32> =
                        (0..len).map(|i| ((i % 997) as f32 / 997.0) - 0.5).collect();
                    flex_f32(data, shape)
                }
                DType::Bool(_) => {
                    let data: Vec<u8> = (0..len)
                        .map(|i| (i.wrapping_mul(997) % 100 < 30) as u8)
                        .collect();
                    flex_bool(data, shape)
                }
                _ => unreachable!(),
            }
        }

        fn run_both(
            batch: usize,
            heads: usize,
            seq_q: usize,
            seq_kv: usize,
            head_dim: usize,
            val_dim: usize,
            with_mask: bool,
            with_bias: bool,
            options: AttentionModuleOptions,
            label: &str,
        ) {
            let f32_dt = DType::F32;
            let bool_dt = DType::Bool(BoolStore::Native);
            let q = make_tensor(&[batch, heads, seq_q, head_dim], f32_dt);
            let k = make_tensor(&[batch, heads, seq_kv, head_dim], f32_dt);
            let v = make_tensor(&[batch, heads, seq_kv, val_dim], f32_dt);
            let score_shape = [batch, heads, seq_q, seq_kv];
            let mask = with_mask.then(|| make_tensor(&score_shape, bool_dt));
            let bias = with_bias.then(|| make_tensor(&score_shape, f32_dt));

            let flash = super::attention_flash(
                q.clone(),
                k.clone(),
                v.clone(),
                mask.clone(),
                bias.clone(),
                options,
            );
            let naive = super::attention_naive(q, k, v, mask, bias, options);

            let flash_data: &[f32] = flash.storage();
            let naive_data: &[f32] = naive.storage();
            assert_eq!(
                flash_data.len(),
                naive_data.len(),
                "{label}: length mismatch"
            );

            for (i, (&f, &n)) in flash_data.iter().zip(naive_data.iter()).enumerate() {
                let diff = (f - n).abs();
                let tol = 1e-4 * f.abs().max(n.abs()).max(1.0);
                assert!(
                    diff < tol,
                    "{label}: position {i}: flash={f} vs naive={n}, diff={diff}"
                );
            }
        }

        let default = AttentionModuleOptions::default();
        let causal = AttentionModuleOptions {
            is_causal: true,
            ..Default::default()
        };
        let all_opts = AttentionModuleOptions {
            scale: Some(0.05),
            softcap: Some(30.0),
            is_causal: true,
        };

        run_both(1, 1, 4, 4, 8, 8, false, false, default, "basic_4x4");
        run_both(
            2,
            4,
            8,
            8,
            16,
            16,
            false,
            false,
            default,
            "multi_head_batch",
        );
        run_both(1, 2, 4, 32, 16, 16, false, false, default, "cross_attn");
        run_both(1, 1, 4, 128, 16, 16, false, false, default, "multi_tile");
        run_both(1, 2, 16, 16, 32, 32, false, false, causal, "causal");
        run_both(2, 2, 16, 16, 32, 32, false, false, all_opts, "all_options");
        run_both(1, 1, 32, 256, 64, 64, false, false, causal, "large_causal");
        run_both(1, 2, 8, 8, 16, 16, true, false, default, "with_mask");
        run_both(1, 2, 8, 8, 16, 16, false, true, default, "with_bias");
        run_both(
            2,
            2,
            16,
            128,
            32,
            32,
            true,
            true,
            causal,
            "mask_bias_causal",
        );
        run_both(1, 1, 4, 100, 16, 16, false, false, default, "partial_tile");
        run_both(
            1,
            2,
            8,
            100,
            16,
            16,
            false,
            false,
            causal,
            "partial_tile_causal",
        );
    }

    #[test]
    fn test_f64_flash_attention() {
        let q = flex_f64(vec![1.0f64, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let k = q.clone();
        let v = flex_f64(vec![10.0f64, 20.0], &[1, 1, 2, 1]);

        let result = super::attention(q, k, v, None, None, Default::default());
        let data: &[f64] = result.storage();

        assert!((data[0] - 13.30).abs() < 0.1, "got {}", data[0]);
        assert!((data[1] - 16.70).abs() < 0.1, "got {}", data[1]);
    }

    /// Verify f16 cast-to-f32 round-trip produces correct results for both
    /// paths.
    #[test]
    fn test_f16_attention() {
        let q_vec: Vec<f16> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let v_vec: Vec<f16> = [10.0f32, 20.0].iter().map(|&v| f16::from_f32(v)).collect();
        let q = flex_f16(q_vec, &[1, 1, 2, 2]);
        let k = q.clone();
        let v = flex_f16(v_vec, &[1, 1, 2, 1]);

        let flash = super::attention_flash(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            None,
            Default::default(),
        );
        let naive = super::attention_naive(q, k, v, None, None, Default::default());

        let flash_data: &[f16] = flash.storage();
        let naive_data: &[f16] = naive.storage();

        // softmax([1/sqrt(2), 0]) = [0.670, 0.330] -> row0: ~13.3, row1: ~16.7
        for (label, data) in [("flash", flash_data), ("naive", naive_data)] {
            let r0 = data[0].to_f32();
            let r1 = data[1].to_f32();
            assert!((r0 - 13.30).abs() < 0.2, "{label} row0: got {r0}");
            assert!((r1 - 16.70).abs() < 0.2, "{label} row1: got {r1}");
        }
    }
