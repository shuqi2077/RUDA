    use super::*;

    #[test]
    fn test_add_inplace_f32() {
        let mut a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        add_inplace_f32(&mut a, &b);
        assert_eq!(a, [11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0]);
    }

    #[test]
    fn test_sub_inplace_f32() {
        let mut a = [10.0f32, 20.0, 30.0, 40.0, 50.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        sub_inplace_f32(&mut a, &b);
        assert_eq!(a, [9.0, 18.0, 27.0, 36.0, 45.0]);
    }

    #[test]
    fn test_mul_inplace_f32() {
        let mut a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0f32, 2.0, 2.0, 2.0, 2.0];
        mul_inplace_f32(&mut a, &b);
        assert_eq!(a, [2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_div_inplace_f32() {
        let mut a = [10.0f32, 20.0, 30.0, 40.0];
        let b = [2.0f32, 4.0, 5.0, 8.0];
        div_inplace_f32(&mut a, &b);
        assert_eq!(a, [5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_cmp_gt_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [2.0f32, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0];
        let mut out = [0u8; 7];
        cmp_f32(&a, &b, &mut out, CmpOp::Gt);
        assert_eq!(out, [0, 0, 1, 0, 1, 1, 1]);
    }

    #[test]
    fn test_cmp_ge_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [2.0f32, 2.0, 2.0, 5.0];
        let mut out = [0u8; 4];
        cmp_f32(&a, &b, &mut out, CmpOp::Ge);
        assert_eq!(out, [0, 1, 1, 0]);
    }

    #[test]
    fn test_cmp_eq_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 3.0, 3.0, 5.0, 5.0];
        let mut out = [0u8; 5];
        cmp_f32(&a, &b, &mut out, CmpOp::Eq);
        assert_eq!(out, [1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_cmp_ne_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0f32, 3.0, 3.0, 5.0, 5.0];
        let mut out = [0u8; 5];
        cmp_f32(&a, &b, &mut out, CmpOp::Ne);
        assert_eq!(out, [0, 1, 0, 1, 0]);
    }

    #[test]
    fn test_cmp_scalar_gt_f32() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = [0u8; 5];
        cmp_scalar_f32(&a, 3.0, &mut out, CmpOp::Gt);
        assert_eq!(out, [0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_bool_not_u8() {
        let a = [1u8, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0];
        let mut out = [0u8; 18];
        bool_not_u8(&a, &mut out);
        let expected = [0u8, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_not_inplace_u8() {
        let mut a = [1u8, 0, 1, 0];
        bool_not_inplace_u8(&mut a);
        assert_eq!(a, [0, 1, 0, 1]);
    }

    #[test]
    fn test_bool_and_u8() {
        let a = [1u8, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        let b = [1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1];
        let mut out = [0u8; 18];
        bool_and_u8(&a, &b, &mut out);
        let expected = [1u8, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_or_u8() {
        let a = [1u8, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        let b = [1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1];
        let mut out = [0u8; 18];
        bool_or_u8(&a, &b, &mut out);
        let expected = [1u8, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_xor_u8() {
        let a = [1u8, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0];
        let b = [1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1];
        let mut out = [0u8; 18];
        bool_xor_u8(&a, &b, &mut out);
        let expected = [0u8, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1];
        assert_eq!(out, expected);
    }

    #[test]
    fn test_bool_and_inplace_u8() {
        let mut a = [1u8, 1, 0, 0];
        let b = [1u8, 0, 1, 0];
        bool_and_inplace_u8(&mut a, &b);
        assert_eq!(a, [1, 0, 0, 0]);
    }

    #[test]
    fn test_bool_or_inplace_u8() {
        let mut a = [1u8, 1, 0, 0];
        let b = [1u8, 0, 1, 0];
        bool_or_inplace_u8(&mut a, &b);
        assert_eq!(a, [1, 1, 1, 0]);
    }

    #[test]
    fn test_bool_xor_inplace_u8() {
        let mut a = [1u8, 1, 0, 0];
        let b = [1u8, 0, 1, 0];
        bool_xor_inplace_u8(&mut a, &b);
        assert_eq!(a, [0, 1, 1, 0]);
    }

    #[test]
    fn test_abs_inplace_f32() {
        let mut a = [-3.0f32, -1.0, 0.0, 1.0, 3.0, -5.0, 7.0];
        abs_inplace_f32(&mut a);
        assert_eq!(a, [3.0, 1.0, 0.0, 1.0, 3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_recip_inplace_f32() {
        let mut a = [1.0f32, 2.0, 4.0, 0.5, 10.0];
        recip_inplace_f32(&mut a);
        assert_eq!(a, [1.0, 0.5, 0.25, 2.0, 0.1]);
    }

    // ================================================================
    // mask_where / mask_fill tests
    // ================================================================

    #[test]
    fn test_mask_where_f32_basic() {
        let tensor = [1.0f32, 2.0, 3.0, 4.0];
        let mask = [1u8, 0, 1, 0];
        let value = [10.0f32, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        mask_where_f32(&tensor, &mask, &value, &mut out);
        assert_eq!(out, [10.0, 2.0, 30.0, 4.0]);
    }

    #[test]
    fn test_mask_where_f32_all_true() {
        let tensor = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = [1u8; 9];
        let value = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let mut out = [0.0f32; 9];
        mask_where_f32(&tensor, &mask, &value, &mut out);
        assert_eq!(out, value);
    }

    #[test]
    fn test_mask_where_f32_all_false() {
        let tensor = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mask = [0u8; 9];
        let value = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let mut out = [0.0f32; 9];
        mask_where_f32(&tensor, &mask, &value, &mut out);
        assert_eq!(out, tensor);
    }

    #[test]
    fn test_mask_where_f64_basic() {
        let tensor = [1.0f64, 2.0, 3.0];
        let mask = [0u8, 1, 0];
        let value = [10.0f64, 20.0, 30.0];
        let mut out = [0.0f64; 3];
        mask_where_f64(&tensor, &mask, &value, &mut out);
        assert_eq!(out, [1.0, 20.0, 3.0]);
    }

    #[test]
    fn test_mask_where_i64_basic() {
        let tensor = [10i64, 20, 30, 40, 50];
        let mask = [1u8, 0, 1, 0, 1];
        let value = [-1i64, -2, -3, -4, -5];
        let mut out = [0i64; 5];
        mask_where_i64(&tensor, &mask, &value, &mut out);
        assert_eq!(out, [-1, 20, -3, 40, -5]);
    }

    #[test]
    fn test_mask_where_u8_basic() {
        let tensor = [0u8, 1, 0, 1];
        let mask = [1u8, 1, 0, 0];
        let value = [1u8, 0, 1, 0];
        let mut out = [0u8; 4];
        mask_where_u8(&tensor, &mask, &value, &mut out);
        assert_eq!(out, [1, 0, 0, 1]);
    }

    #[test]
    fn test_mask_fill_f32_basic() {
        let tensor = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let mask = [1u8, 0, 1, 0, 1, 0, 1];
        let mut out = [0.0f32; 7];
        mask_fill_f32(&tensor, &mask, -1.0, &mut out);
        assert_eq!(out, [-1.0, 2.0, -1.0, 4.0, -1.0, 6.0, -1.0]);
    }

    #[test]
    fn test_mask_fill_f64_basic() {
        let tensor = [1.0f64, 2.0, 3.0];
        let mask = [0u8, 1, 0];
        let mut out = [0.0f64; 3];
        mask_fill_f64(&tensor, &mask, 99.0, &mut out);
        assert_eq!(out, [1.0, 99.0, 3.0]);
    }

    #[test]
    fn test_mask_fill_i64_basic() {
        let tensor = [10i64, 20, 30, 40];
        let mask = [1u8, 0, 0, 1];
        let mut out = [0i64; 4];
        mask_fill_i64(&tensor, &mask, -1, &mut out);
        assert_eq!(out, [-1, 20, 30, -1]);
    }

    #[test]
    fn test_mask_fill_u8_basic() {
        let tensor = [0u8, 1, 0, 1, 0];
        let mask = [1u8, 1, 0, 0, 1];
        let mut out = [0u8; 5];
        mask_fill_u8(&tensor, &mask, 1, &mut out);
        assert_eq!(out, [1, 1, 0, 1, 1]);
    }

    #[test]
    fn test_mask_where_f32_nan() {
        let tensor = [f32::NAN, 2.0, 3.0, f32::NAN];
        let mask = [1u8, 0, 1, 0];
        let value = [10.0f32, 20.0, 30.0, 40.0];
        let mut out = [0.0f32; 4];
        mask_where_f32(&tensor, &mask, &value, &mut out);
        // mask=1 picks value, mask=0 picks tensor (including NaN)
        assert_eq!(out[0], 10.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 30.0);
        assert!(out[3].is_nan());
    }

    // Lane-boundary tests: sizes that exercise SIMD + scalar tail on all ISAs.
    // 17 elements for f32 = 4 NEON iters + 1 tail, or 2 AVX2 iters + 1 tail.

    #[test]
    fn test_mask_where_f32_lane_boundary() {
        let n = 17;
        let tensor: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let value: Vec<f32> = (0..n).map(|i| (i as f32) * 10.0).collect();
        let mask: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let mut out = vec![0.0f32; n];
        mask_where_f32(&tensor, &mask, &value, &mut out);
        for i in 0..n {
            let expected = if i % 2 != 0 { value[i] } else { tensor[i] };
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_mask_fill_f32_lane_boundary() {
        let n = 17;
        let tensor: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mask: Vec<u8> = (0..n).map(|i| (i % 3 == 0) as u8).collect();
        let mut out = vec![0.0f32; n];
        mask_fill_f32(&tensor, &mask, -1.0, &mut out);
        for i in 0..n {
            let expected = if i % 3 == 0 { -1.0 } else { tensor[i] };
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_mask_where_u8_lane_boundary() {
        // 33 elements for u8: exercises 2 NEON iters + 1 tail, or 1 AVX2 iter + 1 tail
        let n = 33;
        let tensor: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let value: Vec<u8> = (0..n).map(|i| ((i + 1) % 2) as u8).collect();
        let mask: Vec<u8> = (0..n).map(|i| (i % 3 == 0) as u8).collect();
        let mut out = vec![0u8; n];
        mask_where_u8(&tensor, &mask, &value, &mut out);
        for i in 0..n {
            let expected = if i % 3 == 0 { value[i] } else { tensor[i] };
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_mask_where_f64_lane_boundary() {
        // 9 elements for f64: 4 NEON iters + 1 tail (2 lanes), or 2 AVX2 iters + 1 tail (4 lanes)
        let n = 9;
        let tensor: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let value: Vec<f64> = (0..n).map(|i| (i as f64) * -1.0).collect();
        let mask: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
        let mut out = vec![0.0f64; n];
        mask_where_f64(&tensor, &mask, &value, &mut out);
        for i in 0..n {
            let expected = if i % 2 != 0 { value[i] } else { tensor[i] };
            assert_eq!(out[i], expected, "mismatch at index {i}");
        }
    }

    #[test]
    fn test_mask_where_empty() {
        let mut out = vec![0.0f32; 0];
        mask_where_f32(&[], &[], &[], &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_mask_fill_empty() {
        let mut out = vec![0.0f32; 0];
        mask_fill_f32(&[], &[], 1.0, &mut out);
        assert!(out.is_empty());
    }

    // bool_not_u8 writes into a `*mut bool` via macerator's store_as_bool.
    // Rust's bool is only valid as 0x00 or 0x01 (any other byte is UB when
    // read back as bool). A previous audit flagged that SIMD mask stores
    // might emit 0xFF. Verify every output byte is normalized to 0/1.
    #[test]
    fn bool_not_u8_output_is_normalized_0_or_1() {
        // Spans SIMD body + scalar tail on any realistic lane width:
        //   17 elements -> NEON 16-byte SIMD + 1 tail; AVX2 32 spills to tail.
        //   127 elements -> SIMD body + 15 tail for 16-byte lanes.
        for &len in &[1usize, 8, 15, 16, 17, 31, 32, 63, 127, 256] {
            let a: Vec<u8> = (0..len).map(|i| (i % 2) as u8).collect();
            let mut out = vec![0xAAu8; len];
            super::bool_not_u8(&a, &mut out);
            for (i, &b) in out.iter().enumerate() {
                assert!(
                    b == 0 || b == 1,
                    "len={}: out[{}] = 0x{:02x}, expected 0x00 or 0x01",
                    len,
                    i,
                    b
                );
                let expected = if a[i] == 0 { 1 } else { 0 };
                assert_eq!(
                    b, expected,
                    "len={}: out[{}] = {}, expected {}",
                    len, i, b, expected
                );
            }
        }
    }

    #[test]
    fn bool_not_inplace_u8_output_is_normalized_0_or_1() {
        for &len in &[1usize, 8, 15, 16, 17, 31, 32, 63, 127, 256] {
            let mut a: Vec<u8> = (0..len).map(|i| (i % 2) as u8).collect();
            let original = a.clone();
            super::bool_not_inplace_u8(&mut a);
            for (i, &b) in a.iter().enumerate() {
                assert!(
                    b == 0 || b == 1,
                    "len={}: a[{}] = 0x{:02x}, expected 0x00 or 0x01",
                    len,
                    i,
                    b
                );
                let expected = if original[i] == 0 { 1 } else { 0 };
                assert_eq!(
                    b, expected,
                    "len={}: a[{}] = {}, expected {}",
                    len, i, b, expected
                );
            }
        }
    }

    // Edge cases: empty input, homogeneous all-zero, homogeneous all-one.
    // Homogeneous inputs exercise the SIMD mask-to-byte conversion for
    // all-true and all-false cases, which alternating inputs do not.
    #[test]
    fn bool_not_u8_edge_cases() {
        // Empty input.
        let mut out: Vec<u8> = Vec::new();
        super::bool_not_u8(&[], &mut out);
        assert!(out.is_empty());

        // All zeros -> all ones.
        let a = alloc::vec![0u8; 32];
        let mut out = alloc::vec![0xAAu8; 32];
        super::bool_not_u8(&a, &mut out);
        assert!(out.iter().all(|&b| b == 1));

        // All ones -> all zeros.
        let a = alloc::vec![1u8; 32];
        let mut out = alloc::vec![0xAAu8; 32];
        super::bool_not_u8(&a, &mut out);
        assert!(out.iter().all(|&b| b == 0));
    }
