    use super::*;
    use alloc::vec;

    #[test]
    fn test_from_data_roundtrip() {
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tensor = HostTensor::from_data(data.clone());
        let result = tensor.into_data();
        assert_eq!(data.shape, result.shape);
        assert_eq!(data.dtype, result.dtype);
    }

    #[test]
    fn test_collapse_for_copy_squeezes_size1_and_merges_contig() {
        // Permuted ConvNeXt input: [1, 48, 244, 224].permute([0,2,3,1]).
        let shape = vec![1, 244, 224, 48];
        let strides = vec![2_623_488_isize, 224, 1, 54656];
        let collapsed = collapse_for_copy(&shape, &strides);
        let (s, st) = collapsed.as_slices();
        assert_eq!(s, &[54656, 48]);
        assert_eq!(st, &[1, 54656]);
    }

    #[test]
    fn test_collapse_for_copy_already_contiguous_3d() {
        let collapsed = collapse_for_copy(&[2, 3, 4], &[12, 4, 1]);
        let (s, st) = collapsed.as_slices();
        assert_eq!(s, &[24]);
        assert_eq!(st, &[1]);
    }

    #[test]
    fn test_collapse_for_copy_transpose_2d() {
        let collapsed = collapse_for_copy(&[5, 3], &[1, 5]);
        let (s, st) = collapsed.as_slices();
        assert_eq!(s, &[5, 3]);
        assert_eq!(st, &[1, 5]);
    }

    #[test]
    fn test_collapse_for_copy_all_size1() {
        let collapsed = collapse_for_copy(&[1, 1, 1], &[0, 0, 0]);
        let (s, st) = collapsed.as_slices();
        assert!(s.is_empty());
        assert!(st.is_empty());
    }

    /// Regression: an empty 1D view produced by `narrow` at a
    /// non-zero offset forces `copy_contiguous` to run (it can't
    /// early-return via the contiguous-at-offset-0 shortcut). The
    /// old `debug_assert_eq!(n, shape.product().max(1))` tripped
    /// for this shape because `.max(1)` produced 1 while the true
    /// numel is 0.
    #[test]
    fn test_to_contiguous_zero_sized_narrowed() {
        let t = HostTensor::from_data(TensorData::new(
            (0..6).map(|i| i as f32).collect::<Vec<_>>(),
            vec![6],
        ));
        // narrow(dim, start=3, len=0): shape [0], start_offset 3.
        let empty_view = t.narrow(0, 3, 0);
        assert_eq!(empty_view.shape().to_vec(), vec![0]);
        assert_ne!(empty_view.layout().start_offset(), 0);

        let contig = empty_view.to_contiguous();
        assert_eq!(contig.shape().to_vec(), vec![0]);
        assert_eq!(contig.layout().start_offset(), 0);
        assert_eq!(contig.into_data().bytes.len(), 0);
    }

    /// Regression for #4855: a prefix view (e.g. `narrow(dim, 0, n)`) has
    /// canonical contiguous strides and start_offset 0, but its underlying
    /// buffer is still the larger original. `to_contiguous` must materialize
    /// a right-sized copy so callers keying off `storage().len()` (like the
    /// SIMD `mask_fill_*` kernels reached from `triu`/`tril` in LU on tall
    /// matrices) don't walk past the logical shape.
    #[test]
    fn test_to_contiguous_prefix_view_shrinks_buffer() {
        let data: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let t = HostTensor::from_data(TensorData::new(data, vec![8, 5]));

        let prefix = t.narrow(0, 0, 5);
        assert_eq!(prefix.shape().to_vec(), vec![5, 5]);
        assert_eq!(prefix.layout().strides(), &[5, 1]);
        assert_eq!(prefix.layout().start_offset(), 0);
        assert!(prefix.is_contiguous());
        assert_eq!(prefix.storage::<f32>().len(), 40);

        let contig = prefix.to_contiguous();
        assert_eq!(contig.storage::<f32>().len(), 25);
        assert_eq!(contig.layout().num_elements(), 25);
        assert_eq!(
            contig.storage::<f32>(),
            &(0..5)
                .flat_map(|r| (0..5).map(move |c| (r * 5 + c) as f32))
                .collect::<Vec<_>>()[..]
        );
    }

    /// 4D permuted layout round-trips through the collapse + tiled
    /// copy path. Mirrors the ConvNeXt channels-last permute.
    #[test]
    fn test_to_contiguous_4d_permuted_matches_naive() {
        let dims = [1, 48, 4, 5];
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t = HostTensor::from_data(TensorData::new(data.clone(), dims.to_vec()));
        let permuted = t.permute(&[0, 2, 3, 1]);
        assert!(!permuted.is_contiguous());

        let contig = permuted.to_contiguous();
        assert!(contig.is_contiguous());
        assert_eq!(contig.shape().to_vec(), vec![1, 4, 5, 48]);

        // Expected via manual strided walk of the source.
        let mut expected = Vec::with_capacity(n);
        for h in 0..4 {
            for w in 0..5 {
                for c in 0..48 {
                    let idx = c * 20 + h * 5 + w;
                    expected.push(data[idx]);
                }
            }
        }

        let result_data = contig.into_data();
        let values = result_data.as_slice::<f32>().unwrap();
        assert_eq!(values, expected.as_slice());
    }



    #[test]
    fn test_reshape() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = HostTensor::from_data(data);
        let reshaped = tensor.reshape(Shape::from(vec![3, 2]));
        assert_eq!(reshaped.shape().to_vec(), vec![3, 2]);
    }

    #[test]
    fn test_transpose() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = HostTensor::from_data(data);
        let transposed = tensor.transpose(0, 1);
        assert_eq!(transposed.shape().to_vec(), vec![3, 2]);
        assert!(!transposed.is_contiguous());
    }

    #[test]
    fn test_clone_is_cheap() {
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
        let tensor = HostTensor::from_data(data);

        // Before clone, tensor is unique
        assert!(tensor.is_unique());

        // Clone shares data
        let cloned = tensor.clone();
        assert!(!tensor.is_unique());
        assert!(!cloned.is_unique());

        // Both point to same data
        assert!(core::ptr::eq(
            tensor.bytes().as_ptr(),
            cloned.bytes().as_ptr()
        ));
    }

    #[test]
    fn test_cow_on_mutation() {
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
        let tensor = HostTensor::from_data(data);
        let mut cloned = tensor.clone();

        // Both share data
        assert!(!tensor.is_unique());
        assert!(!cloned.is_unique());

        // Mutate cloned - triggers COW
        let storage: &mut [f32] = cloned.storage_mut();
        storage[0] = 99.0;

        // Now cloned has its own copy, tensor is unique again
        assert!(tensor.is_unique());
        assert!(cloned.is_unique());

        // Data is different
        assert_ne!(tensor.bytes().as_ptr(), cloned.bytes().as_ptr());
        assert_eq!(tensor.storage::<f32>()[0], 1.0);
        assert_eq!(cloned.storage::<f32>()[0], 99.0);
    }

    #[test]
    fn test_into_data_narrowed_at_offset_zero() {
        // [1, 2, 3, 4, 5, 6] shape [2, 3]
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tensor = HostTensor::from_data(data);
        // narrow to first row: shape [1, 3], offset 0, contiguous
        let narrowed = tensor.narrow(0, 0, 1);
        assert!(narrowed.is_contiguous());
        assert_eq!(narrowed.layout().start_offset(), 0);

        let result = narrowed.into_data();
        assert_eq!(result.shape.to_vec(), vec![1, 3]);
        // Must have exactly 3 f32s = 12 bytes, not 24
        assert_eq!(result.bytes.len(), 3 * core::mem::size_of::<f32>());
        let values: Vec<f32> = result.to_vec().unwrap();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }


    #[test]
    fn safety_strided_vector_copy_initializes_every_element() {
        let tensor = HostTensor::new(
            Bytes::from_elems((0..8).map(|x| x as i32).collect::<Vec<_>>()),
            Layout::new(Shape::from(vec![4]), vec![2], 0),
            DType::I32,
        );
        assert_eq!(tensor.to_contiguous().storage::<i32>(), &[0, 2, 4, 6]);
    }

    #[test]
    fn safety_tiled_copy_preserves_float_bit_patterns() {
        let bits = [0x8000_0000_u32, 0x7fc0_1234, 0x7f80_0000, 0xff80_0000, 0, 0x3f80_0000];
        let tensor = HostTensor::from_data(TensorData::new(
            bits.iter().copied().map(f32::from_bits).collect::<Vec<_>>(), vec![2, 3],
        ));
        let result = tensor.transpose(0, 1).to_contiguous();
        let result_bits = result.storage::<f32>().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(result_bits, vec![bits[0], bits[3], bits[1], bits[4], bits[2], bits[5]]);
    }

    #[test]
    fn safety_negative_stride_nd_copy_matches_logical_order() {
        let tensor = HostTensor::new(
            Bytes::from_elems((0..8).collect::<Vec<i32>>()),
            Layout::new(Shape::from(vec![2, 2, 2]), vec![-4, 1, 2], 4),
            DType::I32,
        );
        assert_eq!(tensor.to_contiguous().storage::<i32>(), &[4, 6, 5, 7, 0, 2, 1, 3]);
    }

    #[test]
    fn safety_scalar_copy_at_nonzero_offset() {
        let tensor = HostTensor::new(
            Bytes::from_elems(vec![11_i32, 42]),
            Layout::new(Shape::from(Vec::<usize>::new()), vec![], 1),
            DType::I32,
        );
        assert_eq!(tensor.to_contiguous().storage::<i32>(), &[42]);
    }
