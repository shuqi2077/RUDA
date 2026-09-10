pub use ruda_core::tensor::host::HostTensor as HostTensor;
pub(crate) use ruda_core::tensor::host::storage::dtype_size;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use ruda_tensor::{TensorData, TensorMetadata};

    /// Exercise the `row_stride > col_stride` branch of the 2D tiled
    /// copy (the ConvNeXt case hits the other branch).
    #[test]
    fn test_to_contiguous_2d_row_stride_gt_col_stride() {
        // `slice(s![..;2, ..])` on a [6, 3] contiguous tensor gives a
        // [3, 3] view with strides [6, 1] that doesn't collapse, so
        // the 2D branch runs with row_stride > col_stride.
        let data: Vec<f32> = (0..18).map(|i| i as f32).collect();
        let t = HostTensor::from_data(TensorData::new(data, vec![6, 3]));
        let stepped = crate::ops::slice::slice(
            t,
            &[
                ruda_tensor::Slice::new(0, Some(6), 2),
                ruda_tensor::Slice::new(0, None, 1),
            ],
        );
        // Verify the layout matches what the branch requires.
        assert_eq!(stepped.layout().shape().to_vec(), vec![3, 3]);
        assert_eq!(stepped.layout().strides(), &[6, 1]);
        assert!(!stepped.layout().is_contiguous());

        let contig = stepped.to_contiguous();
        assert!(contig.is_contiguous());
        assert_eq!(contig.shape().to_vec(), vec![3, 3]);

        let result_data = contig.into_data();
        let values = result_data.as_slice::<f32>().unwrap();
        // Expected: rows 0, 2, 4 of the original 6x3 tensor.
        let expected = vec![
            0.0f32, 1.0, 2.0, // row 0
            6.0, 7.0, 8.0, // row 2
            12.0, 13.0, 14.0, // row 4
        ];
        assert_eq!(values, expected.as_slice());
    }
}
