pub use ruprim_host::reduce::*;

// ============================================================================
// Tests
// ============================================================================

// Tests kept here exercise flex-specific behavior: dtype storage selection
// for every numeric width (i8/i16/i32/i64/u8/u16/u32/u64, bf16/f16/f32/f64)
// and edge cases of the dtype-specific reduction kernels (zero-sized
// non-reduced dims across contiguous/half/widening paths, dim sizes that
// exceed the element's max, f16 mean overflow fusion, zero-size panics on
// max_dim/min_dim). Plain 1d/2d smokes, NaN propagation, negative-stride
// (flipped/transposed/narrowed) variants, 4D middle-dim reductions, and
// permuted-argmax regressions have been migrated to ruda-backend-tests so
// they run against every backend. When adding new tests, keep them here
// only if they probe flex dtype storage or panic on flex internals;
// otherwise add them to crates/ruda-backend-tests/tests/tensor/{float,int}/ops/.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use ruda_tensor::TensorData;
    use ruda_tensor::ops::{FloatTensorOps, IntTensorOps};
    use ruda_tensor::{bf16, f16};

    use crate::{Host, HostTensor};

    #[test]
    fn test_mean_f16_overflow_intermediate_sum() {
        // Scalar `mean()` for f16 must fuse sum+divide on the f32 accumulator.
        // Sum of 0..1024 is 523776, well above f16::MAX (65504), so a naive
        // sum-then-divide that materialises the intermediate in f16 would clip
        // to inf. The final mean (511.5) fits f16 comfortably.
        let data: Vec<f16> = (0..1024).map(|i| f16::from_f32(i as f32)).collect();
        let tensor = HostTensor::from_data(TensorData::new(data, [1024]));

        let result = Host::float_mean(tensor);
        let result_data = result.into_data();
        let values: &[f16] = bytemuck::cast_slice(&result_data.bytes);

        assert_eq!(values.len(), 1);
        let mean = values[0].to_f32();
        assert!(mean.is_finite(), "mean overflowed to {mean}");
        assert!((mean - 511.5).abs() < 0.5, "expected ~511.5, got {mean}");
    }

    #[test]
    fn test_mean_dim_f16_zero_outer_dim() {
        // Regression test for mean_dim_half / sum_dim_contiguous_f32 clamping
        // `outer_size.max(1)` in the last-dim branch: shape [0, 4] reducing
        // dim=1 has outer_size=0, and the old clamp produced rows=1 which
        // ran sum_rows_f32 past the end of an empty buffer. Result should be
        // an empty tensor with shape [0, 1].
        let data: Vec<f16> = Vec::new();
        let tensor = HostTensor::from_data(TensorData::new(data, [0, 4]));
        let result = Host::float_mean_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![0, 1]);
        let result_data = result.into_data();
        let values: &[f16] = bytemuck::cast_slice(&result_data.bytes);
        assert!(values.is_empty());
    }

    #[test]
    fn test_sum_dim_f32_zero_outer_dim() {
        // Regression: the f32 last-dim SIMD path (`reduce_last_dim_f32`) used
        // `rows = outer_size.max(1)` which would index past the end of an
        // empty data buffer for shape [0, K]. Guarded by the `out_size == 0`
        // early return in `reduce_dim_f32`.
        let data: Vec<f32> = Vec::new();
        let tensor = HostTensor::from_data(TensorData::new(data, [0, 4]));
        let result = Host::float_sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![0, 1]);
        assert!(result.into_data().bytes.is_empty());
    }

    #[test]
    fn test_sum_dim_f32_zero_inner_dim() {
        // Mirror of the outer-zero case: shape [3, 0] reducing dim=0 has
        // inner_size=0.
        let data: Vec<f32> = Vec::new();
        let tensor = HostTensor::from_data(TensorData::new(data, [3, 0]));
        let result = Host::float_sum_dim(tensor, 0);

        assert_eq!(result.layout().shape().to_vec(), vec![1, 0]);
        assert!(result.into_data().bytes.is_empty());
    }

    #[test]
    fn test_sum_dim_f64_zero_outer_dim() {
        // Covers `reduce_dim_impl` (generic contiguous/non-contiguous path)
        // for the zero-sized non-reduced dim case.
        let data: Vec<f64> = Vec::new();
        let tensor = HostTensor::from_data(TensorData::new(data, [0, 4]));
        let result = Host::float_sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![0, 1]);
        assert!(result.into_data().bytes.is_empty());
    }

    #[test]
    fn test_sum_dim_i8_zero_outer_dim() {
        // Covers `reduce_dim_widening` (i8/i16/u8/u16 path that accumulates
        // in i64) for the zero-sized non-reduced dim case.
        let data: Vec<i8> = Vec::new();
        let tensor = HostTensor::from_data(TensorData::new(data, [0, 4]));
        let result = Host::int_sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![0, 1]);
        assert!(result.into_data().bytes.is_empty());
    }

    #[test]
    fn test_sum_dim_bf16_zero_outer_dim() {
        // Covers `reduce_dim_half` (bf16/f16 sum_dim/prod_dim path) for the
        // zero-sized non-reduced dim case.
        let data: Vec<bf16> = Vec::new();
        let tensor = HostTensor::from_data(TensorData::new(data, [0, 4]));
        let result = Host::float_sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![0, 1]);
        assert!(result.into_data().bytes.is_empty());
    }

    #[test]
    fn test_mean_dim_i8_large_dimension() {
        // dim_size=200 exceeds i8::MAX (127). Before the fix, 200 as i8 = -56,
        // causing wrong results (or 256 as i8 = 0 causing div-by-zero).
        let mut data: Vec<i8> = vec![0i8; 200];
        data[0] = 100;
        let tensor = HostTensor::from_data(TensorData::new(data, [1, 200]));
        let result = Host::int_mean_dim(tensor, 1);

        let result_data = result.into_data();
        let values: Vec<i8> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        // integer division: 100 / 200 = 0
        assert_eq!(values, vec![0]);
    }

    #[test]
    fn test_mean_dim_i16_large_dimension() {
        // dim_size=40000 exceeds i16::MAX (32767).
        let mut data: Vec<i16> = vec![0i16; 40000];
        data[0] = 32000;
        let tensor = HostTensor::from_data(TensorData::new(data, [1, 40000]));
        let result = Host::int_mean_dim(tensor, 1);

        let result_data = result.into_data();
        let values: Vec<i16> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![0]);
    }

    #[test]
    fn test_sum_i32() {
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let tensor = HostTensor::from_data(TensorData::new(data, [5]));
        let result = Host::int_sum(tensor);

        assert_eq!(result.layout().shape().to_vec(), vec![1]);
        let result_data = result.into_data();
        let values: Vec<i32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![15]);
    }

    #[test]
    fn test_sum_dim_i32() {
        let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        let tensor = HostTensor::from_data(TensorData::new(data, [2, 3]));
        let result = Host::int_sum_dim(tensor, 1);

        assert_eq!(result.layout().shape().to_vec(), vec![2, 1]);
        let result_data = result.into_data();
        let values: Vec<i32> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        assert_eq!(values, vec![6, 15]);
    }

    #[test]
    fn test_argmax_i32() {
        let data: Vec<i32> = vec![1, 5, 3, 2, 4];
        let tensor = HostTensor::from_data(TensorData::new(data, [5]));
        let result = Host::int_argmax(tensor, 0);

        assert_eq!(result.layout().shape().to_vec(), vec![1]);
        let result_data = result.into_data();
        #[cfg(target_pointer_width = "64")]
        let values: Vec<i64> = bytemuck::cast_slice(&result_data.bytes).to_vec();
        #[cfg(target_pointer_width = "32")]
        let values: Vec<i64> = bytemuck::cast_slice::<u8, i32>(&result_data.bytes)
            .iter()
            .map(|&v| v as i64)
            .collect();
        assert_eq!(values, vec![1]);
    }

    #[test]
    #[should_panic(expected = "dimension 0 has size 0")]
    fn test_max_dim_zero_size_panics() {
        let tensor = HostTensor::from_data(TensorData::new(Vec::<f32>::new(), [0, 3]));
        Host::float_max_dim(tensor, 0);
    }

    #[test]
    #[should_panic(expected = "dimension 1 has size 0")]
    fn test_min_dim_zero_size_panics() {
        let tensor = HostTensor::from_data(TensorData::new(Vec::<f32>::new(), [3, 0]));
        Host::float_min_dim(tensor, 1);
    }

    // === Unsigned integer dtype tests ===

    #[test]
    fn test_sum_u32() {
        let tensor = HostTensor::from_data(TensorData::new(vec![10u32, 20, 30], [3]));
        let result = Host::int_sum(tensor);
        let data: Vec<u32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![60]);
    }

    #[test]
    fn test_sum_u64() {
        let tensor = HostTensor::from_data(TensorData::new(vec![100u64, 200, 300], [3]));
        let result = Host::int_sum(tensor);
        let data: Vec<u64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![600]);
    }

    #[test]
    fn test_sum_dim_u8() {
        let tensor = HostTensor::from_data(TensorData::new(vec![1u8, 2, 3, 4], [2, 2]));
        let result = Host::int_sum_dim(tensor, 1);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![3, 7]);
    }

    #[test]
    fn test_prod_u16() {
        let tensor = HostTensor::from_data(TensorData::new(vec![2u16, 3, 5], [3]));
        let result = Host::int_prod(tensor);
        let data: Vec<u16> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![30]);
    }

    #[test]
    fn test_max_u32() {
        let tensor = HostTensor::from_data(TensorData::new(vec![5u32, 100, 42], [3]));
        let result = Host::int_max(tensor);
        let data: Vec<u32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![100]);
    }

    #[test]
    fn test_min_u8() {
        let tensor = HostTensor::from_data(TensorData::new(vec![5u8, 1, 42], [3]));
        let result = Host::int_min(tensor);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1]);
    }

    #[test]
    fn test_max_dim_u64() {
        let tensor = HostTensor::from_data(TensorData::new(vec![10u64, 20, 30, 5], [2, 2]));
        let result = Host::int_max_dim(tensor, 1);
        let data: Vec<u64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![20, 30]);
    }

    #[test]
    fn test_min_dim_u16() {
        let tensor = HostTensor::from_data(TensorData::new(vec![10u16, 2, 30, 5], [2, 2]));
        let result = Host::int_min_dim(tensor, 1);
        let data: Vec<u16> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![2, 5]);
    }

    #[test]
    fn test_mean_dim_u8() {
        let tensor = HostTensor::from_data(TensorData::new(vec![10u8, 20, 30, 40], [2, 2]));
        let result = Host::int_mean_dim(tensor, 1);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![15, 35]);
    }

    #[test]
    fn test_max_dim_with_indices_u32() {
        let tensor = HostTensor::from_data(TensorData::new(vec![5u32, 10, 3, 8], [2, 2]));
        let (values, indices) = Host::int_max_dim_with_indices(tensor, 1);
        let vals: Vec<u32> = values.into_data().to_vec().unwrap();
        let idxs: Vec<isize> = bytemuck::cast_slice(&indices.into_data().bytes).to_vec();
        assert_eq!(vals, vec![10, 8]);
        assert_eq!(idxs, vec![1, 1]);
    }

    // Cross-path consistency: `argmax`/`argmin` route short rows to the
    // scalar kernel and rows of length >= EXTREMUM_SIMD_ROW_THRESHOLD to
    // the SIMD kernel. Both kernels must agree on "first NaN wins".
    // This is a flex-internal dispatch concern; the behavioral NaN
    // propagation contract itself is exercised in ruda-backend-tests
    // under the `flex` feature gate (see issue #4814).
    #[test]
    fn test_argmax_scalar_and_simd_paths_agree_on_leading_nan() {
        let short =
            HostTensor::from_data(TensorData::new(vec![f32::NAN, f32::NAN, f32::NAN], [1, 3]));
        let short_idxs: Vec<isize> =
            bytemuck::cast_slice(&super::argmax(short, 1).into_data().bytes).to_vec();

        let mut long_data = alloc::vec![1.0f32; 600];
        long_data[0] = f32::NAN;
        long_data[1] = f32::NAN;
        long_data[300] = 5.0;
        let long = HostTensor::from_data(TensorData::new(long_data, [1, 600]));
        let long_idxs: Vec<isize> =
            bytemuck::cast_slice(&super::argmax(long, 1).into_data().bytes).to_vec();

        assert_eq!(short_idxs, vec![0], "scalar path");
        assert_eq!(long_idxs, vec![0], "SIMD path");
    }
}
