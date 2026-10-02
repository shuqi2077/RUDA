//! Device-native integer quantization with explicit format checks.
//! Per-tensor indexing preserves integer storage. Per-block indexing uses a
//! dequantize/transform/requantize reference path (possibly additional rounding).
use ruda_tensor::{
    DType, ExecutionError, FloatDType, IntDType, Shape, TensorData, TensorMetadata,
    ops::QTensorOps,
    quantization::{QuantScheme, QuantValue, QuantStore, QuantParam, QuantLevel,
        QuantizedBytes, QuantizationParametersPrimitive, params_shape},
};
use crate::{LibTorch, LibTorchDevice, TchElement, TchTensor, TchQTensor, TchShape, IntoKind};
use super::TchOps;

pub(crate) fn supported_scheme(scheme: &QuantScheme) -> bool {
    matches!(scheme.value, QuantValue::Q8S | QuantValue::Q8F |
        QuantValue::Q4S | QuantValue::Q4F | QuantValue::Q2S | QuantValue::Q2F)
        && scheme.param == QuantParam::F32
        && matches!(scheme.store, QuantStore::Native | QuantStore::PackedU32(0))
        && match scheme.level {
            QuantLevel::Tensor => true,
            QuantLevel::Block(block) => !block.as_slice().contains(&0),
        }
}
fn checked_scheme(scheme: &QuantScheme) -> QuantScheme {
    assert!(supported_scheme(scheme),
        "LibTorch quantization requires integer Q8/Q4/Q2, FP32 scales, nonzero blocks and Native/PackedU32(0) input");
    // Low-bit schemes serialize as actual packed u32 words. Computation uses
    // an unpacked i8 staging tensor; it is NOT a native INT4/INT2 kernel.
    scheme.with_store(if scheme.value.size_bits() < 8 {
        QuantStore::PackedU32(0)
    } else { QuantStore::Native })
}

// Broadcast block scales through interleaved parameter/block axes, then trim
// the padded edge. Partial final blocks are supported in every dimension.
fn expanded_scales(scales: &tch::Tensor, shape: &Shape, level: QuantLevel) -> tch::Tensor {
    match level {
        QuantLevel::Tensor => scales.reshape(vec![1i64; shape.rank()]),
        QuantLevel::Block(block) => {
            let block = block.to_dim_vec(shape.rank());
            let params = params_shape(shape, level);
            let mut source = Vec::new();
            let mut expanded = Vec::new();
            let mut padded = Vec::new();
            for axis in 0..shape.rank() {
                source.extend([params[axis] as i64, 1]);
                expanded.extend([params[axis] as i64, block[axis] as i64]);
                padded.push((params[axis]*block[axis] as usize) as i64);
            }
            let mut result = scales.reshape(source).expand(expanded, false).reshape(padded);
            for axis in 0..shape.rank() { result = result.narrow(axis as i64, 0, shape[axis] as i64); }
            result
        }
    }
}
fn valid_scales(scales: tch::Tensor, shape: &Shape, scheme: &QuantScheme) -> TchTensor {
    let expected = params_shape(shape, scheme.level);
    assert_eq!(scales.numel(), expected.num_elements(), "quantization scale count mismatch");
    let scales = scales.to_kind(tch::Kind::Float);
    assert!(scales.isfinite().logical_and(&scales.ge(0.0)).all().int64_value(&[]) != 0,
        "quantization scales must be finite and nonnegative");
    TchTensor::new(scales.clamp_min(f32::MIN_POSITIVE as f64).reshape(TchShape::from(expected).dims))
}
fn layout_scheme(mut scheme: QuantScheme, rank: usize) -> QuantScheme {
    if let QuantLevel::Block(block) = scheme.level {
        if block.as_slice().len() > rank { scheme = scheme.with_level(QuantLevel::Tensor); }
    }
    scheme
}

/// Produce a wire representation matching DType, including sub-byte packing.
/// TensorData::quantized's convenience constructor stores raw i8s and must not
/// be used for the packed Q4/Q2 case.
fn export_data(values: Vec<i8>, shape: Shape, scheme: QuantScheme, scales: &[f32]) -> TensorData {
    if scheme.value.size_bits() == 8 {
        return TensorData::quantized(values, shape, scheme, scales);
    }
    let bits = scheme.value.size_bits();
    let per_word = 32 / bits;
    let mask = (1u32 << bits) - 1;
    let mut words = values.chunks(per_word).map(|chunk| {
        chunk.iter().enumerate().fold(0u32, |word, (i, value)|
            word | ((*value as u32 & mask) << (i * bits)))
    }).collect::<Vec<_>>();
    words.extend(scales.iter().map(|value| value.to_bits()));
    TensorData { bytes: ruda_tensor::Bytes::from_elems(words), shape,
        dtype: DType::QFloat(scheme.with_store(QuantStore::PackedU32(0))) }
}

impl<E: TchElement> LibTorch<E> {
    fn q_map(tensor: TchQTensor, transform: impl FnOnce(TchTensor) -> TchTensor) -> TchQTensor {
        if tensor.scheme.level == QuantLevel::Tensor {
            return TchQTensor { values: transform(tensor.values), ..tensor };
        }
        let scheme = tensor.scheme;
        let value = transform(Self::dequantize(tensor, FloatDType::F32));
        let scheme = layout_scheme(scheme, value.rank());
        Self::quantize_dynamic(value, &scheme)
    }
    fn q_map_indices(tensor: TchQTensor, transform: impl FnOnce(TchTensor) -> (TchTensor, TchTensor),
        dtype: IntDType) -> (TchQTensor, TchTensor)
    {
        let scheme = tensor.scheme;
        let (quantized, indices) = if scheme.level == QuantLevel::Tensor {
            let (values, indices) = transform(tensor.values);
            (TchQTensor { values, scales: tensor.scales, scheme }, indices)
        } else {
            let (values, indices) = transform(Self::dequantize(tensor, FloatDType::F32));
            let scheme = layout_scheme(scheme, values.rank());
            (Self::quantize_dynamic(values, &scheme), indices)
        };
        (quantized, TchTensor::new(indices.tensor.to_kind(dtype.into_kind())))
    }
}

impl<E: TchElement> QTensorOps<Self> for LibTorch<E> {
    fn q_from_data(data: TensorData, device: &LibTorchDevice) -> TchQTensor {
        let DType::QFloat(original) = data.dtype else { panic!("expected quantized TensorData") };
        let scheme = checked_scheme(&original);
        let shape = data.shape.clone();
        let quantized = QuantizedBytes { num_elements: data.num_elements(), scheme: original, bytes: data.into_bytes() };
        let (values, params) = quantized.into_vec_i8_with_shape(&shape);
        let scales = valid_scales(tch::Tensor::from_slice(&params.scales).to((*device).into()), &shape, &scheme);
        let values = TchTensor::from_data::<i8>(TensorData::new(values, shape), (*device).into());
        TchQTensor { values, scales, scheme }
    }
    fn quantize(tensor: TchTensor, scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>) -> TchQTensor
    {
        let scheme = checked_scheme(scheme);
        assert_eq!(tensor.tensor.device(), qparams.scales.tensor.device(), "scales on another device");
        let shape = tensor.shape();
        let scales = valid_scales(qparams.scales.tensor, &shape, &scheme);
        let scale = expanded_scales(&scales.tensor, &shape, scheme.level);
        let x = tensor.tensor.to_kind(tch::Kind::Float)/scale;
        let (lo, hi) = scheme.value.range();
        // Match Host's half-away-from-zero rounding, not torch.round's ties-even.
        let rounded = x.sign()*(x.abs()+0.5).floor();
        let values = TchTensor::new(rounded.clamp(lo as f64, hi as f64).to_kind(tch::Kind::Int8));
        TchQTensor { values, scales, scheme }
    }
    fn quantize_dynamic(tensor: TchTensor, requested: &QuantScheme) -> TchQTensor {
        let scheme = checked_scheme(requested);
        let shape = tensor.shape();
        assert!(shape.num_elements() > 0, "dynamic quantization requires nonempty tensors");
        let x = tensor.tensor.to_kind(tch::Kind::Float).abs();
        let alpha = match scheme.level {
            QuantLevel::Tensor => x.max().reshape([1]),
            QuantLevel::Block(block) => {
                let block = block.to_dim_vec(shape.rank());
                let params = params_shape(&shape, scheme.level);
                let mut padding = Vec::new();
                let mut interleaved = Vec::new();
                let mut reductions = Vec::new();
                for axis in (0..shape.rank()).rev() {
                    padding.extend([0, (params[axis]*block[axis] as usize-shape[axis]) as i64]);
                }
                for axis in 0..shape.rank() {
                    interleaved.extend([params[axis] as i64, block[axis] as i64]);
                    reductions.push((2*axis+1) as i64);
                }
                x.constant_pad_nd(padding).reshape(interleaved).amax(&reductions[..], false)
            }
        };
        let (lo, hi) = scheme.value.range();
        let scales = TchTensor::new(alpha*(2.0/(hi-lo) as f64));
        Self::quantize(tensor, &scheme, QuantizationParametersPrimitive { scales })
    }
    fn dequantize(tensor: TchQTensor, dtype: FloatDType) -> TchTensor {
        let scales = expanded_scales(&tensor.scales.tensor, &tensor.shape(), tensor.scheme.level);
        TchTensor::new((tensor.values.tensor.to_kind(tch::Kind::Float)*scales).to_kind(dtype.into_kind()))
    }
    fn q_device(tensor: &TchQTensor) -> LibTorchDevice { tensor.values.tensor.device().into() }
    fn q_to_device(tensor: TchQTensor, device: &LibTorchDevice) -> TchQTensor {
        TchQTensor { values: TchOps::to_device(tensor.values, device),
            scales: TchOps::to_device(tensor.scales, device), scheme: tensor.scheme }
    }
    async fn q_into_data(tensor: TchQTensor) -> Result<TensorData, ExecutionError> {
        let shape = tensor.shape();
        // The only host transfer here is the explicit TensorData export boundary.
        let values = Vec::<i8>::try_from(&tensor.values.tensor.to(tch::Device::Cpu).reshape([-1]))
            .expect("could not export quantized values");
        let scales = Vec::<f32>::try_from(&tensor.scales.tensor.to(tch::Device::Cpu).reshape([-1]))
            .expect("could not export quantization scales");
        Ok(export_data(values, shape, tensor.scheme, &scales))
    }
    fn q_reshape(tensor: TchQTensor, shape: Shape) -> TchQTensor {
        Self::q_map(tensor, |x| TchOps::reshape(x, shape))
    }
    fn q_swap_dims(tensor: TchQTensor, a: usize, b: usize) -> TchQTensor {
        Self::q_map(tensor, |x| TchOps::swap_dims(x, a, b))
    }
    fn q_permute(tensor: TchQTensor, axes: &[usize]) -> TchQTensor {
        Self::q_map(tensor, |x| TchOps::permute(x, axes))
    }
    fn q_flip(tensor: TchQTensor, axes: &[usize]) -> TchQTensor {
        Self::q_map(tensor, |x| TchOps::flip(x, axes))
    }
    fn q_expand(tensor: TchQTensor, shape: Shape) -> TchQTensor {
        Self::q_map(tensor, |x| TchOps::expand(x, shape))
    }
    fn q_select(tensor: TchQTensor, dim: usize, indices: TchTensor) -> TchQTensor {
        Self::q_map(tensor, |x| TchTensor::new(x.tensor.index_select(dim as i64,
            &indices.tensor.to_kind(tch::Kind::Int64))))
    }
    fn q_slice(tensor: TchQTensor, slices: &[ruda_tensor::Slice]) -> TchQTensor {
        Self::q_map(tensor, |x| TchOps::slice_with_steps(x, slices))
    }
    fn q_gather(dim: usize, tensor: TchQTensor, indices: TchTensor) -> TchQTensor {
        Self::q_map(tensor, |x| TchOps::gather(dim, x, indices))
    }
    fn q_argmax(tensor: TchQTensor, dim: usize, dtype: IntDType) -> TchTensor {
        let x = Self::dequantize(tensor, FloatDType::F32);
        TchTensor::new(x.tensor.argmax(dim as i64, true).to_kind(dtype.into_kind()))
    }
    fn q_argmin(tensor: TchQTensor, dim: usize, dtype: IntDType) -> TchTensor {
        let x = Self::dequantize(tensor, FloatDType::F32);
        TchTensor::new(x.tensor.argmin(dim as i64, true).to_kind(dtype.into_kind()))
    }
    fn q_max_dim(tensor: TchQTensor, dim: usize) -> TchQTensor {
        Self::q_map(tensor, |x| TchOps::max_dim(x, dim))
    }
    fn q_min_dim(tensor: TchQTensor, dim: usize) -> TchQTensor {
        Self::q_map(tensor, |x| TchOps::min_dim(x, dim))
    }
    fn q_max_dim_with_indices(tensor: TchQTensor, dim: usize, dtype: IntDType) -> (TchQTensor, TchTensor) {
        Self::q_map_indices(tensor, |x| TchOps::max_dim_with_indices(x, dim), dtype)
    }
    fn q_min_dim_with_indices(tensor: TchQTensor, dim: usize, dtype: IntDType) -> (TchQTensor, TchTensor) {
        Self::q_map_indices(tensor, |x| TchOps::min_dim_with_indices(x, dim), dtype)
    }
    fn q_sort(tensor: TchQTensor, dim: usize, descending: bool) -> TchQTensor {
        Self::q_map(tensor, |x| TchOps::sort(x, dim, descending))
    }
    fn q_sort_with_indices(tensor: TchQTensor, dim: usize, descending: bool, dtype: IntDType)
        -> (TchQTensor, TchTensor)
    {
        Self::q_map_indices(tensor, |x| TchOps::sort_with_indices(x, dim, descending), dtype)
    }
    fn q_argsort(tensor: TchQTensor, dim: usize, descending: bool, dtype: IntDType) -> TchTensor {
        let x = Self::dequantize(tensor, FloatDType::F32);
        TchTensor::new(x.tensor.argsort(dim as i64, descending).to_kind(dtype.into_kind()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruda_tensor::{ops::FloatTensorOps, read_sync};
    type B = LibTorch<f32>;

    fn floats(values: Vec<f32>, shape: impl Into<Shape>) -> TchTensor {
        B::float_from_data(TensorData::new(values, shape.into()), &LibTorchDevice::Cpu)
    }
    fn data(tensor: TchQTensor) -> TensorData { read_sync(B::q_into_data(tensor)).unwrap() }

    #[test]
    fn packed_subbyte_roundtrip_matches_logical_shape_and_wire_size() {
        for value in [QuantValue::Q4S, QuantValue::Q4F, QuantValue::Q2S, QuantValue::Q2F] {
            let scheme = checked_scheme(&QuantScheme::default().with_value(value));
            let (lo, hi) = value.range();
            let values = (0..19).map(|i| (i as f32 % (hi-lo+1.0) + lo) as i8).collect::<Vec<_>>();
            let encoded = export_data(values.clone(), [19].into(), scheme, &[0.5]);
            assert_eq!(encoded.bytes.len(), (19usize.div_ceil(32/value.size_bits())+1)*4);
            let restored = B::q_from_data(encoded.clone(), &LibTorchDevice::Cpu);
            assert_eq!(data(restored), encoded);
            let (decoded, params) = QuantizedBytes { bytes: encoded.bytes, scheme, num_elements: 19 }
                .into_vec_i8_with_shape(&Shape::new([19]));
            assert_eq!(decoded, values);
            assert_eq!(params.scales, vec![0.5]);
        }
    }

    #[test]
    fn native_odd_length_and_clipped_half_away_rounding() {
        let scheme = QuantScheme::default().with_value(QuantValue::Q8S).with_store(QuantStore::Native);
        let x = floats(vec![-100.0,-0.25,0.0,0.25,100.0], [5]);
        let q = B::quantize(x, &scheme, QuantizationParametersPrimitive { scales: floats(vec![0.5], [1]) });
        let encoded = data(q);
        let q = B::q_from_data(encoded.clone(), &LibTorchDevice::Cpu);
        let actual = read_sync(B::float_into_data(B::dequantize(q, FloatDType::F32))).unwrap();
        assert_eq!(actual.to_vec::<f32>().unwrap(), vec![-63.5,-0.5,0.0,0.5,63.5]);
        assert_eq!(encoded.shape, Shape::new([5]));
    }

    #[test]
    fn ragged_block_scales_and_permutation_are_finite() {
        let scheme = QuantScheme::default().with_value(QuantValue::Q4S)
            .with_level(QuantLevel::block([2,3]));
        let q = B::quantize_dynamic(floats((0..15).map(|i| i as f32 - 7.0).collect(), [3,5]), &scheme);
        assert_eq!(q.scales.shape(), Shape::new([2,2]));
        let q = B::q_from_data(data(q), &LibTorchDevice::Cpu);
        let out = B::q_permute(q, &[1,0]);
        assert_eq!(out.shape(), Shape::new([5,3]));
        let out = B::dequantize(out, FloatDType::F32);
        assert!(out.tensor.isfinite().all().int64_value(&[]) != 0);
    }

    #[test]
    fn sort_and_extrema_return_requested_index_dtype() {
        let scheme = QuantScheme::default().with_value(QuantValue::Q8S);
        let q = B::quantize_dynamic(floats(vec![3.0,-1.0,2.0], [3]), &scheme);
        let (_, indices) = B::q_sort_with_indices(q.clone(), 0, false, IntDType::I32);
        assert_eq!(indices.tensor.kind(), tch::Kind::Int);
        let indices = Vec::<i32>::try_from(&indices.tensor).unwrap();
        assert_eq!(indices, vec![1,2,0]);
        let (_, indices) = B::q_max_dim_with_indices(q, 0, IntDType::I16);
        assert_eq!(indices.tensor.kind(), tch::Kind::Int16);
    }

    #[test]
    #[should_panic(expected="finite and nonnegative")]
    fn negative_scales_are_rejected() {
        let scheme = QuantScheme::default();
        B::quantize(floats(vec![1.0], [1]), &scheme,
            QuantizationParametersPrimitive { scales: floats(vec![-1.0], [1]) });
    }
}
