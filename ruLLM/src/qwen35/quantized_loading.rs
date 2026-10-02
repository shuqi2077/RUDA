//! AWQ GEMM checkpoint compatibility: unpack once on the host into dense model
//! weights, in the embedding's dtype. This is not memory-saving INT4 inference.
use crate::{HuggingFaceLoadError, huggingface::{AwqQuantizationConfig, checkpoint::Checkpoint}};
use ruda_tensor::api::{backend::Backend, DType, Tensor, TensorData};

pub(super) fn configuration(raw: &serde_json::Value)
    -> Result<Option<AwqQuantizationConfig>, HuggingFaceLoadError>
{
    let top = raw.get("quantization_config").filter(|v| !v.is_null());
    let nested = raw.get("text_config").and_then(|v| v.get("quantization_config")).filter(|v| !v.is_null());
    if top.is_some() && nested.is_some() && top != nested {
        return Err(HuggingFaceLoadError("conflicting Qwen3.5 quantization configs".into()));
    }
    top.or(nested).map(|value| {
        let config: AwqQuantizationConfig = serde_json::from_value(value.clone())
            .map_err(|e| HuggingFaceLoadError(format!(
                "Qwen3.5 supports AWQ GEMM 4-bit loading into dense weights: {e}")))?;
        config.validate()?;
        Ok(config)
    }).transpose()
}

fn packed_tensor(checkpoint: &Checkpoint, name: &str, shape: [usize; 2], integer: bool)
    -> Result<TensorData, HuggingFaceLoadError>
{
    let snapshot = checkpoint.tensors.get(name)
        .ok_or_else(|| HuggingFaceLoadError(format!("missing AWQ tensor {name}")))?;
    let dtype_ok = if integer { snapshot.dtype == DType::I32 }
        else { matches!(snapshot.dtype, DType::F16 | DType::BF16 | DType::F32) };
    if snapshot.shape != shape.into() || !dtype_ok {
        return Err(HuggingFaceLoadError(format!(
            "{name}: expected {} {shape:?}, found {:?} {:?}",
            if integer { "I32" } else { "floating scales" }, snapshot.dtype, snapshot.shape)));
    }
    snapshot.to_data().map_err(|e| HuggingFaceLoadError(format!("{name}: {e}")))
}

// AWQ GEMM's stored nibble order is [0,2,4,6,1,3,5,7]. This inverse mapping
// matches ruBLAS/src/tensor_int4/kernel.rs. Zeros are direct (not GPTQ's +1).
fn unpack_dense(qweight: &[i32], qzeros: &[i32], scales: &[f32],
    input: usize, output: usize, group_size: usize) -> Vec<f32>
{
    let mut dense = vec![0.0; input * output];
    for i in 0..input {
        for o in 0..output {
            let lane = o % 8;
            let shift = ((lane % 2)*4 + lane/2)*4;
            let q = ((qweight[i*(output/8)+o/8] as u32 >> shift) & 15) as f32;
            let z = ((qzeros[(i/group_size)*(output/8)+o/8] as u32 >> shift) & 15) as f32;
            dense[o*input+i] = (q-z)*scales[(i/group_size)*output+o];
        }
    }
    dense
}

pub(super) fn load_awq_linear<B: Backend>(checkpoint: &mut Checkpoint, prefix: &str,
    input: usize, output: usize, config: &AwqQuantizationConfig, dtype: DType, device: &B::Device)
    -> Result<Tensor<B, 2>, HuggingFaceLoadError>
{
    let layout = config.layout(input, output)?;
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
        return Err(HuggingFaceLoadError("AWQ dense target dtype must be F16, BF16 or F32".into()));
    }
    for suffix in ["weight", "g_idx"] {
        if checkpoint.tensors.contains_key(&format!("{prefix}.{suffix}")) {
            return Err(HuggingFaceLoadError(format!("{prefix}: ambiguous/unsupported AWQ GEMM tensor .{suffix}")));
        }
    }
    let names = [format!("{prefix}.qweight"), format!("{prefix}.qzeros"), format!("{prefix}.scales")];
    let weights = packed_tensor(checkpoint, &names[0], [input, output/8], true)?;
    let zeros = packed_tensor(checkpoint, &names[1], [layout.groups(), output/8], true)?;
    let scales = packed_tensor(checkpoint, &names[2], [layout.groups(), output], false)?;
    let scale_values = scales.iter::<f32>().collect::<Vec<_>>();
    if scale_values.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(HuggingFaceLoadError(format!("{prefix}: invalid AWQ scales")));
    }
    let dense = unpack_dense(&weights.iter::<i32>().collect::<Vec<_>>(),
        &zeros.iter::<i32>().collect::<Vec<_>>(), &scale_values, input, output, layout.group_size);
    let data = TensorData::new(dense, [output, input]).convert_dtype(dtype);
    let tensor = Tensor::from_data(data, (device, dtype));
    for name in names { checkpoint.consumed.insert(name); }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Independent packing in stored-nibble order, not the decoder's formula.
    fn pack(values: &[u32]) -> i32 {
        [0,2,4,6,1,3,5,7].iter().enumerate().fold(0u32, |word, (nibble, &column)|
            word | (values[column] << (4*nibble))) as i32
    }
    fn config() -> serde_json::Value {
        json!({"quant_method":"awq", "bits":4, "group_size":2, "zero_point":true, "version":"gemm"})
    }

    #[test]
    fn awq_signed_words_zero_points_and_multiple_groups() {
        let input = 4;
        let output = 16;
        let q = (0..input).flat_map(|i| (0..output/8).map(move |chunk|
            pack(&(0..8).map(|o| ((i+chunk*8+o)%16) as u32).collect::<Vec<_>>())))
            .collect::<Vec<_>>();
        let z = vec![pack(&[1,2,3,4,5,6,7,8]); 4];
        let scales = (0..32).map(|v| (v+1) as f32/32.0).collect::<Vec<_>>();
        let actual = unpack_dense(&q,&z,&scales,input,output,2);
        assert!(q.iter().any(|word| *word < 0));
        for i in 0..input { for o in 0..output {
            let expected = (((i+o)%16) as f32-((o%8)+1) as f32)*scales[(i/2)*output+o];
            assert_eq!(actual[o*input+i], expected);
        }}
    }

    #[test]
    fn nested_quantization_and_conflict_validation() {
        assert!(configuration(&json!({})).unwrap().is_none());
        let c = config();
        assert!(configuration(&json!({"text_config":{"quantization_config": c.clone()}})).unwrap().is_some());
        assert!(configuration(&json!({"quantization_config":c.clone(),
            "text_config":{"quantization_config":c.clone()}})).is_ok());
        let mut other = c.clone(); other["group_size"] = json!(4);
        assert!(configuration(&json!({"quantization_config":c,
            "text_config":{"quantization_config":other}})).is_err());
    }

    #[test]
    fn unsupported_formats_and_incomplete_groups_are_rejected() {
        let mut config: AwqQuantizationConfig = serde_json::from_value(config()).unwrap();
        assert!(config.layout(4,16).is_ok());
        assert!(config.layout(3,16).is_err());
        assert!(config.layout(4,15).is_err());
        config.version = "gemv".into(); assert!(config.validate().is_err());
        config.version = "gemm".into(); config.quant_method = "gptq".into();
        assert!(config.validate().is_err());
    }
}
