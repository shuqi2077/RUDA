use super::*;
use ruda_tensor::api::activation::sigmoid;

pub(super) struct Attention<B: Backend> {
    pub q: Linear<B>,
    pub k: Linear<B>,
    pub v: Linear<B>,
    pub out: Linear<B>,
    pub q_norm: Norm<B>,
    pub k_norm: Norm<B>,
}

pub(super) fn mask_floor(dtype: DType) -> f32 {
    match dtype {
        DType::F32 => f32::MIN,
        DType::F16 => half::f16::MIN.to_f32(),
        DType::BF16 => half::bf16::MIN.to_f32(),
        _ => panic!("unsupported Qwen3.5 attention dtype"),
    }
}

pub(super) fn text_rope<B: Backend>(
    config: &Qwen35TextConfig,
    start: usize,
    sequence: usize,
    dtype: DType,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let dim = (config.head_dim as f64 * config.rope_parameters.partial_rotary_factor) as usize;
    let theta = config.rope_parameters.rope_theta as f32;
    let inverse = (0..dim / 2)
        .map(|i| theta.powf(-((2 * i) as f32) / dim as f32))
        .collect::<Vec<_>>();
    let inv =
        Tensor::<B, 2>::from_data(TensorData::new(inverse, [1, dim / 2]), (device, DType::F32));
    let pos = Tensor::<B, 2>::from_data(
        TensorData::new(
            (start..start + sequence)
                .map(|n| n as f32)
                .collect::<Vec<_>>(),
            [sequence, 1],
        ),
        (device, DType::F32),
    );
    let freq = pos * inv;
    let freq = Tensor::cat(vec![freq.clone(), freq], 1).reshape([1, 1, sequence, dim]);
    (freq.clone().cos().cast(dtype), freq.sin().cast(dtype))
}

pub(super) fn multimodal_rope<B: Backend>(
    config: &Qwen35TextConfig,
    coordinates: &[[usize; 3]],
    batch: usize,
    sequence: usize,
    dtype: DType,
    device: &B::Device,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>), GenerationError> {
    let dim = (config.head_dim as f64 * config.rope_parameters.partial_rotary_factor) as usize;
    let sections = config.rope_parameters.mrope_section;
    if !config.rope_parameters.mrope_interleaved
        || sections.iter().try_fold(0usize, |n, &s| n.checked_add(s)) != Some(dim / 2)
        || batch.checked_mul(sequence) != Some(coordinates.len())
        || coordinates.iter().flatten().any(|&p| p >= config.max_position_embeddings)
    {
        return Err(GenerationError("invalid Qwen3.5 interleaved MRoPE positions or sections".into()));
    }
    let theta = config.rope_parameters.rope_theta as f32;
    let inverse = (0..dim / 2)
        .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
        .collect::<Vec<_>>();
    let mut frequencies = Vec::with_capacity(coordinates.len() * dim);
    for position in coordinates {
        let row = (0..dim / 2).map(|i| {
            let axis = if i % 3 == 1 && i / 3 < sections[1] { 1 }
                else if i % 3 == 2 && i / 3 < sections[2] { 2 } else { 0 };
            position[axis] as f32 * inverse[i]
        }).collect::<Vec<_>>();
        frequencies.extend_from_slice(&row);
        frequencies.extend_from_slice(&row);
    }
    let frequencies = Tensor::<B, 4>::from_data(
        TensorData::new(frequencies, [batch, 1, sequence, dim]), (device, DType::F32),
    );
    Ok((frequencies.clone().cos().cast(dtype), frequencies.sin().cast(dtype)))
}

pub(super) fn rotate<B: Backend>(input: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
    let [b, h, s, d] = input.dims();
    let r = cos.dims()[3];
    let rot = input.clone().slice([0..b, 0..h, 0..s, 0..r]);
    let first = rot.clone().slice([0..b, 0..h, 0..s, 0..r / 2]);
    let second = rot.clone().slice([0..b, 0..h, 0..s, r / 2..r]);
    let rotated = rot * cos + Tensor::cat(vec![-second, first], 3) * sin;
    if r == d {
        rotated
    } else {
        Tensor::cat(vec![rotated, input.slice([0..b, 0..h, 0..s, r..d])], 3)
    }
}

pub(super) fn repeat_heads<B: Backend>(x: Tensor<B, 4>, repeats: usize) -> Tensor<B, 4> {
    if repeats == 1 {
        return x;
    }
    let [b, h, s, d] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .repeat_dim(2, repeats)
        .reshape([b, h * repeats, s, d])
}

impl<R, F, I, BT> Attention<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    pub(super) fn probabilities(
        scores: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 4>, GenerationError> {
        let dtype = scores.dtype();
        let probs = rudnn::normalization::softmax_last_axis(scores.cast(DType::F32).into_primitive().tensor())
            .map_err(|error| GenerationError(error.to_string()))?;
        Ok(Tensor::<DeviceBackend<R,F,I,BT>,4>::from_primitive(TensorPrimitive::Float(probs)).cast(dtype))
    }

    pub fn forward(
        &self,
        x: Tensor<DeviceBackend<R, F, I, BT>, 3>,
        c: &Qwen35TextConfig,
        cos: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        sin: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        key_cache: &mut Option<Tensor<DeviceBackend<R, F, I, BT>, 4>>,
        value_cache: &mut Option<Tensor<DeviceBackend<R, F, I, BT>, 4>>,
        position: usize,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let [b, s, _] = x.dims();
        let (h, k, d) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let qgate = self.q.forward(x.clone()).reshape([b, s, h, 2 * d]);
        let gate = qgate
            .clone()
            .slice([0..b, 0..s, 0..h, d..2 * d])
            .reshape([b, s, h * d]);
        let q = self
            .q_norm
            .forward(qgate.slice([0..b, 0..s, 0..h, 0..d]))?
            .swap_dims(1, 2);
        let key = self
            .k_norm
            .forward(self.k.forward(x.clone()).reshape([b, s, k, d]))?
            .swap_dims(1, 2);
        let value = self.v.forward(x).reshape([b, s, k, d]).swap_dims(1, 2);
        let q = rotate(q, cos.clone(), sin.clone());
        let key = rotate(key, cos, sin);
        let key = match key_cache.as_ref() {
            Some(old) => Tensor::cat(vec![old.clone(), key], 2),
            None => key,
        };
        let value = match value_cache.as_ref() {
            Some(old) => Tensor::cat(vec![old.clone(), value], 2),
            None => value,
        };
        *key_cache = Some(key.clone());
        *value_cache = Some(value.clone());
        let length = position + s;
        let floor = mask_floor(q.dtype());
        let mask = (0..s)
            .flat_map(|i| {
                (0..length).map(move |j| {
                    if j <= position + i {
                        0.0f32
                    } else {
                    floor
                    }
                })
            })
            .collect::<Vec<_>>();
        let mask = Tensor::<DeviceBackend<R, F, I, BT>, 4>::from_data(
            TensorData::new(mask, [1, 1, s, length]),
            (&q.device(), q.dtype()),
        );
        let scores =
            q.matmul(repeat_heads(key, h / k).swap_dims(2, 3)) * (d as f64).sqrt().recip() + mask;
        let probs = Self::probabilities(scores)?;
        let out = Self::value_product(probs, repeat_heads(value, h / k))?
            .swap_dims(1, 2)
            .reshape([b, s, h * d]);
        Ok(self.out.forward(out * sigmoid(gate)))
    }

    pub(super) fn value_product(
        probabilities: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        values: Tensor<DeviceBackend<R, F, I, BT>, 4>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 4>, GenerationError> {
        let dtype = probabilities.dtype();
        if !matches!(dtype, DType::BF16 | DType::F16) {
            return Ok(probabilities.matmul(values));
        }
        let output = rublas::tensor_matmul::matmul(
            probabilities.clone().into_primitive().tensor(), values.clone().into_primitive().tensor(), None,
            rublas::tensor_matmul::MatmulStrategy::CmmaResidueFirst, dtype,
        );
        match output {
            Ok(output) => Ok(Tensor::from_primitive(TensorPrimitive::Float(output))),
            Err(rublas::kernel_ir::definition::MatmulSetupError::Unavailable(_)) => {
                Ok(probabilities.matmul(values))
            }
            Err(error) => Err(GenerationError(error.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn causal_mask_uses_dtype_finite_minimum() {
        assert_eq!(mask_floor(DType::F32), -f32::MAX);
        assert_eq!(mask_floor(DType::F16), -65504.0);
        assert_eq!(mask_floor(DType::BF16).to_bits(), 0xff7f0000);
    }
}
