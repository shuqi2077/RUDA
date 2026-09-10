use super::*;

pub(super) fn qwen2_rope<B: Backend>(
    config: &LlamaConfig,
    device: &B::Device,
) -> RotaryEncoding<B> {
    let dtype = Tensor::<B, 1>::zeros([1], device).dtype();
    qwen2_rope_with_dtype(config, device, dtype)
}

pub(crate) fn qwen2_rope_with_dtype<B: Backend>(
    config: &LlamaConfig,
    device: &B::Device,
    dtype: DType,
) -> RotaryEncoding<B> {
    let half = config.head_dimension() / 2;
    let inverse_frequencies = (0..half)
        .map(|i| 1.0f32 / (config.rope_theta as f32).powf(i as f32 / half as f32))
        .collect::<Vec<_>>();
    let theta = Tensor::<B, 1>::from_data(
        TensorData::new(inverse_frequencies, [half]),
        (device, DType::F32),
    );
    let mut rope = RotaryEncodingConfig::new(config.max_sequence_length, config.head_dimension())
        .with_theta(config.rope_theta)
        .init(device);
    rope.theta = theta.clone();
    let positions = Tensor::<B, 2>::from_data(
        TensorData::new(
            (0..config.max_sequence_length)
                .map(|i| i as f32)
                .collect::<Vec<_>>(),
            [config.max_sequence_length, 1],
        ),
        (device, DType::F32),
    );
    let frequencies = positions * theta.unsqueeze::<2>();
    rope.freq_complex = Tensor::cat(vec![frequencies.clone().cos(), frequencies.sin()], 1)
        .reshape([config.max_sequence_length, 2, half])
        .transpose()
        .unsqueeze_dim::<4>(2)
        .repeat_dim(2, 2)
        .reshape([config.max_sequence_length, config.head_dimension(), 2])
        .cast(dtype);
    rope
}
