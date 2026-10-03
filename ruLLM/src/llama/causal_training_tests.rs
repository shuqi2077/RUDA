use super::*;
use ruda_nn::loss::{CausalCrossEntropyConfig, CausalLanguageModel};
use ruda_tensor::{TensorData, Tolerance};
use ruda_tensor_host::Host;

#[test]
fn native_decoder_hidden_contract_preserves_logits_and_exact_causal_loss() {
    let config = LlamaConfig {
        vocab_size: 7,
        d_model: 4,
        d_ff: 8,
        num_hidden_layers: 1,
        num_query_heads: 2,
        num_kv_heads: 1,
        max_sequence_length: 8,
        rms_norm_epsilon: 1e-6,
        rope_theta: 10000.0,
    };
    let device = Default::default();
    let model = LlamaForCausalLm::<Host>::init(&config, &device).unwrap();
    let tokens = Tensor::<Host, 2, Int>::from_data([[1, 2, 3, 4]], &device);
    let hidden = CausalLanguageModel::forward_hidden(&model, tokens.clone());
    let projected = model.project(hidden.reshape([4, 4])).reshape([1, 4, 7]);
    projected.to_data().assert_approx_eq::<f32>(
        &model.forward(tokens.clone()).to_data(),
        Tolerance::absolute(1e-6),
    );
    let labels = Tensor::<Host, 2, Int>::from_data([[-100, 2, -100, 4]], &device);
    let logp = ruda_tensor::api::activation::log_softmax(projected, 2);
    let expected =
        -(logp.clone().slice([0..1, 0..1, 2..3]).sum() + logp.slice([0..1, 2..3, 4..5]).sum()) / 2;
    for chunk in [1, 2, 8] {
        let result = CausalCrossEntropyConfig::new()
            .with_token_chunk_size(chunk)
            .forward_model(&model, tokens.clone(), labels.clone());
        result
            .valid_tokens
            .to_data()
            .assert_eq(&TensorData::from([2_i64]), false);
        result
            .mean()
            .to_data()
            .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(1e-6));
    }
    let packed = model.into_packed_training();
    let result = CausalCrossEntropyConfig::new()
        .with_token_chunk_size(2)
        .forward_model(&packed, tokens, labels);
    result
        .mean()
        .to_data()
        .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(2e-6));
}
