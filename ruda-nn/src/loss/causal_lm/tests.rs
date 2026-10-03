use super::*;
use crate::{Linear, LoRALinearConfig, TestAutodiffBackend as B};
use ruda_model::{
    module::Param,
    tensor::{TensorData, Tolerance},
};

#[test]
fn chunked_causal_loss_and_gradients_match_full_vocabulary_reference() {
    let device = Default::default();
    let head = Linear::<B> {
        weight: Param::from_tensor(Tensor::from_floats(
            [[1.0, 0.0, -1.0], [0.5, -0.5, 0.0]],
            &device,
        )),
        bias: None,
    };
    let hidden = Tensor::<B, 3>::from_floats(
        [
            [[1.0, 2.0], [-1.0, 0.5], [2.0, -1.0]],
            [[0.0, 1.0], [0.5, -0.5], [-1.0, -2.0]],
        ],
        &device,
    )
    .require_grad();
    let labels = Tensor::<B, 2, Int>::from_data([[-100, 2, -100], [0, 1, 0]], &device);
    let full_hidden = hidden.clone().slice([0..2, 0..2, 0..2]).reshape([4, 2]);
    let logp = log_softmax(head.forward(full_hidden), 1);
    let expected = -(logp.clone().slice([0..1, 2..3]).sum()
        + logp.clone().slice([2..3, 1..2]).sum()
        + logp.slice([3..4, 0..1]).sum())
        / 3;
    let expected_grads = expected.clone().backward();
    let data = expected.to_data();
    for chunk in [1, 2, 3, 8] {
        let output = CausalCrossEntropyConfig::new()
            .with_token_chunk_size(chunk)
            .forward_hidden(hidden.clone(), labels.clone(), |rows| head.forward(rows));
        output
            .valid_tokens
            .to_data()
            .assert_eq(&TensorData::from([3_i64]), false);
        let loss = output.mean();
        loss.to_data()
            .assert_approx_eq::<f32>(&data, Tolerance::absolute(1e-6));
        let grads = loss.backward();
        hidden
            .grad(&grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &hidden.grad(&expected_grads).unwrap().to_data(),
                Tolerance::absolute(1e-6),
            );
        head.weight
            .val()
            .grad(&grads)
            .unwrap()
            .to_data()
            .assert_approx_eq::<f32>(
                &head.weight.val().grad(&expected_grads).unwrap().to_data(),
                Tolerance::absolute(1e-6),
            );
    }
}

#[test]
fn ignored_short_and_unshifted_sequences_have_defined_losses() {
    let device = Default::default();
    for sequence in [0, 1, 3] {
        let hidden = Tensor::<B, 3>::ones([1, sequence, 2], &device).require_grad();
        let labels = Tensor::<B, 2, Int>::full([1, sequence], -100, &device);
        let result = CausalCrossEntropyConfig::new().forward_hidden(hidden, labels, |rows| rows);
        assert_eq!(result.mean().into_scalar(), 0.0);
        assert_eq!(result.valid_tokens.into_scalar(), 0);
    }
    let result = CausalCrossEntropyConfig::new()
        .with_shift(false)
        .with_ignore_index(-7)
        .forward_hidden(
            Tensor::<B, 3>::zeros([1, 2, 3], &device),
            Tensor::<B, 2, Int>::from_data([[1, -7]], &device),
            |rows| rows,
        );
    assert!((result.loss_sum.into_scalar() - 3.0_f32.ln()).abs() < 1e-6);
    assert_eq!(result.valid_tokens.into_scalar(), 1);
}

#[test]
fn chunked_projection_trains_lora_without_base_gradients() {
    let device = Default::default();
    let base = Linear::<B> {
        weight: Param::from_tensor(Tensor::from_floats(
            [[1.0, 0.0, -1.0], [0.5, -0.5, 0.0]],
            &device,
        )),
        bias: None,
    };
    let head = LoRALinearConfig::new(1, 1.0).init(base);
    let hidden = Tensor::<B, 3>::ones([1, 3, 2], &device).require_grad();
    let result = CausalCrossEntropyConfig::new()
        .with_token_chunk_size(1)
        .forward_hidden(
            hidden.clone(),
            Tensor::<B, 2, Int>::from_data([[0, 1, 2]], &device),
            |rows| head.forward(rows),
        );
    let gradients = result.mean().backward();
    assert!(hidden.grad(&gradients).is_some());
    assert!(head.adapter_b.weight.val().grad(&gradients).is_some());
    assert!(head.base.weight.val().grad(&gradients).is_none());
}
