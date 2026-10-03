use super::*;
use crate::TestAutodiffBackend as B;
use ruda_model::{
    module::Param,
    tensor::{TensorData, Tolerance},
};

fn base() -> Linear<B> {
    Linear {
        weight: Param::from_tensor(Tensor::from_floats(
            [[1.0, 2.0], [3.0, 4.0]],
            &Default::default(),
        )),
        bias: Some(Param::from_tensor(Tensor::from_floats(
            [0.5, -0.5],
            &Default::default(),
        ))),
    }
}

#[test]
fn lora_initial_output_and_frozen_base_gradients() {
    let original = base();
    let base_id = original.weight.id;
    let adapter = LoRALinearConfig::new(1, 2.0).init(original.clone());
    let input = Tensor::<B, 2>::from_floats([[1.0, -1.0]], &Default::default()).require_grad();
    let output = adapter.forward(input.clone());
    output
        .to_data()
        .assert_eq(&original.forward(input.clone()).to_data(), false);
    assert_eq!(adapter.base.weight.id, base_id);
    let grads = output.sum().backward();
    assert!(adapter.base.weight.val().grad(&grads).is_none());
    assert!(
        adapter
            .base
            .bias
            .as_ref()
            .unwrap()
            .val()
            .grad(&grads)
            .is_none()
    );
    assert!(adapter.adapter_a.weight.val().grad(&grads).is_some());
    assert!(adapter.adapter_b.weight.val().grad(&grads).is_some());
    input
        .grad(&grads)
        .unwrap()
        .to_data()
        .assert_eq(&TensorData::from([[3.0, 7.0]]), false);
}

#[test]
fn lora_forward_backward_and_merge_match_dense_equations() {
    let mut adapter = LoRALinearConfig::new(1, 2.0).init(base());
    adapter.adapter_a.weight = adapter
        .adapter_a
        .weight
        .map(|_| Tensor::<B, 2>::from_floats([[1.0], [-2.0]], &Default::default()).require_grad());
    adapter.adapter_b.weight = adapter
        .adapter_b
        .weight
        .map(|_| Tensor::<B, 2>::from_floats([[0.25, -0.5]], &Default::default()).require_grad());
    let input = Tensor::<B, 3>::from_floats([[[2.0, 1.0], [-1.0, 3.0]]], &Default::default())
        .require_grad();
    let actual = adapter.forward(input.clone());
    actual
        .to_data()
        .assert_eq(&TensorData::from([[[5.5, 7.5], [5.0, 16.5]]]), false);
    let grads = actual.sum().backward();
    input
        .grad(&grads)
        .unwrap()
        .to_data()
        .assert_eq(&TensorData::from([[[2.5, 8.0], [2.5, 8.0]]]), false);
    adapter
        .adapter_a
        .weight
        .val()
        .grad(&grads)
        .unwrap()
        .to_data()
        .assert_eq(&TensorData::from([[-0.5], [-2.0]]), false);
    adapter
        .adapter_b
        .weight
        .val()
        .grad(&grads)
        .unwrap()
        .to_data()
        .assert_eq(&TensorData::from([[-14.0, -14.0]]), false);
    let merged = adapter.merge();
    merged
        .forward(input)
        .to_data()
        .assert_approx_eq::<f32>(&actual.to_data(), Tolerance::absolute(1e-6));
    assert!(!merged.weight.val().is_require_grad());
}
