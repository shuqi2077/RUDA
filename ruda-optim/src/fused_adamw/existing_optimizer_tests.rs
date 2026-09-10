// SPDX-License-Identifier: Apache-2.0
use super::{AdamWOptions, StepControl, reference::adamw_reference};
use crate::{AdamWConfig, SimpleOptimizer};
use ruda_model::tensor::{Tensor, TensorData};
use ruda_tensor_host::Host;

#[test]
fn reference_agrees_with_existing_adamw_for_supported_modes() {
    for amsgrad in [false, true] {
        let options = AdamWOptions { amsgrad, ..Default::default() };
        let optimizer = AdamWConfig::new().with_beta_1(options.beta1).with_beta_2(options.beta2)
            .with_epsilon(options.epsilon).with_weight_decay(options.weight_decay).with_amsgrad(amsgrad).build();
        let device = Default::default();
        let mut original = Tensor::<Host, 1>::from_data([0.5f32, -0.3, 1.0], &device);
        let mut original_state = None;
        let mut reference = vec![0.5f32, -0.3, 1.0];
        let mut reference_state = None;
        for step in 0..20 {
            let gradient = vec![0.2 + step as f32 * 0.01, -0.05, 0.3];
            let actual = optimizer.step(options.learning_rate as f64, original,
                Tensor::<Host, 1>::from_data(TensorData::new(gradient.clone(), [3]), &device), original_state);
            let expected = adamw_reference(&reference, &gradient, reference_state.as_ref(), &options, StepControl::default()).unwrap();
            let values = actual.0.clone().into_data().to_vec::<f32>().unwrap();
            for (&a, &b) in values.iter().zip(&expected.parameters) {
                assert!((a - b).abs() < 1e-5, "{a} != {b}");
            }
            original = actual.0; original_state = actual.1;
            reference = expected.parameters; reference_state = expected.state;
        }
    }
}
