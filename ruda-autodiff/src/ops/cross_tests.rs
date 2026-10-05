use crate::Autodiff;
use alloc::vec;
use ruda_tensor::{TensorData, api::Tensor};

type B = Autodiff<ruda_tensor_host::Host>;

#[test]
fn cross_reduces_broadcast_gradients_for_each_parent_and_axis() {
    let device = Default::default();
    for axis in [0, 1] {
        for reverse in [false, true] {
            for (track_lhs, track_rhs) in [(true, true), (true, false), (false, true)] {
                let small = Tensor::<B, 2>::from_data([[1., 2., 3.]], &device);
                let large = Tensor::<B, 2>::from_data([[4., 5., 6.], [7., 8., 9.]], &device);
                let weights = Tensor::<B, 2>::from_data([[1., 2., 4.], [3., 5., 7.]], &device);
                let (lhs, rhs) = if reverse {
                    (large, small)
                } else {
                    (small, large)
                };
                let orient = |value: Tensor<B, 2>| {
                    if axis == 0 {
                        value.swap_dims(0, 1).detach()
                    } else {
                        value
                    }
                };
                let lhs = orient(lhs).set_require_grad(track_lhs);
                let rhs = orient(rhs).set_require_grad(track_rhs);
                let output = lhs.clone().cross(rhs.clone(), axis);
                let sign: f32 = if reverse { -1. } else { 1. };
                let mut expected = TensorData::new(
                    vec![
                        -3. * sign,
                        6. * sign,
                        -3. * sign,
                        -6. * sign,
                        12. * sign,
                        -6. * sign,
                    ],
                    [2, 3],
                );
                if axis == 0 {
                    expected = Tensor::<B, 2>::from_data(expected, &device)
                        .swap_dims(0, 1)
                        .into_data();
                }
                output.to_data().assert_eq(&expected, true);
                let gradients = (output * orient(weights)).sum().backward();
                let small_grad = vec![19., -32., 14.];
                let large_grad = vec![-2., 1., 0., 1., -2., 1.];
                let (left, right) = if reverse {
                    (
                        TensorData::new(large_grad, [2, 3]),
                        TensorData::new(small_grad, [1, 3]),
                    )
                } else {
                    (
                        TensorData::new(small_grad, [1, 3]),
                        TensorData::new(large_grad, [2, 3]),
                    )
                };
                for (input, tracked, expected) in [(lhs, track_lhs, left), (rhs, track_rhs, right)]
                {
                    let actual = input.grad(&gradients);
                    if tracked {
                        let actual = actual.expect("missing cross gradient");
                        assert_eq!(actual.dims(), input.dims());
                        let expected = orient(Tensor::<B, 2>::from_data(expected, &device)) * sign;
                        actual.to_data().assert_eq(&expected.into_data(), true);
                    } else {
                        assert!(actual.is_none());
                    }
                }
            }
        }
    }
}
