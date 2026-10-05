pub use ruda_tensor::api::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestAutodiffBackend as B, TestBackend};

    #[test]
    fn full_axis_topk_preserves_values_indices_and_all_input_gradients() {
        let device = Default::default();
        let input =
            Tensor::<B, 2>::from_floats([[1., 3., 2.], [-2., 0., -1.]], &device).require_grad();
        let expected = TensorData::from([[3., 2., 1.], [0., -1., -2.]]);
        let indices = TensorData::from([[1i32, 2, 0], [1, 2, 0]]);
        input
            .clone()
            .argtopk(3, 1)
            .to_data()
            .assert_eq(&indices, false);
        let output = input.clone().topk(3, 1);
        output.to_data().assert_eq(&expected, false);
        let gradients = output.sum().backward();
        input
            .grad(&gradients)
            .unwrap()
            .to_data()
            .assert_eq(&TensorData::from([[1., 1., 1.], [1., 1., 1.]]), false);
        let (values, actual_indices) = input.clone().topk_with_indices(3, 1);
        values.to_data().assert_eq(&expected, false);
        actual_indices.to_data().assert_eq(&indices, false);
        assert_eq!(input.clone().topk(0, 1).dims(), [2, 0]);
        assert_eq!(input.argtopk(0, 1).dims(), [2, 0]);

        let input = Tensor::<TestBackend, 2, Int>::from_ints([[1, 3, 2], [-2, 0, -1]], &device);
        input
            .clone()
            .topk(3, 1)
            .to_data()
            .assert_eq(&TensorData::from([[3i32, 2, 1], [0, -1, -2]]), false);
        input
            .clone()
            .argtopk(3, 1)
            .to_data()
            .assert_eq(&indices, false);
        assert_eq!(input.topk(0, 1).dims(), [2, 0]);
        let empty = Tensor::<TestBackend, 2>::zeros([2, 0], &device);
        assert_eq!(empty.clone().topk(0, 1).dims(), [2, 0]);
        assert_eq!(empty.argtopk(0, 1).dims(), [2, 0]);
    }
}
