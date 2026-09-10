use super::*;

#[test]
fn should_assert_appox_eq_limit() {
    let data1 = TensorData::from([[3.0, 5.0, 6.0]]);
    let data2 = TensorData::from([[3.03, 5.0, 6.0]]);

    data1.assert_approx_eq::<f32>(&data2, Tolerance::absolute(3e-2));
    data1.assert_approx_eq::<f16>(&data2, Tolerance::absolute(3e-2));
}

#[test]
#[should_panic]
fn should_assert_approx_eq_above_limit() {
    let data1 = TensorData::from([[3.0, 5.0, 6.0]]);
    let data2 = TensorData::from([[3.031, 5.0, 6.0]]);

    data1.assert_approx_eq::<f32>(&data2, Tolerance::absolute(1e-2));
}

#[test]
#[should_panic]
fn should_assert_approx_eq_check_shape() {
    let data1 = TensorData::from([[3.0, 5.0, 6.0, 7.0]]);
    let data2 = TensorData::from([[3.0, 5.0, 6.0]]);

    data1.assert_approx_eq::<f32>(&data2, Tolerance::absolute(1e-2));
}
