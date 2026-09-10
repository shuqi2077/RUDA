
use super::*;
use crate::GradientsParams;
use crate::TestAutodiffBackend;
use ruda_model::module::{Module, Param};
use ruda_model::tensor::{Tensor, TensorData};
use ruda_nn::{Linear, LinearConfig, LinearRecord};

fn given_linear_layer(weight: TensorData, bias: TensorData) -> Linear<TestAutodiffBackend> {
    let device = Default::default();
    let record = LinearRecord {
        weight: Param::from_data(weight, &device),
        bias: Some(Param::from_data(bias, &device)),
    };

    LinearConfig::new(6, 6).init(&device).load_record(record)
}
#[test]
fn test_cubic_interpolate() {
    let tolerance = 1e-8;

    // basic
    let (x1, f1, g1, x2, f2, g2) = (-1.0, 1.0, -2.0, 1.0, 1.0, 2.0);
    let result = cubic_interpolate(x1, f1, g1, x2, f2, g2, None);
    assert!(
        (result - 0.00000).abs() < tolerance,
        "Basic: Result {} should be close to 0.0",
        result
    );

    // bound
    let (x1, f1, g1, x2, f2, g2) = (0.0, 0.25, -1.0, 1.0, 0.25, 1.0);
    let bounds = Some((0.6, 1.0));
    let result = cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds);
    assert!(
        (result - 0.6000000000).abs() < tolerance,
        "Bound: Result {} should be clamped to 0.6",
        result
    );

    // d2_square < 0,should return mid value
    let (x1, f1, g1, x2, f2, g2) = (0.0, 0.0, 10.0, 1.0, 5.0, 10.0);
    let result = cubic_interpolate(x1, f1, g1, x2, f2, g2, Some((0.0, 1.0)));
    assert!(
        (result - 0.5000000).abs() < tolerance,
        "Fallback: Result {} should be midpoint 0.5",
        result
    );

    // asymmetric
    let (x1, f1, g1, x2, f2, g2) = (0.0, 1.0, -5.0, 1.0, 0.5, 1.0);
    let result = cubic_interpolate(x1, f1, g1, x2, f2, g2, None);
    assert!(
        (result - 0.4606553370833684).abs() < tolerance,
        "Asymmetric: Result {} should be 0.4606553370833684",
        result
    );

    // not good value
    let (x1, f1, g1, x2, f2, g2) = (
        1.231232145,
        -0.12567458754,
        9.1231243007,
        8.239105015,
        -100.9012398021,
        123201321.0293982,
    );
    let result_1 = cubic_interpolate(x1, f1, g1, x2, f2, g2, None);
    let result_2 = cubic_interpolate(x1, f1, g1, x2, f2, g2, Some((-4.4, 4.4)));
    assert!(
        (result_1 - 5.9031480234724434).abs() < tolerance,
        "not good value 1: Result {} should be 5.9031480234724434",
        result
    );
    assert!(
        (result_2 - 4.4000000000000004).abs() < tolerance,
        "not good value 2: Result {} should be 4.4000000000000004",
        result
    );
}
#[test]
fn test_strong_wolfe_direct_comparison() {
    let device = Default::default();
    let tol = 1e-6;

    {
        let x = Tensor::<TestAutodiffBackend, 1>::from_floats([2.1321912957_f64], &device);
        let d = Tensor::<TestAutodiffBackend, 1>::from_floats([0.91312321_f64], &device);
        let t_initial = 1.213132_f64;
        fn func<B: Backend>(
            x_base: &Tensor<B, 1>,
            t_val: f64,
            d_vec: &Tensor<B, 1>,
        ) -> (f64, Tensor<B, 1>) {
            let curr_x = x_base.clone().add(d_vec.clone().mul_scalar(t_val));
            let x2 = curr_x.clone().mul(curr_x.clone());
            let x3 = x2.clone().mul(curr_x.clone());
            let x4 = x2.clone().mul(x2.clone());

            // f(x) = x^4 - 2*x^2 + x
            let f_elements = x4 - x2.mul_scalar(2.0) + curr_x.clone();

            let f_val = f_elements.sum().into_scalar().to_f64();

            // g(x) = 4*x^3 - 4*x + 1
            let g = x3.mul_scalar(4.0) - curr_x.clone().mul_scalar(4.0)
                + Tensor::ones_like(&curr_x);

            (f_val, g)
        }
        let (f_init, g_init) = func(&x, 0.0, &d);
        let gtd_init = g_init.clone().dot(d.clone()).into_scalar().to_f64();
        println!("Initial State: f={},gtd = {}", f_init, gtd_init);
        assert!((f_init - 13.7080059052).abs() < tol);
        assert!((gtd_init - 28.5305728912).abs() < tol);
        let mut calls = 0;
        let mut obj_func =
            |xb: &Tensor<TestAutodiffBackend, 1>,
             tv: f64,
             dv: &Tensor<TestAutodiffBackend, 1>| {
                calls += 1;
                func(xb, tv, dv)
            };

        let (f_final, _g_final, t_final, evals) = strong_wolfe(
            &mut obj_func,
            &x,
            t_initial,
            &d,
            f_init,
            g_init,
            gtd_init,
            1e-4, // c1
            0.9,  // c2
            1e-9, // tolerance_change
            10,   // max_ls
        );
        let g_f = _g_final.into_scalar().to_f64();
        println!(
            "f_final:{:?},_g_final:{:?},t_final:{:?},evals:{:?}",
            f_final, g_f, t_final, evals
        );
        assert!((f_final - 13.708005905151367).abs() < tol);
        assert!((g_f - 31.2450428009).abs() < tol);
        assert!((t_final.to_f64() - 0.0).abs() < tol);
        assert_eq!(evals, 10);
        assert_eq!(evals, calls);
    }
}
#[test]
fn test_lbfgs_strong_wolfe_comparison() {
    let device = Default::default();
    let tol = 1e-5;
    let x_data = Tensor::<TestAutodiffBackend, 2>::from_data([[1.0], [2.0], [3.0]], &device);
    let y_true = Tensor::<TestAutodiffBackend, 2>::from_data([[3.0], [5.0], [7.0]], &device);
    let weight = TensorData::from([[0.5f64]]);
    let bias = TensorData::from([0.1f64]);
    let module = given_linear_layer(weight, bias);

    let mut optimizer: LBFGS<TestAutodiffBackend> = LBFGSConfig::new()
        .with_line_search_fn(LineSearchFn::StrongWolfe)
        .init();
    let mut closure = |mod_in: Linear<TestAutodiffBackend>| {
        let output = mod_in.forward(x_data.clone());
        let loss = ruda_nn::loss::MseLoss::new().forward(
            output,
            y_true.clone(),
            ruda_nn::loss::Reduction::Sum,
        );

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &mod_in);

        (loss.into_scalar().to_f64(), grads_params)
    };
    let initial_loss = closure(module.clone()).0;
    assert!((initial_loss - 50.1300048828).abs() < tol);
    let (updated_module, final_loss) = optimizer.step(0.001, module, &mut closure);
    assert!((final_loss - 0.0234732367).abs() < tol);
    let optimized_data: f64 = updated_module.weight.val().into_scalar().to_f64();
    let optimized_bias: f64 = updated_module
        .bias
        .as_ref()
        .unwrap()
        .val()
        .into_scalar()
        .to_f64();
    assert!((optimized_data - 2.0570652485).abs() < tol);
    assert!((optimized_bias - 0.8106800914).abs() < tol);
}
#[test]
fn test_lbfgs_no_strong_wolfe_comparison() {
    let device = Default::default();
    let tol = 1e-5;
    let x_data = Tensor::<TestAutodiffBackend, 2>::from_data([[1.0], [2.0], [3.0]], &device);
    let y_true = Tensor::<TestAutodiffBackend, 2>::from_data([[3.0], [5.0], [7.0]], &device);
    let weight = TensorData::from([[0.5f64]]);
    let bias = TensorData::from([0.1f64]);
    let module = given_linear_layer(weight, bias);

    let mut optimizer: LBFGS<TestAutodiffBackend> = LBFGSConfig::new()
        .with_line_search_fn(LineSearchFn::None)
        .init();
    let mut closure = |mod_in: Linear<TestAutodiffBackend>| {
        let output = mod_in.forward(x_data.clone());
        let loss = ruda_nn::loss::MseLoss::new().forward(
            output,
            y_true.clone(),
            ruda_nn::loss::Reduction::Sum,
        );

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &mod_in);

        (loss.into_scalar().to_f64(), grads_params)
    };
    let initial_loss = closure(module.clone()).0;
    assert!((initial_loss - 50.1300048828).abs() < tol);
    let (updated_module, final_loss) = optimizer.step(0.001, module, &mut closure);
    assert!((final_loss - 48.2181930542).abs() < tol);
    let optimized_data: f64 = updated_module.weight.val().into_scalar().to_f64();
    let optimized_bias: f64 = updated_module
        .bias
        .as_ref()
        .unwrap()
        .val()
        .into_scalar()
        .to_f64();

    assert!((optimized_data - 0.5302446192).abs() < tol);
    assert!((optimized_bias - 0.1142520783).abs() < tol);
}
