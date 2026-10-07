
use ruda_model::config::Config;
use ruda_model::module::Content;
use ruda_model::module::DisplaySettings;
use ruda_model::module::Initializer;
use ruda_model::module::Module;
use ruda_model::module::ModuleDisplay;
use ruda_model::module::Param;
use ruda_model::tensor::{DType, FloatDType, Tensor};
use ruda_model::tensor::TensorPrimitive;
use ruda_model::tensor::backend::Backend;

/// Configuration to create a [LayerNorm](LayerNorm) layer using the [init function](LayerNormConfig::init).
#[derive(Debug, Config)]
pub struct LayerNormConfig {
    /// The size of the input features.
    pub d_model: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub epsilon: f64,
    /// If a bias (beta) should be applied during the normalization. Default: true
    #[config(default = true)]
    pub bias: bool,
}

/// Applies Layer Normalization over an input tensor as described in the paper [Layer Normalization](https://arxiv.org/abs/1607.06450).
///
/// `Y = norm(X) * γ + β`
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `γ` is the learnable weight (scale)
/// - `β` is the learnable bias (optional)
///
/// Should be created using [LayerNormConfig](LayerNormConfig).
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct LayerNorm<B: Backend> {
    /// The learnable weight (scale).
    pub gamma: Param<Tensor<B, 1>>,
    /// The learnable bias (optional).
    pub beta: Option<Param<Tensor<B, 1>>>,
    /// A value required for numerical stability.
    epsilon: f64,
}

impl LayerNormConfig {
    /// Initialize a new [layer norm](LayerNorm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        let gamma = Initializer::Ones.init([self.d_model], device);
        let beta = if self.bias {
            Some(Initializer::Zeros.init([self.d_model], device))
        } else {
            None
        };

        LayerNorm {
            gamma,
            beta,
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> LayerNorm<B> {
    /// Assemble actual loaded affine leaves and the original epsilon without initialization.
    pub fn from_parameters(gamma: Param<Tensor<B,1>>,beta: Option<Param<Tensor<B,1>>>,epsilon:f64) -> Self {
        if let Some(beta) = &beta {assert_eq!(gamma.val().dims(),beta.val().dims(),"LayerNorm affine dimensions differ");}
        Self {gamma,beta,epsilon}
    }

    /// The configured normalization epsilon, independent of parameter storage.
    pub fn epsilon(&self) -> f64 { self.epsilon }

    pub fn forward_with_compute_dtype<const D: usize>(
        &self,
        input: Tensor<B, D>,
        dtype: FloatDType,
    ) -> Tensor<B, D> {
        let output_dtype = input.dtype();
        let dtype: DType = dtype.into();
        let gamma = self.gamma.val().cast(dtype).into_primitive().tensor();
        let beta = self.beta.as_ref()
            .map(|b| b.val().cast(dtype).into_primitive().tensor());
        Tensor::<B, D>::from_primitive(TensorPrimitive::Float(B::layer_norm(
            input.cast(dtype).into_primitive().tensor(),
            gamma,
            beta,
            self.epsilon,
        )))
        .cast(output_dtype)
    }
    /// Applies the forward pass on the input tensor.
    ///
    /// See the [LayerNorm](LayerNorm) documentation for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let gamma = self.gamma.val().into_primitive().tensor();
        let beta = self
            .beta
            .as_ref()
            .map(|b| b.val().into_primitive().tensor());

        Tensor::from_primitive(TensorPrimitive::Float(B::layer_norm(
            input.into_primitive().tensor(),
            gamma,
            beta,
            self.epsilon,
        )))
    }
}

impl<B: Backend> ModuleDisplay for LayerNorm<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_model] = self.gamma.shape().dims();
        content
            .add("d_model", &d_model)
            .add("epsilon", &self.epsilon)
            .add("bias", &self.beta.is_some())
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use ruda_model::tensor::TensorData;
    use ruda_model::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[cfg(feature = "std")]
    use crate::{TestAutodiffBackend, TestBackend};

    #[cfg(not(feature = "std"))]
    use crate::TestBackend;

    #[test]
    fn layer_norm_forward() {
        let device = Default::default();
        let module = LayerNormConfig::new(10).init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[
                -0.6897, -2.7106, 2.2222, -1.0330, -0.8933, 1.1765, 0.0601, 1.5252, -0.3630, 0.6728,
            ]]),
            &device,
        );

        let output = module.forward(input);

        let expected = TensorData::from([[
            -0.4990, -1.9680, 1.6178, -0.7486, -0.6470, 0.8576, 0.0461, 1.1111, -0.2614, 0.4915,
        ]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn layer_norm_forward_large_epsilon() {
        let device = Default::default();
        let module = LayerNormConfig::new(10)
            .with_epsilon(1e-1)
            .init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[
                -0.6897, -2.7106, 2.2222, -1.0330, -0.8933, 1.1765, 0.0601, 1.5252, -0.3630, 0.6728,
            ]]),
            &device,
        );

        let output = module.forward(input);

        let expected = TensorData::from([[
            -0.4863, -1.9180, 1.5766, -0.7295, -0.6305, 0.8358, 0.0449, 1.0828, -0.2548, 0.4790,
        ]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn layer_norm_forward_no_bias() {
        let device = Default::default();
        let module = LayerNormConfig::new(10)
            .with_bias(false)
            .init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[
                -0.6897, -2.7106, 2.2222, -1.0330, -0.8933, 1.1765, 0.0601, 1.5252, -0.3630, 0.6728,
            ]]),
            &device,
        );

        let output = module.forward(input);

        // With bias=false, output matches the bias=true case (beta is zero-initialized
        // by default), confirming the `None` branch in the backend hook produces the
        // pre-beta result.
        let expected = TensorData::from([[
            -0.4990, -1.9680, 1.6178, -0.7486, -0.6470, 0.8576, 0.0461, 1.1111, -0.2614, 0.4915,
        ]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[cfg(feature = "std")]
    #[test]
    fn layer_norm_backward() {
        let device = Default::default();
        let module = LayerNormConfig::new(2).init::<TestAutodiffBackend>(&device);
        let tensor_1 = Tensor::<TestAutodiffBackend, 2>::from_data(
            TensorData::from([[0.0, 1.0], [3.0, 4.0]]),
            &device,
        )
        .require_grad();
        let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(
            TensorData::from([[6.0, 7.0], [9.0, 10.0]]),
            &device,
        )
        .require_grad();

        let x = tensor_1.clone().matmul(tensor_2.clone());

        let output = module.forward(x);
        let grads = output.backward();

        let tensor_1_grad = tensor_1.grad(&grads).unwrap();
        let tensor_2_grad = tensor_2.grad(&grads).unwrap();
        let gamma_grad = module.gamma.grad(&grads).unwrap();
        let beta_grad = module.beta.as_ref().unwrap().grad(&grads).unwrap();

        let expected = TensorData::from([-2.0, 2.0]);
        gamma_grad
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        let expected = TensorData::from([2.0, 2.0]);
        beta_grad
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        let expected = TensorData::zeros::<f32, _>(tensor_1_grad.shape());
        tensor_1_grad
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());

        let expected = TensorData::zeros::<f32, _>(tensor_2_grad.shape());
        tensor_2_grad
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[cfg(feature = "std")]
    #[test]
    fn layer_norm_affine_gradients_with_and_without_bias() {
        let device = Default::default();
        for has_bias in [true, false] {
            let module = LayerNormConfig::new(3).with_bias(has_bias).init::<TestAutodiffBackend>(&device);
            let x = Tensor::<TestAutodiffBackend, 2>::from_data([[1., 2., 4.], [3., -1., 2.]], &device).require_grad();
            let dy = Tensor::<TestAutodiffBackend, 2>::from_data([[0.2, -0.5, 0.7], [1.1, 0.3, -0.4]], &device);
            let gradients = (module.forward(x.clone()) * dy).backward();
            let mut dx = [[0f32; 3]; 2]; let mut dw = [0f32; 3]; let mut db = [0f32; 3];
            for (r, (row, grad)) in [[1f64, 2., 4.], [3., -1., 2.]].into_iter()
                .zip([[0.2, -0.5, 0.7], [1.1, 0.3, -0.4]]).enumerate() {
                let mean = row.iter().sum::<f64>()/3.;
                let rstd = (row.iter().map(|v| (v-mean).powi(2)).sum::<f64>()/3.+1e-5).sqrt().recip();
                let norm = row.map(|v| (v-mean)*rstd);
                let gmean = grad.iter().sum::<f64>()/3.;
                let dot = grad.iter().zip(norm).map(|(g,n)| g*n).sum::<f64>()/3.;
                for c in 0..3 {
                    dx[r][c] = (rstd*(grad[c]-gmean-norm[c]*dot)) as f32;
                    dw[c] += (grad[c]*norm[c]) as f32; db[c] += grad[c] as f32;
                }
            }
            x.grad(&gradients).unwrap().to_data().assert_approx_eq::<FT>(&TensorData::from(dx), Tolerance::default());
            module.gamma.val().grad(&gradients).unwrap().to_data().assert_approx_eq::<FT>(&TensorData::from(dw), Tolerance::default());
            if let Some(beta) = &module.beta {
                beta.val().grad(&gradients).unwrap().to_data().assert_approx_eq::<FT>(&TensorData::from(db), Tolerance::default());
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn layer_norm_compute_dtype_preserves_storage_and_original_gradients() {
        let device = Default::default();
        for dtype in [DType::F16, DType::BF16] {
            let module = LayerNormConfig::new(3).init::<TestAutodiffBackend>(&device);
            let input = Tensor::<TestAutodiffBackend, 2>::from_floats([[1., 2., 4.], [3., -1., 2.]], &device)
                .cast(dtype).require_grad();
            let reference = module.forward(input.clone().cast(DType::F32)).cast(dtype);
            let output = module.forward_with_compute_dtype(input.clone(), FloatDType::F32);
            assert_eq!(output.dtype(), dtype);
            assert_eq!(output.dims(), input.dims());
            output.clone().cast(DType::F32).to_data().assert_approx_eq::<f32>(
                &reference.clone().cast(DType::F32).to_data(), Tolerance::absolute(1e-6));
            let expected_grads = reference.square().sum().backward();
            let grads = output.square().sum().backward();
            input.grad(&grads).unwrap().cast(DType::F32).to_data().assert_approx_eq::<f32>(
                &input.grad(&expected_grads).unwrap().cast(DType::F32).to_data(), Tolerance::absolute(1e-6));
            for param in [module.gamma.val(), module.beta.as_ref().unwrap().val()] {
                param.grad(&grads).unwrap().to_data().assert_approx_eq::<f32>(
                    &param.grad(&expected_grads).unwrap().to_data(), Tolerance::absolute(1e-6));
            }
        }
    }

    #[test]
    fn display() {
        let config = LayerNormConfig::new(6);
        let layer_norm = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            format!("{layer_norm}"),
            "LayerNorm {d_model: 6, epsilon: 0.00001, bias: true, params: 12}"
        );
    }

    #[test]
    fn display_no_bias() {
        let config = LayerNormConfig::new(6).with_bias(false);
        let layer_norm = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            format!("{layer_norm}"),
            "LayerNorm {d_model: 6, epsilon: 0.00001, bias: false, params: 6}"
        );
    }
}
