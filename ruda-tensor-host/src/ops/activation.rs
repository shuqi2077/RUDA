use ruda_tensor::Scalar;
use ruda_tensor::ops::ActivationOps;
use ruda_tensor::tensor::FloatTensor;
use crate::Host;

pub use rudnn_host::activation::{softmax, layer_norm};

impl ActivationOps<Host> for Host {
    fn relu(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        rudnn_host::activation::relu(tensor)
    }

    fn relu_backward(output: FloatTensor<Host>, grad: FloatTensor<Host>) -> FloatTensor<Host> {
        rudnn_host::activation::relu_backward(output, grad)
    }

    fn leaky_relu(tensor: FloatTensor<Host>, negative_slope: Scalar) -> FloatTensor<Host> {
        rudnn_host::activation::leaky_relu(tensor, negative_slope)
    }

    fn prelu(tensor: FloatTensor<Host>, alpha: FloatTensor<Host>) -> FloatTensor<Host> {
        rudnn_host::activation::prelu(tensor, alpha)
    }

    fn gelu(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        rudnn_host::activation::gelu(tensor)
    }

    fn gelu_backward(x: FloatTensor<Host>, grad: FloatTensor<Host>) -> FloatTensor<Host> {
        rudnn_host::activation::gelu_backward(x, grad)
    }

    fn sigmoid(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        rudnn_host::activation::sigmoid(tensor)
    }

    fn sigmoid_backward(output: FloatTensor<Host>, grad: FloatTensor<Host>) -> FloatTensor<Host> {
        rudnn_host::activation::sigmoid_backward(output, grad)
    }

    fn hard_sigmoid(tensor: FloatTensor<Host>, alpha: Scalar, beta: Scalar) -> FloatTensor<Host> {
        rudnn_host::activation::hard_sigmoid(tensor, alpha, beta)
    }

    fn log_sigmoid(tensor: FloatTensor<Host>) -> FloatTensor<Host> {
        rudnn_host::activation::log_sigmoid(tensor)
    }

    fn log_sigmoid_backward(x: FloatTensor<Host>, grad: FloatTensor<Host>) -> FloatTensor<Host> {
        rudnn_host::activation::log_sigmoid_backward(x, grad)
    }

    fn softmax(tensor: FloatTensor<Host>, dim: usize) -> FloatTensor<Host> {
        rudnn_host::activation::softmax(tensor, dim)
    }
}
