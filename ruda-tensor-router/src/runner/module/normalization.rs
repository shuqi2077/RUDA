use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn apply_rms_norm(&self, handles: &mut HandleContainer<B::Handle>, desc: &RmsNormOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let gamma = handles.get_float_tensor::<B>(&desc.gamma);
        let out = B::rms_norm_with_stats(x, gamma, desc.epsilon.elem());
        handles.register_float_tensor::<B>(&desc.out.id, out.output);
        handles.register_float_tensor::<B>(&desc.rstd.id, out.rstd);
    }

    pub(super) fn apply_rms_norm_backward(&self, handles: &mut HandleContainer<B::Handle>, desc: &RmsNormBackwardOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let gamma = handles.get_float_tensor::<B>(&desc.gamma);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let rstd = handles.get_float_tensor::<B>(&desc.rstd);
        let out = B::rms_norm_backward(x, gamma, grad, rstd);
        handles.register_float_tensor::<B>(&desc.input_grad.id, out.input);
        handles.register_float_tensor::<B>(&desc.weight_grad.id, out.weight);
    }

    pub(super) fn apply_layer_norm(&self, handles: &mut HandleContainer<B::Handle>, desc: &LayerNormOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let gamma = handles.get_float_tensor::<B>(&desc.gamma);
        let beta = desc.beta.as_ref().map(|value| handles.get_float_tensor::<B>(value));
        let out = B::layer_norm_with_stats(x, gamma, beta, desc.epsilon.elem());
        handles.register_float_tensor::<B>(&desc.out.id, out.output);
        handles.register_float_tensor::<B>(&desc.mean.id, out.mean);
        handles.register_float_tensor::<B>(&desc.rstd.id, out.rstd);
    }

    pub(super) fn apply_layer_norm_backward(&self, handles: &mut HandleContainer<B::Handle>, desc: &LayerNormBackwardOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let gamma = handles.get_float_tensor::<B>(&desc.gamma);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let mean = handles.get_float_tensor::<B>(&desc.mean);
        let rstd = handles.get_float_tensor::<B>(&desc.rstd);
        let out = B::layer_norm_backward(x, gamma, grad, mean, rstd);
        handles.register_float_tensor::<B>(&desc.input_grad.id, out.input);
        handles.register_float_tensor::<B>(&desc.weight_grad.id, out.weight);
        handles.register_float_tensor::<B>(&desc.bias_grad.id, out.bias);
    }
}
