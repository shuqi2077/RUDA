use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn apply_prelu_native(&self, handles: &mut HandleContainer<B::Handle>, desc: &PreluOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let alpha = handles.get_float_tensor::<B>(&desc.alpha);
        handles.register_float_tensor::<B>(&desc.out.id, B::prelu_native(x, alpha));
    }

    pub(super) fn apply_prelu_native_backward_select(&self, handles: &mut HandleContainer<B::Handle>, desc: &PreluBackwardSelectOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let alpha = handles.get_float_tensor::<B>(&desc.alpha);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let output = B::prelu_native_backward_select(x, alpha, grad, [desc.input_grad.is_some(), desc.weight_grad.is_some()]);
        for (target, value) in [desc.input_grad.as_ref(), desc.weight_grad.as_ref()].into_iter().zip(output) {
            if let Some(target) = target { handles.register_float_tensor::<B>(&target.id, value.expect("requested PReLU gradient")); }
        }
    }

    pub(super) fn apply_group_norm(&self, handles: &mut HandleContainer<B::Handle>, desc: &GroupNormOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let gamma = desc.gamma.as_ref().map(|value| handles.get_float_tensor::<B>(value));
        let beta = desc.beta.as_ref().map(|value| handles.get_float_tensor::<B>(value));
        let out = B::group_norm_with_stats(x, gamma, beta, desc.groups, desc.epsilon.elem());
        handles.register_float_tensor::<B>(&desc.out.id, out.output);
        handles.register_float_tensor::<B>(&desc.mean.id, out.mean);
        handles.register_float_tensor::<B>(&desc.rstd.id, out.rstd);
    }

    pub(super) fn apply_group_norm_backward_select(&self, handles: &mut HandleContainer<B::Handle>, desc: &GroupNormBackwardSelectOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let gamma = desc.gamma.as_ref().map(|value| handles.get_float_tensor::<B>(value));
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let mean = handles.get_float_tensor::<B>(&desc.mean);
        let rstd = handles.get_float_tensor::<B>(&desc.rstd);
        let out = B::group_norm_backward_select(x, gamma, grad, mean, rstd, desc.groups,
            [desc.input_grad.is_some(), desc.weight_grad.is_some(), desc.bias_grad.is_some()]);
        for (target, value) in [desc.input_grad.as_ref(), desc.weight_grad.as_ref(), desc.bias_grad.as_ref()].into_iter().zip(out) {
            if let Some(target) = target { handles.register_float_tensor::<B>(&target.id, value.expect("requested GroupNorm gradient")); }
        }
    }

    pub(super) fn apply_gelu_native(&self, handles: &mut HandleContainer<B::Handle>, desc: &GeluOpIr) {
        let input = handles.get_float_tensor::<B>(&desc.x);
        handles.register_float_tensor::<B>(&desc.out.id, B::gelu_native(input, desc.approximate));
    }

    pub(super) fn apply_gelu_native_backward(&self, handles: &mut HandleContainer<B::Handle>, desc: &GeluBackwardOpIr) {
        let input = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        handles.register_float_tensor::<B>(&desc.out.id, B::gelu_native_backward(input, grad, desc.approximate));
    }

    pub(super) fn apply_silu_native(&self, handles: &mut HandleContainer<B::Handle>, desc: &UnaryOpIr) {
        let input = handles.get_float_tensor::<B>(&desc.input);
        handles.register_float_tensor::<B>(&desc.out.id, B::silu_native(input));
    }

    pub(super) fn apply_silu_native_backward(&self, handles: &mut HandleContainer<B::Handle>, desc: &SiluBackwardOpIr) {
        let input = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        handles.register_float_tensor::<B>(&desc.out.id, B::silu_native_backward(input, grad));
    }

    pub(super) fn apply_softmax(&self, handles: &mut HandleContainer<B::Handle>, desc: &SoftmaxOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let out = B::softmax_with_stats(x, desc.dim, desc.logarithmic);
        handles.register_float_tensor::<B>(&desc.out.id, out.output);
        handles.register_float_tensor::<B>(&desc.working.id, out.working);
    }

    pub(super) fn apply_softmax_backward(&self, handles: &mut HandleContainer<B::Handle>, desc: &SoftmaxBackwardOpIr) {
        let working = handles.get_float_tensor::<B>(&desc.working);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let out = B::softmax_native_backward(working, grad, desc.dim, desc.logarithmic);
        handles.register_float_tensor::<B>(&desc.out.id, out);
    }

    pub(super) fn apply_rms_norm_backward_select(&self, handles: &mut HandleContainer<B::Handle>, desc: &RmsNormBackwardSelectOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let gamma = handles.get_float_tensor::<B>(&desc.gamma);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let rstd = handles.get_float_tensor::<B>(&desc.rstd);
        let out = B::rms_norm_backward_select(x, gamma, grad, rstd,
            [desc.input_grad.is_some(), desc.weight_grad.is_some()]);
        for (target, value) in [desc.input_grad.as_ref(), desc.weight_grad.as_ref()].into_iter().zip(out) {
            if let Some(target) = target { handles.register_float_tensor::<B>(&target.id, value.expect("requested RMSNorm gradient")); }
        }
    }

    pub(super) fn apply_layer_norm_backward_select(&self, handles: &mut HandleContainer<B::Handle>, desc: &LayerNormBackwardSelectOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let gamma = handles.get_float_tensor::<B>(&desc.gamma);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let mean = handles.get_float_tensor::<B>(&desc.mean);
        let rstd = handles.get_float_tensor::<B>(&desc.rstd);
        let out = B::layer_norm_backward_select(x, gamma, grad, mean, rstd,
            [desc.input_grad.is_some(), desc.weight_grad.is_some(), desc.bias_grad.is_some()]);
        for (target, value) in [desc.input_grad.as_ref(), desc.weight_grad.as_ref(), desc.bias_grad.as_ref()].into_iter().zip(out) {
            if let Some(target) = target { handles.register_float_tensor::<B>(&target.id, value.expect("requested LayerNorm gradient")); }
        }
    }

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
