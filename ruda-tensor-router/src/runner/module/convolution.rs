use super::*;
use ruda_tensor::graph::{
    Conv1dBiasBackwardOpIr, Conv1dOpIr, Conv1dWeightBackwardOpIr, Conv1dXBackwardOpIr,
    Conv2dBiasBackwardOpIr, Conv2dOpIr, Conv2dWeightBackwardOpIr, Conv2dXBackwardOpIr,
    Conv3dBiasBackwardOpIr, Conv3dOpIr, Conv3dWeightBackwardOpIr, Conv3dXBackwardOpIr,
    ConvTranspose1dOpIr, ConvTranspose2dOpIr, ConvTranspose3dOpIr, DeformConv2dBackwardOpIr,
    DeformConv2dOpIr,
};

impl<B: BackendIr> Runner<B> {
    pub(super) fn apply_conv1d(&self, handles: &mut HandleContainer<B::Handle>, desc: &Conv1dOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let bias = desc
            .bias
            .as_ref()
            .map(|bias| handles.get_float_tensor::<B>(bias));

        let output = B::conv1d(x, weight, bias, desc.clone().options.into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv1d_xbackward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &Conv1dXBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::conv1d_x_backward(x, weight, output_grad, desc.clone().options.into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv1d_weight_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &Conv1dWeightBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::conv1d_weight_backward(x, weight, output_grad, desc.clone().options.into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv1d_bias_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &Conv1dBiasBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let bias = handles.get_float_tensor::<B>(&desc.bias);
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::conv1d_bias_backward(x, bias, output_grad);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv2d(&self, handles: &mut HandleContainer<B::Handle>, desc: &Conv2dOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let bias = desc
            .bias
            .as_ref()
            .map(|bias| handles.get_float_tensor::<B>(bias));

        let output = B::conv2d(x, weight, bias, desc.clone().options.into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv2d_xbackward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &Conv2dXBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::conv2d_x_backward(x, weight, output_grad, desc.clone().options.into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv2d_weight_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &Conv2dWeightBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::conv2d_weight_backward(x, weight, output_grad, desc.clone().options.into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv2d_bias_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &Conv2dBiasBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let bias = handles.get_float_tensor::<B>(&desc.bias);
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::conv2d_bias_backward(x, bias, output_grad);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv3d(&self, handles: &mut HandleContainer<B::Handle>, desc: &Conv3dOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let bias = desc
            .bias
            .as_ref()
            .map(|bias| handles.get_float_tensor::<B>(bias));

        let output = B::conv3d(x, weight, bias, desc.options.clone().into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv3d_xbackward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &Conv3dXBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::conv3d_x_backward(x, weight, output_grad, desc.clone().options.into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv3d_weight_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &Conv3dWeightBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::conv3d_weight_backward(x, weight, output_grad, desc.clone().options.into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv3d_bias_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &Conv3dBiasBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let bias = handles.get_float_tensor::<B>(&desc.bias);
        let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

        let output = B::conv3d_bias_backward(x, bias, output_grad);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_deformable_conv2d(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &Box<DeformConv2dOpIr>,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let offset = handles.get_float_tensor::<B>(&desc.offset);
        let mask = desc
            .mask
            .as_ref()
            .map(|mask| handles.get_float_tensor::<B>(mask));
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let bias = desc
            .bias
            .as_ref()
            .map(|bias| handles.get_float_tensor::<B>(bias));

        let output = B::deform_conv2d(x, offset, weight, mask, bias, desc.options.clone().into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_deformable_conv2d_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &Box<DeformConv2dBackwardOpIr>,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let offset = handles.get_float_tensor::<B>(&desc.offset);
        let mask = desc
            .mask
            .as_ref()
            .map(|mask| handles.get_float_tensor::<B>(mask));
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let bias = desc
            .bias
            .as_ref()
            .map(|bias| handles.get_float_tensor::<B>(bias));
        let output_grad = handles.get_float_tensor::<B>(&desc.out_grad);

        let output = B::deform_conv2d_backward(
            x,
            offset,
            weight,
            mask,
            bias,
            output_grad,
            desc.options.clone().into(),
        );

        handles.register_float_tensor::<B>(&desc.input_grad.id, output.x_grad);
        handles.register_float_tensor::<B>(&desc.offset_grad.id, output.offset_grad);
        handles.register_float_tensor::<B>(&desc.weight_grad.id, output.weight_grad);
        if let Some((mask_grad, field)) = output.mask_grad.zip(desc.mask_grad.as_ref()) {
            handles.register_float_tensor::<B>(&field.id, mask_grad);
        }
        if let Some((bias_grad, field)) = output.bias_grad.zip(desc.bias_grad.as_ref()) {
            handles.register_float_tensor::<B>(&field.id, bias_grad);
        }
    }

    pub(super) fn apply_conv_transpose1d(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &ConvTranspose1dOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let bias = desc
            .bias
            .as_ref()
            .map(|bias| handles.get_float_tensor::<B>(bias));

        let output = B::conv_transpose1d(x, weight, bias, desc.options.clone().into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv_transpose2d(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &ConvTranspose2dOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let bias = desc
            .bias
            .as_ref()
            .map(|bias| handles.get_float_tensor::<B>(bias));

        let output = B::conv_transpose2d(x, weight, bias, desc.options.clone().into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_conv_transpose3d(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &ConvTranspose3dOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let weight = handles.get_float_tensor::<B>(&desc.weight);
        let bias = desc
            .bias
            .as_ref()
            .map(|bias| handles.get_float_tensor::<B>(bias));

        let output = B::conv_transpose3d(x, weight, bias, desc.options.clone().into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }
}
