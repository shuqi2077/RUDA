use super::*;
use ruda_tensor::graph::{
    AdaptiveAvgPool1dBackwardOpIr, AdaptiveAvgPool1dOpIr, AdaptiveAvgPool2dBackwardOpIr,
    AdaptiveAvgPool2dOpIr, AvgPool1dBackwardOpIr, AvgPool1dOpIr, AvgPool2dBackwardOpIr,
    AdaptiveAvgPool3dOpIr, AdaptiveAvgPool3dBackwardOpIr,
    AvgPool3dOpIr, AvgPool3dBackwardOpIr,
    AvgPool2dOpIr, InterpolateBackwardOpIr, InterpolateOpIr, MaxPool1dOpIr,
    MaxPool1dWithIndicesBackwardOpIr, MaxPool1dWithIndicesOpIr, MaxPool2dOpIr,
    MaxPool2dWithIndicesBackwardOpIr, MaxPool2dWithIndicesOpIr,
    MaxPool3dOpIr, MaxPool3dWithIndicesOpIr, MaxPool3dWithIndicesBackwardOpIr,
    Interpolate1dOpIr, Interpolate1dBackwardOpIr, Interpolate3dOpIr, Interpolate3dBackwardOpIr,
};

impl<B: BackendIr> Runner<B> {
    pub(super) fn apply_avg_pool3d(&self, handles: &mut HandleContainer<B::Handle>, desc: &AvgPool3dOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let output = B::avg_pool3d(x, desc.kernel_size, desc.stride, desc.padding,
            desc.count_include_pad, desc.ceil_mode);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_avg_pool3d_backward(&self, handles: &mut HandleContainer<B::Handle>, desc: &AvgPool3dBackwardOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let output = B::avg_pool3d_backward(x, grad, desc.kernel_size, desc.stride, desc.padding,
            desc.count_include_pad, desc.ceil_mode);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_adaptive_avg_pool3d(&self, handles: &mut HandleContainer<B::Handle>,
        desc: &AdaptiveAvgPool3dOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let output = B::adaptive_avg_pool3d(x, desc.output_size);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_adaptive_avg_pool3d_backward(&self, handles: &mut HandleContainer<B::Handle>,
        desc: &AdaptiveAvgPool3dBackwardOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let output = B::adaptive_avg_pool3d_backward(x, grad);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_avg_pool1d(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &AvgPool1dOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);

        let output = B::avg_pool1d(
            x,
            desc.kernel_size,
            desc.stride,
            desc.padding,
            desc.count_include_pad,
            desc.ceil_mode,
        );
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_avg_pool2d(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &AvgPool2dOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);

        let output = B::avg_pool2d(
            x,
            desc.kernel_size,
            desc.stride,
            desc.padding,
            desc.count_include_pad,
            desc.ceil_mode,
        );
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_avg_pool1d_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &AvgPool1dBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);

        let output = B::avg_pool1d_backward(
            x,
            grad,
            desc.kernel_size,
            desc.stride,
            desc.padding,
            desc.count_include_pad,
            desc.ceil_mode,
        );
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_avg_pool2d_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &AvgPool2dBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);

        let output = B::avg_pool2d_backward(
            x,
            grad,
            desc.kernel_size,
            desc.stride,
            desc.padding,
            desc.count_include_pad,
            desc.ceil_mode,
        );
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_adaptive_avg_pool1d(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &AdaptiveAvgPool1dOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);

        let output = B::adaptive_avg_pool1d(x, desc.output_size);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_adaptive_avg_pool2d(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &AdaptiveAvgPool2dOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);

        let output = B::adaptive_avg_pool2d(x, desc.output_size);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_adaptive_avg_pool1d_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &AdaptiveAvgPool1dBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);

        let output = B::adaptive_avg_pool1d_backward(x, grad);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_adaptive_avg_pool2d_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &AdaptiveAvgPool2dBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);

        let output = B::adaptive_avg_pool2d_backward(x, grad);
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_max_pool1d(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &MaxPool1dOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);

        let output = B::max_pool1d(
            x,
            desc.kernel_size,
            desc.stride,
            desc.padding,
            desc.dilation,
            desc.ceil_mode,
        );
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_max_pool1d_with_indices(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &MaxPool1dWithIndicesOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);

        let output = B::max_pool1d_with_indices(
            x,
            desc.kernel_size,
            desc.stride,
            desc.padding,
            desc.dilation,
            desc.ceil_mode,
        );
        handles.register_float_tensor::<B>(&desc.out.id, output.output);
        handles.register_int_tensor::<B>(&desc.out_indices.id, output.indices);
    }

    pub(super) fn apply_max_pool1d_with_indices_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &MaxPool1dWithIndicesBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let output_grad = handles.get_float_tensor::<B>(&desc.grad);
        let indices = handles.get_int_tensor::<B>(&desc.indices);

        let output = B::max_pool1d_with_indices_backward(
            x,
            desc.kernel_size,
            desc.stride,
            desc.padding,
            desc.dilation,
            desc.ceil_mode,
            output_grad,
            indices,
        );
        handles.register_float_tensor::<B>(&desc.out.id, output.x_grad);
    }

    pub(super) fn apply_max_pool2d(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &MaxPool2dOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);

        let output = B::max_pool2d(
            x,
            desc.kernel_size,
            desc.stride,
            desc.padding,
            desc.dilation,
            desc.ceil_mode,
        );
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_max_pool2d_with_indices(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &MaxPool2dWithIndicesOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);

        let output = B::max_pool2d_with_indices(
            x,
            desc.kernel_size,
            desc.stride,
            desc.padding,
            desc.dilation,
            desc.ceil_mode,
        );
        handles.register_float_tensor::<B>(&desc.out.id, output.output);
        handles.register_int_tensor::<B>(&desc.out_indices.id, output.indices);
    }

    pub(super) fn apply_max_pool2d_with_indices_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &MaxPool2dWithIndicesBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let output_grad = handles.get_float_tensor::<B>(&desc.grad);
        let indices = handles.get_int_tensor::<B>(&desc.indices);

        let output = B::max_pool2d_with_indices_backward(
            x,
            desc.kernel_size,
            desc.stride,
            desc.padding,
            desc.dilation,
            desc.ceil_mode,
            output_grad,
            indices,
        );
        handles.register_float_tensor::<B>(&desc.out.id, output.x_grad);
    }

    pub(super) fn apply_max_pool3d(&self, handles: &mut HandleContainer<B::Handle>, desc: &MaxPool3dOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let out = B::max_pool3d(x, desc.kernel_size, desc.stride, desc.padding, desc.dilation, desc.ceil_mode);
        handles.register_float_tensor::<B>(&desc.out.id, out);
    }

    pub(super) fn apply_max_pool3d_with_indices(&self, handles: &mut HandleContainer<B::Handle>,
        desc: &MaxPool3dWithIndicesOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let out = B::max_pool3d_with_indices(x, desc.kernel_size, desc.stride,
            desc.padding, desc.dilation, desc.ceil_mode);
        handles.register_float_tensor::<B>(&desc.out.id, out.output);
        handles.register_int_tensor::<B>(&desc.out_indices.id, out.indices);
    }

    pub(super) fn apply_max_pool3d_with_indices_backward(&self, handles: &mut HandleContainer<B::Handle>,
        desc: &MaxPool3dWithIndicesBackwardOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let indices = handles.get_int_tensor::<B>(&desc.indices);
        let out = B::max_pool3d_with_indices_backward(x, grad, indices, desc.kernel_size,
            desc.stride, desc.padding, desc.dilation, desc.ceil_mode);
        handles.register_float_tensor::<B>(&desc.out.id, out.x_grad);
    }

    pub(super) fn apply_interpolate1d(&self, handles: &mut HandleContainer<B::Handle>, desc: &Interpolate1dOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let out = B::interpolate1d(x, desc.output_size, desc.options.clone().into());
        handles.register_float_tensor::<B>(&desc.out.id, out);
    }

    pub(super) fn apply_interpolate3d(&self, handles: &mut HandleContainer<B::Handle>, desc: &Interpolate3dOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let out = B::interpolate3d(x, desc.output_size, desc.options.clone().into());
        handles.register_float_tensor::<B>(&desc.out.id, out);
    }

    pub(super) fn apply_interpolate1d_backward(&self, handles: &mut HandleContainer<B::Handle>, desc: &Interpolate1dBackwardOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let out = B::interpolate1d_backward(x, grad, desc.output_size, desc.options.clone().into());
        handles.register_float_tensor::<B>(&desc.out.id, out);
    }

    pub(super) fn apply_interpolate3d_backward(&self, handles: &mut HandleContainer<B::Handle>, desc: &Interpolate3dBackwardOpIr) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);
        let out = B::interpolate3d_backward(x, grad, desc.output_size, desc.options.clone().into());
        handles.register_float_tensor::<B>(&desc.out.id, out);
    }

    pub(super) fn apply_interpolate(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &InterpolateOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);

        let output = B::interpolate(x, desc.output_size, desc.options.clone().into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_interpolate_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &InterpolateBackwardOpIr,
    ) {
        let x = handles.get_float_tensor::<B>(&desc.x);
        let grad = handles.get_float_tensor::<B>(&desc.grad);

        let output =
            B::interpolate_backward(x, grad, desc.output_size, desc.options.clone().into());
        handles.register_float_tensor::<B>(&desc.out.id, output);
    }
}
