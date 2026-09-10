mod attention;
mod convolution;
mod linear;
mod loss;
mod pooling;
mod spectral;

use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn run_module(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        op: &ModuleOperationIr,
    ) {
        match op {
            ModuleOperationIr::Embedding(desc) => self.apply_embedding(handles, desc),
            ModuleOperationIr::EmbeddingBackward(desc) => {
                self.apply_embedding_backward(handles, desc)
            }
            ModuleOperationIr::Linear(desc) => self.apply_linear(handles, desc),
            ModuleOperationIr::LinearXBackward(desc) => self.apply_linear_xbackward(handles, desc),
            ModuleOperationIr::LinearWeightBackward(desc) => {
                self.apply_linear_weight_backward(handles, desc)
            }
            ModuleOperationIr::LinearBiasBackward(desc) => {
                self.apply_linear_bias_backward(handles, desc)
            }
            ModuleOperationIr::Conv1d(desc) => self.apply_conv1d(handles, desc),
            ModuleOperationIr::Conv1dXBackward(desc) => self.apply_conv1d_xbackward(handles, desc),
            ModuleOperationIr::Conv1dWeightBackward(desc) => {
                self.apply_conv1d_weight_backward(handles, desc)
            }
            ModuleOperationIr::Conv1dBiasBackward(desc) => {
                self.apply_conv1d_bias_backward(handles, desc)
            }
            ModuleOperationIr::Conv2d(desc) => self.apply_conv2d(handles, desc),
            ModuleOperationIr::Conv2dXBackward(desc) => self.apply_conv2d_xbackward(handles, desc),
            ModuleOperationIr::Conv2dWeightBackward(desc) => {
                self.apply_conv2d_weight_backward(handles, desc)
            }
            ModuleOperationIr::Conv2dBiasBackward(desc) => {
                self.apply_conv2d_bias_backward(handles, desc)
            }
            ModuleOperationIr::Conv3d(desc) => self.apply_conv3d(handles, desc),
            ModuleOperationIr::Conv3dXBackward(desc) => self.apply_conv3d_xbackward(handles, desc),
            ModuleOperationIr::Conv3dWeightBackward(desc) => {
                self.apply_conv3d_weight_backward(handles, desc)
            }
            ModuleOperationIr::Conv3dBiasBackward(desc) => {
                self.apply_conv3d_bias_backward(handles, desc)
            }
            ModuleOperationIr::DeformableConv2d(desc) => {
                self.apply_deformable_conv2d(handles, desc)
            }
            ModuleOperationIr::DeformableConv2dBackward(desc) => {
                self.apply_deformable_conv2d_backward(handles, desc)
            }
            ModuleOperationIr::ConvTranspose1d(desc) => self.apply_conv_transpose1d(handles, desc),
            ModuleOperationIr::ConvTranspose2d(desc) => self.apply_conv_transpose2d(handles, desc),
            ModuleOperationIr::ConvTranspose3d(desc) => self.apply_conv_transpose3d(handles, desc),
            ModuleOperationIr::AvgPool1d(desc) => self.apply_avg_pool1d(handles, desc),
            ModuleOperationIr::AvgPool2d(desc) => self.apply_avg_pool2d(handles, desc),
            ModuleOperationIr::AvgPool1dBackward(desc) => {
                self.apply_avg_pool1d_backward(handles, desc)
            }
            ModuleOperationIr::AvgPool2dBackward(desc) => {
                self.apply_avg_pool2d_backward(handles, desc)
            }
            ModuleOperationIr::AdaptiveAvgPool1d(desc) => {
                self.apply_adaptive_avg_pool1d(handles, desc)
            }
            ModuleOperationIr::AdaptiveAvgPool2d(desc) => {
                self.apply_adaptive_avg_pool2d(handles, desc)
            }
            ModuleOperationIr::AdaptiveAvgPool1dBackward(desc) => {
                self.apply_adaptive_avg_pool1d_backward(handles, desc)
            }
            ModuleOperationIr::AdaptiveAvgPool2dBackward(desc) => {
                self.apply_adaptive_avg_pool2d_backward(handles, desc)
            }
            ModuleOperationIr::MaxPool1d(desc) => self.apply_max_pool1d(handles, desc),
            ModuleOperationIr::MaxPool1dWithIndices(desc) => {
                self.apply_max_pool1d_with_indices(handles, desc)
            }
            ModuleOperationIr::MaxPool1dWithIndicesBackward(desc) => {
                self.apply_max_pool1d_with_indices_backward(handles, desc)
            }
            ModuleOperationIr::MaxPool2d(desc) => self.apply_max_pool2d(handles, desc),
            ModuleOperationIr::MaxPool2dWithIndices(desc) => {
                self.apply_max_pool2d_with_indices(handles, desc)
            }
            ModuleOperationIr::MaxPool2dWithIndicesBackward(desc) => {
                self.apply_max_pool2d_with_indices_backward(handles, desc)
            }
            ModuleOperationIr::Interpolate(desc) => self.apply_interpolate(handles, desc),
            ModuleOperationIr::InterpolateBackward(desc) => {
                self.apply_interpolate_backward(handles, desc)
            }
            ModuleOperationIr::Rfft(desc) => self.apply_rfft(handles, desc),
            ModuleOperationIr::IRfft(desc) => self.apply_irfft(handles, desc),
            ModuleOperationIr::Attention(desc) => self.apply_attention(handles, desc),
            ModuleOperationIr::CtcLoss(desc) => self.apply_ctc_loss(handles, desc),
            ModuleOperationIr::CtcLossBackward(desc) => self.apply_ctc_loss_backward(handles, desc),
        }
    }
}
