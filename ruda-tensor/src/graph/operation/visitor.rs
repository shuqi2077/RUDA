use super::*;
use alloc::{boxed::Box, vec::Vec};

/// Mutable traversal of tensor metadata, scalar operands and slice ranges.
pub trait IrVisitorMut {
    /// Visit one tensor occurrence, retaining its operation-specific role and status.
    fn visit_tensor_mut(&mut self, _tensor: &mut TensorIr) {}
    /// Visit one scalar operand.
    fn visit_scalar_mut(&mut self, _scalar: &mut ScalarIr) {}
    /// Visit one slice range.
    fn visit_range_mut(&mut self, _range: &mut Slice) {}
}

trait VisitFields {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut);
}
impl VisitFields for TensorIr {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) { visitor.visit_tensor_mut(self); }
}
impl VisitFields for ScalarIr {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) { visitor.visit_scalar_mut(self); }
}
impl VisitFields for Slice {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) { visitor.visit_range_mut(self); }
}
impl<T: VisitFields> VisitFields for Vec<T> {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) {
        for value in self { value.visit_fields(visitor); }
    }
}
impl<T: VisitFields> VisitFields for Option<T> {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) {
        if let Some(value) = self { value.visit_fields(visitor); }
    }
}
impl<T: VisitFields> VisitFields for Box<T> {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) { self.as_mut().visit_fields(visitor); }
}
macro_rules! visit_fields {
    ($($kind:ty => [$($field:ident),*];)*) => {$(
        impl VisitFields for $kind {
            fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) {
                $(self.$field.visit_fields(visitor);)*
            }
        }
    )*};
}
visit_fields! {
    Conv1dOpIr => [x, weight, bias, out];
    Conv1dXBackwardOpIr => [x, weight, output_grad, out];
    Conv1dWeightBackwardOpIr => [x, weight, output_grad, out];
    Conv1dBiasBackwardOpIr => [x, bias, output_grad, out];
    Conv2dOpIr => [x, weight, bias, out];
    Conv2dXBackwardOpIr => [x, weight, output_grad, out];
    Conv2dWeightBackwardOpIr => [x, weight, output_grad, out];
    Conv2dBiasBackwardOpIr => [x, bias, output_grad, out];
    DeformConv2dOpIr => [x, offset, weight, mask, bias, out];
    DeformConv2dBackwardOpIr => [x, offset, weight, mask, bias, out_grad, input_grad, offset_grad, weight_grad, mask_grad, bias_grad];
    Conv3dOpIr => [x, weight, bias, out];
    Conv3dXBackwardOpIr => [x, weight, output_grad, out];
    Conv3dWeightBackwardOpIr => [x, weight, output_grad, out];
    Conv3dBiasBackwardOpIr => [x, bias, output_grad, out];
    ConvTranspose1dOpIr => [x, weight, bias, out];
    ConvTranspose2dOpIr => [x, weight, bias, out];
    ConvTranspose3dOpIr => [x, weight, bias, out];
    RfftOpIr => [signal, out_re, out_im];
    IRfftOpIr => [input_re, input_im, out_signal];
    MatmulOpIr => [lhs, rhs, out];
    CrossOpIr => [lhs, rhs, out];
    LinearOpIr => [x, weight, bias, out];
    LinearXBackwardOpIr => [weight, output_grad, out];
    LinearWeightBackwardOpIr => [x, output_grad, out];
    LinearBiasBackwardOpIr => [output_grad, out];
    ExponentialReluOpIr => [x, alpha, out];
    ExponentialReluBackwardOpIr => [x, grad, alpha, out];
    LeakyReluOpIr => [x, negative_slope, out];
    LeakyReluBackwardOpIr => [x, grad, negative_slope, out];
    PreluOpIr => [x, alpha, out];
    PreluBackwardSelectOpIr => [x, alpha, grad, input_grad, weight_grad];
    GroupNormOpIr => [x, gamma, beta, epsilon, out, mean, rstd];
    GroupNormBackwardSelectOpIr => [x, gamma, grad, mean, rstd, input_grad, weight_grad, bias_grad];
    GeluOpIr => [x, out];
    GeluBackwardOpIr => [x, grad, out];
    SiluBackwardOpIr => [x, grad, out];
    SoftmaxOpIr => [x, out, working];
    SoftmaxBackwardOpIr => [working, grad, out];
    RmsNormBackwardSelectOpIr => [x, gamma, grad, rstd, input_grad, weight_grad];
    LayerNormBackwardSelectOpIr => [x, gamma, grad, mean, rstd, input_grad, weight_grad, bias_grad];
    RmsNormOpIr => [x, gamma, epsilon, out, rstd];
    RmsNormBackwardOpIr => [x, gamma, grad, rstd, input_grad, weight_grad];
    LayerNormOpIr => [x, gamma, beta, epsilon, out, mean, rstd];
    LayerNormBackwardOpIr => [x, gamma, grad, mean, rstd, input_grad, weight_grad, bias_grad];
    AvgPool1dOpIr => [x, out];
    AvgPool2dOpIr => [x, out];
    AvgPool1dBackwardOpIr => [x, grad, out];
    AvgPool2dBackwardOpIr => [x, grad, out];
    AvgPool3dOpIr => [x, out];
    AvgPool3dBackwardOpIr => [x, grad, out];
    AdaptiveAvgPool1dOpIr => [x, out];
    AdaptiveAvgPool2dOpIr => [x, out];
    AdaptiveAvgPool1dBackwardOpIr => [x, grad, out];
    AdaptiveAvgPool2dBackwardOpIr => [x, grad, out];
    AdaptiveAvgPool3dOpIr => [x, out];
    AdaptiveAvgPool3dBackwardOpIr => [x, grad, out];
    MaxPool1dOpIr => [x, out];
    MaxPool1dWithIndicesOpIr => [x, out, out_indices];
    MaxPool1dWithIndicesBackwardOpIr => [x, grad, indices, out];
    MaxPool2dOpIr => [x, out];
    MaxPool2dWithIndicesOpIr => [x, out, out_indices];
    MaxPool2dWithIndicesBackwardOpIr => [x, grad, indices, out];
    MaxPool3dOpIr => [x, out];
    MaxPool3dWithIndicesOpIr => [x, out, out_indices];
    MaxPool3dWithIndicesBackwardOpIr => [x, grad, indices, out];
    InterpolateOpIr => [x, out];
    Interpolate1dOpIr => [x, out];
    Interpolate1dBackwardOpIr => [x, grad, out];
    Interpolate3dOpIr => [x, out];
    Interpolate3dBackwardOpIr => [x, grad, out];
    AttentionOptionsIr => [scale, softcap];
    AttentionOpIr => [query, key, value, mask, attn_bias, options, out];
    CtcLossOpIr => [log_probs, targets, input_lengths, target_lengths, out];
    CtcLossBackwardOpIr => [log_probs, targets, input_lengths, target_lengths, grad_loss, out];
    InterpolateBackwardOpIr => [x, grad, out];
    GridSample2dOpIr => [tensor, grid, out];
    QuantizationParametersIr => [scales];
    QuantizeOpIr => [tensor, qparams, out];
    DequantizeOpIr => [input, out];
    CustomOpIr => [inputs, outputs];
    SwapDimsOpIr => [input, out];
    PermuteOpIr => [input, out];
    ShapeOpIr => [input, out];
    UnfoldOpIr => [input, out];
    FlipOpIr => [input, out];
    RandomOpIr => [out];
    CreationOpIr => [out];
    FullOpIr => [out, value];
    InitOperationIr => [out];
    BinaryOpIr => [lhs, rhs, out];
    UnaryOpIr => [input, out];
    ScalarOpIr => [lhs, rhs, out];
    ReduceOpIr => [input, out];
    ReduceDimOpIr => [input, out];
    CastOpIr => [input, out];
    DimOpIr => [input, out];
    GatherOpIr => [tensor, indices, out];
    ScatterOpIr => [tensor, indices, value, out];
    ScatterNdOpIr => [data, indices, values, out];
    GatherNdOpIr => [data, indices, out];
    SelectOpIr => [tensor, indices, out];
    SelectAssignOpIr => [tensor, indices, value, out];
    SliceOpIr => [tensor, ranges, out];
    SliceAssignOpIr => [tensor, ranges, value, out];
    MaskWhereOpIr => [tensor, mask, value, out];
    MaskFillOpIr => [tensor, mask, value, out];
    ClampOpIr => [tensor, min, max, out];
    RepeatDimOpIr => [tensor, out];
    CatOpIr => [tensors, out];
    ReduceDimWithIndicesOpIr => [tensor, out, out_indices];
    EmbeddingOpIr => [weights, indices, out];
    EmbeddingBackwardOpIr => [weights, out_grad, indices, out];
}
#[cfg(feature = "graph-distributed")]
visit_fields! { AllReduceOpIr => [tensor, out]; }
impl VisitFields for OperationIr {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) {
        match self {
            Self::BaseFloat(value0) => { value0.visit_fields(visitor); }
            Self::BaseInt(value0) => { value0.visit_fields(visitor); }
            Self::BaseBool(value0) => { value0.visit_fields(visitor); }
            Self::NumericFloat(_, value1) => { value1.visit_fields(visitor); }
            Self::NumericInt(_, value1) => { value1.visit_fields(visitor); }
            Self::Bool(value0) => { value0.visit_fields(visitor); }
            Self::Int(value0) => { value0.visit_fields(visitor); }
            Self::Float(_, value1) => { value1.visit_fields(visitor); }
            Self::Module(value0) => { value0.visit_fields(visitor); }
            Self::Init(value0) => { value0.visit_fields(visitor); }
            Self::Custom(value0) => { value0.visit_fields(visitor); }
            Self::Drop(value0) => { value0.visit_fields(visitor); }
            #[cfg(feature = "graph-distributed")]
            Self::Distributed(value0) => { value0.visit_fields(visitor); }
        }
    }
}
impl VisitFields for FloatOperationIr {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) {
        match self {
            Self::Exp(value0) => { value0.visit_fields(visitor); }
            Self::Log(value0) => { value0.visit_fields(visitor); }
            Self::Log1p(value0) => { value0.visit_fields(visitor); }
            Self::Erf(value0) => { value0.visit_fields(visitor); }
            Self::PowfScalar(value0) => { value0.visit_fields(visitor); }
            Self::Sqrt(value0) => { value0.visit_fields(visitor); }
            Self::Cos(value0) => { value0.visit_fields(visitor); }
            Self::Cosh(value0) => { value0.visit_fields(visitor); }
            Self::Sin(value0) => { value0.visit_fields(visitor); }
            Self::Sinh(value0) => { value0.visit_fields(visitor); }
            Self::Tan(value0) => { value0.visit_fields(visitor); }
            Self::Tanh(value0) => { value0.visit_fields(visitor); }
            Self::ArcCos(value0) => { value0.visit_fields(visitor); }
            Self::ArcCosh(value0) => { value0.visit_fields(visitor); }
            Self::ArcSin(value0) => { value0.visit_fields(visitor); }
            Self::ArcSinh(value0) => { value0.visit_fields(visitor); }
            Self::ArcTan(value0) => { value0.visit_fields(visitor); }
            Self::ArcTanh(value0) => { value0.visit_fields(visitor); }
            Self::ArcTan2(value0) => { value0.visit_fields(visitor); }
            Self::Round(value0) => { value0.visit_fields(visitor); }
            Self::Floor(value0) => { value0.visit_fields(visitor); }
            Self::Ceil(value0) => { value0.visit_fields(visitor); }
            Self::Trunc(value0) => { value0.visit_fields(visitor); }
            Self::IntoInt(value0) => { value0.visit_fields(visitor); }
            Self::Matmul(value0) => { value0.visit_fields(visitor); }
            Self::Cross(value0) => { value0.visit_fields(visitor); }
            Self::Random(value0) => { value0.visit_fields(visitor); }
            Self::Recip(value0) => { value0.visit_fields(visitor); }
            Self::IsNan(value0) => { value0.visit_fields(visitor); }
            Self::IsInf(value0) => { value0.visit_fields(visitor); }
            Self::Quantize(value0) => { value0.visit_fields(visitor); }
            Self::Dequantize(value0) => { value0.visit_fields(visitor); }
            Self::GridSample2d(value0) => { value0.visit_fields(visitor); }
            Self::Powf(value0) => { value0.visit_fields(visitor); }
            Self::Rsqrt(value0) => { value0.visit_fields(visitor); }
            Self::Silu(value0) => { value0.visit_fields(visitor); }
            Self::QuantizeDynamic(value0) => { value0.visit_fields(visitor); }
        }
    }
}
impl VisitFields for ModuleOperationIr {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) {
        match self {
            Self::Embedding(value0) => { value0.visit_fields(visitor); }
            Self::EmbeddingBackward(value0) => { value0.visit_fields(visitor); }
            Self::Linear(value0) => { value0.visit_fields(visitor); }
            Self::LinearXBackward(value0) => { value0.visit_fields(visitor); }
            Self::LinearWeightBackward(value0) => { value0.visit_fields(visitor); }
            Self::LinearBiasBackward(value0) => { value0.visit_fields(visitor); }
            Self::Conv1d(value0) => { value0.visit_fields(visitor); }
            Self::Conv1dXBackward(value0) => { value0.visit_fields(visitor); }
            Self::Conv1dWeightBackward(value0) => { value0.visit_fields(visitor); }
            Self::Conv1dBiasBackward(value0) => { value0.visit_fields(visitor); }
            Self::Conv2d(value0) => { value0.visit_fields(visitor); }
            Self::Conv2dXBackward(value0) => { value0.visit_fields(visitor); }
            Self::Conv2dWeightBackward(value0) => { value0.visit_fields(visitor); }
            Self::Conv2dBiasBackward(value0) => { value0.visit_fields(visitor); }
            Self::Conv3d(value0) => { value0.visit_fields(visitor); }
            Self::Conv3dXBackward(value0) => { value0.visit_fields(visitor); }
            Self::Conv3dWeightBackward(value0) => { value0.visit_fields(visitor); }
            Self::Conv3dBiasBackward(value0) => { value0.visit_fields(visitor); }
            Self::DeformableConv2d(value0) => { value0.visit_fields(visitor); }
            Self::DeformableConv2dBackward(value0) => { value0.visit_fields(visitor); }
            Self::ConvTranspose1d(value0) => { value0.visit_fields(visitor); }
            Self::ConvTranspose2d(value0) => { value0.visit_fields(visitor); }
            Self::ConvTranspose3d(value0) => { value0.visit_fields(visitor); }
            Self::AvgPool1d(value0) => { value0.visit_fields(visitor); }
            Self::AvgPool2d(value0) => { value0.visit_fields(visitor); }
            Self::AvgPool1dBackward(value0) => { value0.visit_fields(visitor); }
            Self::AvgPool2dBackward(value0) => { value0.visit_fields(visitor); }
            Self::AdaptiveAvgPool1d(value0) => { value0.visit_fields(visitor); }
            Self::AdaptiveAvgPool2d(value0) => { value0.visit_fields(visitor); }
            Self::AdaptiveAvgPool1dBackward(value0) => { value0.visit_fields(visitor); }
            Self::AdaptiveAvgPool2dBackward(value0) => { value0.visit_fields(visitor); }
            Self::MaxPool1d(value0) => { value0.visit_fields(visitor); }
            Self::MaxPool1dWithIndices(value0) => { value0.visit_fields(visitor); }
            Self::MaxPool1dWithIndicesBackward(value0) => { value0.visit_fields(visitor); }
            Self::MaxPool2d(value0) => { value0.visit_fields(visitor); }
            Self::MaxPool2dWithIndices(value0) => { value0.visit_fields(visitor); }
            Self::MaxPool2dWithIndicesBackward(value0) => { value0.visit_fields(visitor); }
            Self::Interpolate(value0) => { value0.visit_fields(visitor); }
            Self::InterpolateBackward(value0) => { value0.visit_fields(visitor); }
            Self::Rfft(value0) => { value0.visit_fields(visitor); }
            Self::IRfft(value0) => { value0.visit_fields(visitor); }
            Self::Attention(value0) => { value0.visit_fields(visitor); }
            Self::CtcLoss(value0) => { value0.visit_fields(visitor); }
            Self::CtcLossBackward(value0) => { value0.visit_fields(visitor); }
            Self::AdaptiveAvgPool3d(value0) => { value0.visit_fields(visitor); }
            Self::AdaptiveAvgPool3dBackward(value0) => { value0.visit_fields(visitor); }
            Self::AvgPool3d(value0) => { value0.visit_fields(visitor); }
            Self::AvgPool3dBackward(value0) => { value0.visit_fields(visitor); }
            Self::MaxPool3d(value0) => { value0.visit_fields(visitor); }
            Self::MaxPool3dWithIndices(value0) => { value0.visit_fields(visitor); }
            Self::MaxPool3dWithIndicesBackward(value0) => { value0.visit_fields(visitor); }
            Self::Interpolate1d(value0) => { value0.visit_fields(visitor); }
            Self::Interpolate1dBackward(value0) => { value0.visit_fields(visitor); }
            Self::Interpolate3d(value0) => { value0.visit_fields(visitor); }
            Self::Interpolate3dBackward(value0) => { value0.visit_fields(visitor); }
            Self::LayerNorm(value0) => { value0.visit_fields(visitor); }
            Self::LayerNormBackward(value0) => { value0.visit_fields(visitor); }
            Self::RmsNorm(value0) => { value0.visit_fields(visitor); }
            Self::RmsNormBackward(value0) => { value0.visit_fields(visitor); }
            Self::RmsNormBackwardSelect(value0) => { value0.visit_fields(visitor); }
            Self::LayerNormBackwardSelect(value0) => { value0.visit_fields(visitor); }
            Self::Softmax(value0) => { value0.visit_fields(visitor); }
            Self::SoftmaxBackward(value0) => { value0.visit_fields(visitor); }
            Self::SiluNative(value0) => { value0.visit_fields(visitor); }
            Self::SiluNativeBackward(value0) => { value0.visit_fields(visitor); }
            Self::GeluNative(value0) => { value0.visit_fields(visitor); }
            Self::GeluNativeBackward(value0) => { value0.visit_fields(visitor); }
            Self::GroupNorm(value0) => { value0.visit_fields(visitor); }
            Self::GroupNormBackwardSelect(value0) => { value0.visit_fields(visitor); }
            Self::PreluNative(value0) => { value0.visit_fields(visitor); }
            Self::PreluNativeBackwardSelect(value0) => { value0.visit_fields(visitor); }
            Self::LeakyReluNative(value0) => { value0.visit_fields(visitor); }
            Self::LeakyReluNativeBackward(value0) => { value0.visit_fields(visitor); }
            Self::ExponentialReluNative(value0) => { value0.visit_fields(visitor); }
            Self::ExponentialReluNativeBackward(value0) => { value0.visit_fields(visitor); }
        }
    }
}
impl VisitFields for BaseOperationIr {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) {
        match self {
            Self::Reshape(value0) => { value0.visit_fields(visitor); }
            Self::SwapDims(value0) => { value0.visit_fields(visitor); }
            Self::Permute(value0) => { value0.visit_fields(visitor); }
            Self::Flip(value0) => { value0.visit_fields(visitor); }
            Self::Expand(value0) => { value0.visit_fields(visitor); }
            Self::Unfold(value0) => { value0.visit_fields(visitor); }
            Self::Slice(value0) => { value0.visit_fields(visitor); }
            Self::SliceAssign(value0) => { value0.visit_fields(visitor); }
            Self::Select(value0) => { value0.visit_fields(visitor); }
            Self::SelectAssign(value0) => { value0.visit_fields(visitor); }
            Self::MaskWhere(value0) => { value0.visit_fields(visitor); }
            Self::MaskFill(value0) => { value0.visit_fields(visitor); }
            Self::Gather(value0) => { value0.visit_fields(visitor); }
            Self::Scatter(value0) => { value0.visit_fields(visitor); }
            Self::ScatterNd(value0) => { value0.visit_fields(visitor); }
            Self::GatherNd(value0) => { value0.visit_fields(visitor); }
            Self::Equal(value0) => { value0.visit_fields(visitor); }
            Self::EqualElem(value0) => { value0.visit_fields(visitor); }
            Self::RepeatDim(value0) => { value0.visit_fields(visitor); }
            Self::Cat(value0) => { value0.visit_fields(visitor); }
            Self::Cast(value0) => { value0.visit_fields(visitor); }
            Self::Empty(value0) => { value0.visit_fields(visitor); }
            Self::Ones(value0) => { value0.visit_fields(visitor); }
            Self::Zeros(value0) => { value0.visit_fields(visitor); }
        }
    }
}
impl VisitFields for NumericOperationIr {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) {
        match self {
            Self::Add(value0) => { value0.visit_fields(visitor); }
            Self::AddScalar(value0) => { value0.visit_fields(visitor); }
            Self::Sub(value0) => { value0.visit_fields(visitor); }
            Self::SubScalar(value0) => { value0.visit_fields(visitor); }
            Self::Div(value0) => { value0.visit_fields(visitor); }
            Self::DivScalar(value0) => { value0.visit_fields(visitor); }
            Self::Rem(value0) => { value0.visit_fields(visitor); }
            Self::RemScalar(value0) => { value0.visit_fields(visitor); }
            Self::Mul(value0) => { value0.visit_fields(visitor); }
            Self::MulScalar(value0) => { value0.visit_fields(visitor); }
            Self::Abs(value0) => { value0.visit_fields(visitor); }
            Self::Full(value0) => { value0.visit_fields(visitor); }
            Self::MeanDim(value0) => { value0.visit_fields(visitor); }
            Self::Mean(value0) => { value0.visit_fields(visitor); }
            Self::Sum(value0) => { value0.visit_fields(visitor); }
            Self::SumDim(value0) => { value0.visit_fields(visitor); }
            Self::Prod(value0) => { value0.visit_fields(visitor); }
            Self::ProdDim(value0) => { value0.visit_fields(visitor); }
            Self::Greater(value0) => { value0.visit_fields(visitor); }
            Self::GreaterElem(value0) => { value0.visit_fields(visitor); }
            Self::GreaterEqual(value0) => { value0.visit_fields(visitor); }
            Self::GreaterEqualElem(value0) => { value0.visit_fields(visitor); }
            Self::Lower(value0) => { value0.visit_fields(visitor); }
            Self::LowerElem(value0) => { value0.visit_fields(visitor); }
            Self::LowerEqual(value0) => { value0.visit_fields(visitor); }
            Self::LowerEqualElem(value0) => { value0.visit_fields(visitor); }
            Self::ArgMax(value0) => { value0.visit_fields(visitor); }
            Self::ArgTopK(value0) => { value0.visit_fields(visitor); }
            Self::TopK(value0) => { value0.visit_fields(visitor); }
            Self::ArgMin(value0) => { value0.visit_fields(visitor); }
            Self::Max(value0) => { value0.visit_fields(visitor); }
            Self::MaxDimWithIndices(value0) => { value0.visit_fields(visitor); }
            Self::MinDimWithIndices(value0) => { value0.visit_fields(visitor); }
            Self::Min(value0) => { value0.visit_fields(visitor); }
            Self::MaxDim(value0) => { value0.visit_fields(visitor); }
            Self::MinDim(value0) => { value0.visit_fields(visitor); }
            Self::MaxAbs(value0) => { value0.visit_fields(visitor); }
            Self::MaxAbsDim(value0) => { value0.visit_fields(visitor); }
            Self::Clamp(value0) => { value0.visit_fields(visitor); }
            Self::IntRandom(value0) => { value0.visit_fields(visitor); }
            Self::Powi(value0) => { value0.visit_fields(visitor); }
            Self::PowiScalar(value0) => { value0.visit_fields(visitor); }
            Self::CumSum(value0) => { value0.visit_fields(visitor); }
            Self::CumProd(value0) => { value0.visit_fields(visitor); }
            Self::CumMin(value0) => { value0.visit_fields(visitor); }
            Self::CumMax(value0) => { value0.visit_fields(visitor); }
        }
    }
}
impl VisitFields for IntOperationIr {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) {
        match self {
            Self::IntoFloat(value0) => { value0.visit_fields(visitor); }
            Self::BitwiseAnd(value0) => { value0.visit_fields(visitor); }
            Self::BitwiseAndScalar(value0) => { value0.visit_fields(visitor); }
            Self::BitwiseOr(value0) => { value0.visit_fields(visitor); }
            Self::BitwiseOrScalar(value0) => { value0.visit_fields(visitor); }
            Self::BitwiseXor(value0) => { value0.visit_fields(visitor); }
            Self::BitwiseXorScalar(value0) => { value0.visit_fields(visitor); }
            Self::BitwiseNot(value0) => { value0.visit_fields(visitor); }
            Self::BitwiseLeftShift(value0) => { value0.visit_fields(visitor); }
            Self::BitwiseLeftShiftScalar(value0) => { value0.visit_fields(visitor); }
            Self::BitwiseRightShift(value0) => { value0.visit_fields(visitor); }
            Self::BitwiseRightShiftScalar(value0) => { value0.visit_fields(visitor); }
            Self::Matmul(value0) => { value0.visit_fields(visitor); }
        }
    }
}
impl VisitFields for BoolOperationIr {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) {
        match self {
            Self::IntoFloat(value0) => { value0.visit_fields(visitor); }
            Self::IntoInt(value0) => { value0.visit_fields(visitor); }
            Self::Not(value0) => { value0.visit_fields(visitor); }
            Self::And(value0) => { value0.visit_fields(visitor); }
            Self::Or(value0) => { value0.visit_fields(visitor); }
        }
    }
}
#[cfg(feature = "graph-distributed")]
impl VisitFields for DistributedOperationIr {
    fn visit_fields(&mut self, visitor: &mut impl IrVisitorMut) {
        match self {
            Self::AllReduce(value0) => { value0.visit_fields(visitor); }
        }
    }
}

impl OperationIr {
    /// Mutate every tensor, scalar and slice occurrence in declaration order.
    pub fn visit_mut(&mut self, visitor: &mut impl IrVisitorMut) { self.visit_fields(visitor); }
}
