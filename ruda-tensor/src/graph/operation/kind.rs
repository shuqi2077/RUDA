use super::*;


/// Describe all tensor operations possible.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum OperationIr {
    /// Basic operation on a float tensor.
    BaseFloat(BaseOperationIr),
    /// Basic operation on an int tensor.
    BaseInt(BaseOperationIr),
    /// Basic operation on a bool tensor.
    BaseBool(BaseOperationIr),
    /// Numeric operation on a float tensor.
    NumericFloat(DType, NumericOperationIr),
    /// Numeric operation on an int tensor.
    NumericInt(DType, NumericOperationIr),
    /// Operation specific to a bool tensor.
    Bool(BoolOperationIr),
    /// Operation specific to an int tensor.
    Int(IntOperationIr),
    /// Operation specific to a float tensor.
    Float(DType, FloatOperationIr),
    /// Module operation.
    Module(ModuleOperationIr),
    /// Initialize operation.
    Init(InitOperationIr),
    /// A custom operation.
    Custom(CustomOpIr),
    /// A tensor is dropped.
    Drop(TensorIr),
    #[cfg(feature = "graph-distributed")]
    /// Operation specific to a distributed tensor.
    Distributed(DistributedOperationIr),
}

/// Operation intermediate representation specific to a float tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum FloatOperationIr {
    /// Operation corresponding to [exp](crate::ops::FloatTensorOps::float_exp).
    Exp(UnaryOpIr),
    /// Operation corresponding to [log](crate::ops::FloatTensorOps::float_log).
    Log(UnaryOpIr),
    /// Operation corresponding to [log1p](crate::ops::FloatTensorOps::float_log1p).
    Log1p(UnaryOpIr),
    /// Operation corresponding to [erf](crate::ops::FloatTensorOps::float_erf).
    Erf(UnaryOpIr),
    /// Operation corresponding to [powf_scalar](crate::ops::FloatTensorOps::float_powf_scalar).
    PowfScalar(ScalarOpIr),
    /// Operation corresponding to [sqrt](crate::ops::FloatTensorOps::float_sqrt).
    Sqrt(UnaryOpIr),
    /// Operation corresponding to [cos](crate::ops::FloatTensorOps::float_cos).
    Cos(UnaryOpIr),
    /// Operation corresponding to [cosh](crate::ops::FloatTensorOps::float_cosh).
    Cosh(UnaryOpIr),
    /// Operation corresponding to [sin](crate::ops::FloatTensorOps::float_sin).
    Sin(UnaryOpIr),
    /// Operation corresponding to [sin](crate::ops::FloatTensorOps::float_sinh).
    Sinh(UnaryOpIr),
    /// Operation corresponding to [tan](crate::ops::FloatTensorOps::float_tan).
    Tan(UnaryOpIr),
    /// Operation corresponding to [tanh](crate::ops::FloatTensorOps::float_tanh).
    Tanh(UnaryOpIr),
    /// Operation corresponding to [acos](crate::ops::FloatTensorOps::float_acos).
    ArcCos(UnaryOpIr),
    /// Operation corresponding to [acosh](crate::ops::FloatTensorOps::float_acosh).
    ArcCosh(UnaryOpIr),
    /// Operation corresponding to [asin](crate::ops::FloatTensorOps::float_asin).
    ArcSin(UnaryOpIr),
    /// Operation corresponding to [asinh](crate::ops::FloatTensorOps::float_asinh).
    ArcSinh(UnaryOpIr),
    /// Operation corresponding to [atan](crate::ops::FloatTensorOps::float_atan).
    ArcTan(UnaryOpIr),
    /// Operation corresponding to [atanh](crate::ops::FloatTensorOps::float_atanh).
    ArcTanh(UnaryOpIr),
    /// Operation corresponding to [atan2](crate::ops::FloatTensorOps::float_atan2).
    ArcTan2(BinaryOpIr),
    /// Operation corresponding to [round](crate::ops::FloatTensorOps::float_round).
    Round(UnaryOpIr),
    /// Operation corresponding to [floor](crate::ops::FloatTensorOps::float_floor).
    Floor(UnaryOpIr),
    /// Operation corresponding to [ceil](crate::ops::FloatTensorOps::float_ceil).
    Ceil(UnaryOpIr),
    /// Operation corresponding to [trunc](crate::ops::FloatTensorOps::float_trunc).
    Trunc(UnaryOpIr),
    /// Operation corresponding to [into_int](crate::ops::FloatTensorOps::float_into_int).
    IntoInt(CastOpIr),
    /// Operation corresponding to [matmul](crate::ops::FloatTensorOps::float_matmul).
    Matmul(MatmulOpIr),
    /// Operation corresponding to [cross](crate::ops::FloatTensorOps::float_cross).
    Cross(CrossOpIr),
    /// Operation corresponding to [random](crate::ops::FloatTensorOps::float_random).
    Random(RandomOpIr),
    /// Operation corresponding to [recip](crate::ops::FloatTensorOps::float_recip).
    Recip(UnaryOpIr),
    /// Operation corresponding to [is_nan](crate::ops::FloatTensorOps::float_is_nan).
    IsNan(UnaryOpIr),
    /// Operation corresponding to [is_nan](crate::ops::FloatTensorOps::float_is_inf).
    IsInf(UnaryOpIr),
    /// Operation corresponding to [quantize](crate::ops::QTensorOps::quantize).
    Quantize(QuantizeOpIr),
    /// Operation corresponding to [dequantize](crate::ops::QTensorOps::dequantize).
    Dequantize(DequantizeOpIr),
    /// Operation corresponding to [grid_sample_2d](crate::ops::FloatTensorOps::float_grid_sample_2d).
    GridSample2d(GridSample2dOpIr),
    /// Operation corresponding to [powf](crate::ops::FloatTensorOps::float_powi).
    Powf(BinaryOpIr),
    /// Operation corresponding to [rsqrt](crate::ops::FloatTensorOps::float_rsqrt).
    Rsqrt(UnaryOpIr),
    /// Operation corresponding to [silu](crate::ops::ActivationOps::silu).
    Silu(UnaryOpIr),
    /// Operation corresponding to [quantize_dynamic](crate::ops::QTensorOps::quantize_dynamic).
    QuantizeDynamic(CastOpIr),
}

/// Operation intermediate representation specific to module.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum ModuleOperationIr {
    /// Operation corresponding to [embedding](crate::ops::ModuleOps::embedding).
    Embedding(EmbeddingOpIr),
    /// Operation corresponding to [embedding_backward](crate::ops::ModuleOps::embedding_backward).
    EmbeddingBackward(EmbeddingBackwardOpIr),
    /// Operation corresponding to [linear](crate::ops::ModuleOps::linear).
    Linear(LinearOpIr),
    /// Operation corresponding to [linear_x_backward](crate::ops::ModuleOps::linear_x_backward).
    LinearXBackward(LinearXBackwardOpIr),
    /// Operation corresponding to [linear_weight_backward](crate::ops::ModuleOps::linear_weight_backward).
    LinearWeightBackward(LinearWeightBackwardOpIr),
    /// Operation corresponding to [linear_bias_backward](crate::ops::ModuleOps::linear_bias_backward).
    LinearBiasBackward(LinearBiasBackwardOpIr),
    /// Operation corresponding to [conv1d](crate::ops::ModuleOps::conv1d).
    Conv1d(Conv1dOpIr),
    /// Operation corresponding to [conv1d_x_backward](crate::ops::ModuleOps::conv1d_x_backward).
    Conv1dXBackward(Conv1dXBackwardOpIr),
    /// Operation corresponding to [conv1d_weight_backward](crate::ops::ModuleOps::conv1d_weight_backward).
    Conv1dWeightBackward(Conv1dWeightBackwardOpIr),
    /// Operation corresponding to [conv1d_bias_backward](crate::ops::ModuleOps::conv1d_bias_backward).
    Conv1dBiasBackward(Conv1dBiasBackwardOpIr),
    /// Operation corresponding to [conv2d](crate::ops::ModuleOps::conv2d).
    Conv2d(Conv2dOpIr),
    /// Operation corresponding to [conv2d_x_backward](crate::ops::ModuleOps::conv2d_x_backward).
    Conv2dXBackward(Conv2dXBackwardOpIr),
    /// Operation corresponding to [conv2d_weight_backward](crate::ops::ModuleOps::conv2d_weight_backward).
    Conv2dWeightBackward(Conv2dWeightBackwardOpIr),
    /// Operation corresponding to [conv2d_bias_backward](crate::ops::ModuleOps::conv2d_bias_backward).
    Conv2dBiasBackward(Conv2dBiasBackwardOpIr),
    /// Operation corresponding to [conv3d](crate::ops::ModuleOps::conv3d).
    Conv3d(Conv3dOpIr),
    /// Operation corresponding to [conv3d_x_backward](crate::ops::ModuleOps::conv3d_x_backward).
    Conv3dXBackward(Conv3dXBackwardOpIr),
    /// Operation corresponding to [conv3d_weight_backward](crate::ops::ModuleOps::conv3d_weight_backward).
    Conv3dWeightBackward(Conv3dWeightBackwardOpIr),
    /// Operation corresponding to [conv3d_bias_backward](crate::ops::ModuleOps::conv3d_bias_backward).
    Conv3dBiasBackward(Conv3dBiasBackwardOpIr),
    /// Operation corresponding to [deform_conv2d](crate::ops::ModuleOps::deform_conv2d)
    DeformableConv2d(Box<DeformConv2dOpIr>),
    /// Operation corresponding to [deform_conv2d_backward](crate::ops::ModuleOps::deform_conv2d_backward)
    DeformableConv2dBackward(Box<DeformConv2dBackwardOpIr>),
    /// Operation corresponding to [conv transpose 1d](crate::ops::ModuleOps::conv_transpose1d).
    ConvTranspose1d(ConvTranspose1dOpIr),
    /// Operation corresponding to [conv transpose 2d](crate::ops::ModuleOps::conv_transpose2d).
    ConvTranspose2d(ConvTranspose2dOpIr),
    /// Operation corresponding to [conv transpose 3d](crate::ops::ModuleOps::conv_transpose3d).
    ConvTranspose3d(ConvTranspose3dOpIr),
    /// Operation corresponding to [avg pool 1d](crate::ops::ModuleOps::avg_pool1d).
    AvgPool1d(AvgPool1dOpIr),
    /// Operation corresponding to [avg pool 2d](crate::ops::ModuleOps::avg_pool2d).
    AvgPool2d(AvgPool2dOpIr),
    /// Operation corresponding to
    /// [avg pool 1d backward](crate::ops::ModuleOps::avg_pool1d_backward).
    AvgPool1dBackward(AvgPool1dBackwardOpIr),
    /// Operation corresponding to
    /// [avg pool 2d backward](crate::ops::ModuleOps::avg_pool2d_backward).
    AvgPool2dBackward(AvgPool2dBackwardOpIr),
    /// Operation corresponding to
    /// [adaptive avg pool 1d](crate::ops::ModuleOps::adaptive_avg_pool1d).
    AdaptiveAvgPool1d(AdaptiveAvgPool1dOpIr),
    /// Operation corresponding to
    /// [adaptive avg pool 2d](crate::ops::ModuleOps::adaptive_avg_pool2d).
    AdaptiveAvgPool2d(AdaptiveAvgPool2dOpIr),
    /// Operation corresponding to
    /// [adaptive avg pool 1d backward](crate::ops::ModuleOps::adaptive_avg_pool1d_backward).
    AdaptiveAvgPool1dBackward(AdaptiveAvgPool1dBackwardOpIr),
    /// Operation corresponding to
    /// [adaptive avg pool 2d backward](crate::ops::ModuleOps::adaptive_avg_pool2d_backward).
    AdaptiveAvgPool2dBackward(AdaptiveAvgPool2dBackwardOpIr),
    /// Operation corresponding to
    /// [max pool 1d](crate::ops::ModuleOps::max_pool1d).
    MaxPool1d(MaxPool1dOpIr),
    /// Operation corresponding to
    /// [max pool 1d with indices](crate::ops::ModuleOps::max_pool1d_with_indices).
    MaxPool1dWithIndices(MaxPool1dWithIndicesOpIr),
    /// Operation corresponding to
    /// [max pool 1d with indices backward](crate::ops::ModuleOps::max_pool1d_with_indices_backward).
    MaxPool1dWithIndicesBackward(MaxPool1dWithIndicesBackwardOpIr),
    /// Operation corresponding to
    /// [max pool 2d](crate::ops::ModuleOps::max_pool1d).
    MaxPool2d(MaxPool2dOpIr),
    /// Operation corresponding to
    /// [max pool 2d with indices](crate::ops::ModuleOps::max_pool2d_with_indices).
    MaxPool2dWithIndices(MaxPool2dWithIndicesOpIr),
    /// Operation corresponding to
    /// [max pool 2d with indices backward](crate::ops::ModuleOps::max_pool2d_with_indices_backward).
    MaxPool2dWithIndicesBackward(MaxPool2dWithIndicesBackwardOpIr),
    /// Operation corresponding to [interpolate](crate::ops::ModuleOps::interpolate).
    Interpolate(InterpolateOpIr),
    /// Operation corresponding to [interpolate backward](crate::ops::ModuleOps::interpolate_backward).
    InterpolateBackward(InterpolateBackwardOpIr),
    /// Operation corresponding to [rfft](crate::ops::ModuleOps::rfft)
    Rfft(RfftOpIr),
    /// Operation corresponding to [irfft](crate::ops::ModuleOps::irfft)
    IRfft(IRfftOpIr),
    /// Operation corresponding to [attention](crate::ops::ModuleOps::attention).
    Attention(AttentionOpIr),
    /// Operation corresponding to [ctc_loss](crate::ops::ModuleOps::ctc_loss).
    CtcLoss(CtcLossOpIr),
    /// Operation corresponding to
    /// [ctc_loss_backward](crate::ops::ModuleOps::ctc_loss_backward).
    CtcLossBackward(CtcLossBackwardOpIr),
}

/// Basic operations that can be done on any tensor type.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum BaseOperationIr {
    /// Operation corresponding to:
    ///
    /// Float => [reshape](crate::ops::FloatTensorOps::float_reshape).
    /// Int => [reshape](crate::ops::IntTensorOps::int_reshape).
    /// Bool => [reshape](crate::ops::BoolTensorOps::bool_reshape).
    Reshape(ShapeOpIr),

    /// Operation corresponding to:
    ///
    /// Float => [swap_dims](crate::ops::FloatTensorOps::float_swap_dims).
    /// Int => [swap_dims](crate::ops::IntTensorOps::int_swap_dims).
    /// Bool => [swap_dims](crate::ops::BoolTensorOps::bool_swap_dims).
    SwapDims(SwapDimsOpIr),

    /// Operation corresponding to:
    ///
    /// Float => [permute](crate::ops::FloatTensorOps::float_permute).
    /// Int => [permute](crate::ops::IntTensorOps::int_permute).
    /// Bool => [permute](crate::ops::BoolTensorOps::bool_permute).
    Permute(PermuteOpIr),

    /// Operation corresponding to:
    /// Float => [flip](crate::ops::FloatTensorOps::float_flip).
    /// Int => [flip](crate::ops::IntTensorOps::int_flip).
    /// Bool => [flip](crate::ops::BoolTensorOps::bool_flip).
    Flip(FlipOpIr),

    /// Operation corresponding to:
    ///
    /// Float => [expand](crate::ops::FloatTensorOps::float_expand).
    /// Int => [expand](crate::ops::IntTensorOps::int_expand).
    /// Bool => [expand](crate::ops::BoolTensorOps::bool_expand).
    Expand(ShapeOpIr),

    /// Unfold windows along an axis.
    ///
    Unfold(UnfoldOpIr),

    /// Operation corresponding to:
    ///
    /// Float => [slice](crate::ops::FloatTensorOps::float_slice).
    /// Int => [slice](crate::ops::IntTensorOps::int_slice).
    /// Bool => [slice](crate::ops::BoolTensorOps::bool_slice).
    Slice(SliceOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [slice assign](crate::ops::FloatTensorOps::float_slice_assign).
    /// Int => [slice assign](crate::ops::IntTensorOps::int_slice_assign).
    /// Bool => [slice assign](crate::ops::BoolTensorOps::bool_slice_assign).
    SliceAssign(SliceAssignOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [select](crate::ops::FloatTensorOps::float_select).
    /// Int => [select](crate::ops::IntTensorOps::int_select).
    /// Bool => [select](crate::ops::BoolTensorOps::bool_select).
    Select(SelectOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [select assign](crate::ops::FloatTensorOps::float_select_add).
    /// Int => [select assign](crate::ops::IntTensorOps::int_select_add).
    /// Bool => [select assign](crate::ops::BoolTensorOps::bool_select_or).
    SelectAssign(SelectAssignOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [mask where](crate::ops::FloatTensorOps::float_mask_where).
    /// Int => [mask where](crate::ops::IntTensorOps::int_mask_where).
    /// Bool => [mask where](crate::ops::BoolTensorOps::bool_mask_where).
    MaskWhere(MaskWhereOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [mask fill](crate::ops::FloatTensorOps::float_mask_fill).
    /// Int => [mask fill](crate::ops::IntTensorOps::int_mask_fill).
    /// Bool => [mask fill](crate::ops::BoolTensorOps::bool_mask_fill).
    MaskFill(MaskFillOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [gather](crate::ops::FloatTensorOps::float_gather).
    /// Int => [gather](crate::ops::IntTensorOps::int_gather).
    /// Bool => [gather](crate::ops::BoolTensorOps::bool_gather).
    Gather(GatherOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [scatter](crate::ops::FloatTensorOps::float_scatter_add).
    /// Int => [scatter](crate::ops::IntTensorOps::int_scatter_add).
    /// Bool => [scatter](crate::ops::BoolTensorOps::bool_scatter_or).
    Scatter(ScatterOpIr),
    /// Multi-dimensional scatter operation.
    ScatterNd(ScatterNdOpIr),
    /// Multi-dimensional gather operation.
    GatherNd(GatherNdOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [equal](crate::ops::FloatTensorOps::float_equal).
    /// Int => [equal](crate::ops::IntTensorOps::int_equal).
    /// Bool => [equal](crate::ops::BoolTensorOps::bool_equal).
    Equal(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [equal elem](crate::ops::FloatTensorOps::float_equal_elem).
    /// Int => [equal elem](crate::ops::IntTensorOps::int_equal_elem).
    /// Bool => [equal elem](crate::ops::BoolTensorOps::bool_equal_elem).
    EqualElem(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [repeat dim](crate::ops::FloatTensorOps::float_repeat_dim).
    /// Int => [repeat dim](crate::ops::IntTensorOps::int_repeat_dim).
    /// Bool => [repeat dim](crate::ops::BoolTensorOps::bool_repeat_dim).
    RepeatDim(RepeatDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [cat](crate::ops::FloatTensorOps::float_cat).
    /// Int => [cat](crate::ops::IntTensorOps::int_cat).
    /// Bool => [cat](crate::ops::BoolTensorOps::bool_cat).
    Cat(CatOpIr),
    /// Cast operation, no direct operation and should be supported by fusion backend.
    Cast(CastOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [empty](crate::ops::FloatTensorOps::float_empty).
    /// Int => [empty](crate::ops::IntTensorOps::int_empty).
    /// Bool => [empty](crate::ops::BoolTensorOps::bool_empty).
    Empty(CreationOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [ones](crate::ops::FloatTensorOps::float_ones).
    /// Int => [ones](crate::ops::IntTensorOps::int_ones).
    /// Bool => [ones](crate::ops::BoolTensorOps::bool_ones).
    Ones(CreationOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [zeros](crate::ops::FloatTensorOps::float_zeros).
    /// Int => [zeros](crate::ops::IntTensorOps::int_zeros).
    /// Bool => [zeros](crate::ops::BoolTensorOps::bool_zeros).
    Zeros(CreationOpIr),
}

/// Numeric operations on int and float tensors.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum NumericOperationIr {
    /// Operation corresponding to:
    ///
    /// Float => [add](crate::ops::FloatTensorOps::float_add).
    /// Int => [add](crate::ops::IntTensorOps::int_add).
    Add(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [add scalar](crate::ops::FloatTensorOps::float_add_scalar).
    /// Int => [add scalar](crate::ops::IntTensorOps::int_add_scalar).
    AddScalar(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [sub](crate::ops::FloatTensorOps::float_sub).
    /// Int => [sub](crate::ops::IntTensorOps::int_sub).
    Sub(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [sub scalar](crate::ops::FloatTensorOps::float_sub_scalar).
    /// Int => [sub scalar](crate::ops::IntTensorOps::int_sub_scalar).
    SubScalar(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [div](crate::ops::FloatTensorOps::float_div).
    /// Int => [div](crate::ops::IntTensorOps::int_div).
    Div(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [div scalar](crate::ops::FloatTensorOps::float_div_scalar).
    /// Int => [div scalar](crate::ops::IntTensorOps::int_div_scalar).
    DivScalar(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [rem](crate::ops::FloatTensorOps::float_remainder).
    /// Int => [rem](crate::ops::IntTensorOps::int_remainder).
    Rem(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [rem scalar](crate::ops::FloatTensorOps::float_remainder_scalar).
    /// Int => [rem scalar](crate::ops::IntTensorOps::int_remainder_scalar).
    RemScalar(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [mul](crate::ops::FloatTensorOps::float_mul).
    /// Int => [mul](crate::ops::IntTensorOps::int_mul).
    Mul(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [mul scalar](crate::ops::FloatTensorOps::float_mul_scalar).
    /// Int => [mul scalar](crate::ops::IntTensorOps::int_mul_scalar).
    MulScalar(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [abs](crate::ops::FloatTensorOps::float_abs).
    /// Int => [abs](crate::ops::IntTensorOps::int_abs).
    Abs(UnaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [full](crate::ops::FloatTensorOps::float_full).
    /// Int => [full](crate::ops::IntTensorOps::int_full).
    Full(FullOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [mean dim](crate::ops::FloatTensorOps::float_mean_dim).
    /// Int => [mean dim](crate::ops::IntTensorOps::int_mean_dim).
    MeanDim(ReduceDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [mean](crate::ops::FloatTensorOps::float_mean).
    /// Int => [mean](crate::ops::IntTensorOps::int_mean).
    Mean(ReduceOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [sum](crate::ops::FloatTensorOps::float_sum).
    /// Int => [sum](crate::ops::IntTensorOps::int_sum).
    Sum(ReduceOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [sum dim](crate::ops::FloatTensorOps::float_sum_dim).
    /// Int => [sum dim](crate::ops::IntTensorOps::int_sum_dim).
    SumDim(ReduceDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [prod](crate::ops::FloatTensorOps::float_prod).
    /// Int => [prod](crate::ops::IntTensorOps::int_prod).
    Prod(ReduceOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [prod dim](crate::ops::FloatTensorOps::float_prod_dim).
    /// Int => [prod dim](crate::ops::IntTensorOps::int_prod_dim).
    ProdDim(ReduceDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [greater](crate::ops::FloatTensorOps::float_greater).
    /// Int => [greater](crate::ops::IntTensorOps::int_greater).
    Greater(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [greater elem](crate::ops::FloatTensorOps::float_greater_elem).
    /// Int => [greater elem](crate::ops::IntTensorOps::int_greater_elem).
    GreaterElem(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal](crate::ops::FloatTensorOps::float_greater_elem).
    /// Int => [greater elem](crate::ops::IntTensorOps::int_greater_elem).
    GreaterEqual(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal elem](crate::ops::FloatTensorOps::float_greater_equal_elem).
    /// Int => [greater equal elem](crate::ops::IntTensorOps::int_greater_equal_elem).
    GreaterEqualElem(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [lower](crate::ops::FloatTensorOps::float_lower).
    /// Int => [lower](crate::ops::IntTensorOps::int_lower).
    Lower(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [lower elem](crate::ops::FloatTensorOps::float_lower_elem).
    /// Int => [lower elem](crate::ops::IntTensorOps::int_lower_elem).
    LowerElem(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal](crate::ops::FloatTensorOps::float_lower_equal).
    /// Int => [lower equal](crate::ops::IntTensorOps::int_lower_equal).
    LowerEqual(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal elem](crate::ops::FloatTensorOps::float_lower_equal_elem).
    /// Int => [lower equal elem](crate::ops::IntTensorOps::int_lower_equal_elem).
    LowerEqualElem(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [argmax](crate::ops::FloatTensorOps::float_argmax).
    /// Int => [argmax](crate::ops::IntTensorOps::int_argmax).
    ArgMax(ReduceDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [argtopk](crate::ops::FloatTensorOps::float_argtopk).
    /// Int => [argtopk](crate::ops::IntTensorOps::int_argtopk).
    ArgTopK(ReduceDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [topk](crate::ops::FloatTensorOps::float_topk).
    /// Int => [topk](crate::ops::IntTensorOps::int_topk).
    TopK(ReduceDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [argmin](crate::ops::FloatTensorOps::float_argmin).
    /// Int => [argmin](crate::ops::IntTensorOps::int_argmin).
    ArgMin(ReduceDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [max](crate::ops::FloatTensorOps::float_max).
    /// Int => [max](crate::ops::IntTensorOps::int_max).
    Max(ReduceOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [max dim with indices](crate::ops::FloatTensorOps::float_max_dim_with_indices).
    /// Int => [max dim with indices](crate::ops::IntTensorOps::int_max_dim_with_indices).
    MaxDimWithIndices(ReduceDimWithIndicesOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [min dim with indices](crate::ops::FloatTensorOps::float_min_dim_with_indices).
    /// Int => [min dim with indices](crate::ops::IntTensorOps::int_min_dim_with_indices).
    MinDimWithIndices(ReduceDimWithIndicesOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [min](crate::ops::FloatTensorOps::float_min).
    /// Int => [min](crate::ops::IntTensorOps::int_min).
    Min(ReduceOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [max dim](crate::ops::FloatTensorOps::float_max_dim).
    /// Int => [max dim](crate::ops::IntTensorOps::int_max_dim).
    MaxDim(ReduceDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [min dim](crate::ops::FloatTensorOps::float_min_dim).
    /// Int => [min dim](crate::ops::IntTensorOps::int_min_dim).
    MinDim(ReduceDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [max_abs](crate::ops::FloatTensorOps::float_max_abs).
    /// Int => [max_abs](crate::ops::IntTensorOps::int_max_abs).
    MaxAbs(ReduceOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [max_abs dim](crate::ops::FloatTensorOps::float_max_abs_dim).
    /// Int => [max_abs dim](crate::ops::IntTensorOps::int_max_abs_dim).
    MaxAbsDim(ReduceDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [clamp](crate::ops::FloatTensorOps::float_clamp).
    /// Int => [clamp](crate::ops::IntTensorOps::int_clamp).
    Clamp(ClampOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [random](crate::ops::IntTensorOps::int_random).
    IntRandom(RandomOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [powf](crate::ops::FloatTensorOps::float_powi).
    /// Int => [powf](crate::ops::IntTensorOps::int_powi).
    Powi(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [powi_scalar](crate::ops::FloatTensorOps::float_powi_scalar).
    /// Int => [powi_scalar](crate::ops::IntTensorOps::int_powi_scalar).
    PowiScalar(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [cumsum](crate::ops::FloatTensorOps::float_cumsum).
    /// Int => [cumsum](crate::ops::IntTensorOps::int_cumsum).
    CumSum(DimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [cumprod](crate::ops::FloatTensorOps::float_cumprod).
    /// Int => [cumprod](crate::ops::IntTensorOps::int_cumprod).
    CumProd(DimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [cummin](crate::ops::FloatTensorOps::float_cummin).
    /// Int => [cummin](crate::ops::IntTensorOps::int_cummin).
    CumMin(DimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [cummax](crate::ops::FloatTensorOps::float_cummax).
    /// Int => [cummax](crate::ops::IntTensorOps::int_cummax).
    CumMax(DimOpIr),
}

/// Operation intermediate representation specific to an int tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum IntOperationIr {
    /// Operation corresponding to [into float](crate::ops::IntTensorOps::int_into_float).
    IntoFloat(CastOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise and](crate::ops::IntTensorOps::bitwise_and).
    BitwiseAnd(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise and scalar](crate::ops::IntTensorOps::bitwise_and_scalar).
    BitwiseAndScalar(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise or](crate::ops::IntTensorOps::bitwise_or).
    BitwiseOr(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise or scalar](crate::ops::IntTensorOps::bitwise_or_scalar).
    BitwiseOrScalar(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise xor](crate::ops::IntTensorOps::bitwise_xor).
    BitwiseXor(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise xor scalar](crate::ops::IntTensorOps::bitwise_xor_scalar).
    BitwiseXorScalar(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise not](crate::ops::IntTensorOps::bitwise_not).
    BitwiseNot(UnaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise left shift](crate::ops::IntTensorOps::bitwise_left_shift).
    BitwiseLeftShift(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise left shift scalar](crate::ops::IntTensorOps::bitwise_left_shift_scalar).
    BitwiseLeftShiftScalar(ScalarOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise right shift](crate::ops::IntTensorOps::bitwise_right_shift).
    BitwiseRightShift(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise right shift scalar](crate::ops::IntTensorOps::bitwise_right_shift_scalar).
    BitwiseRightShiftScalar(ScalarOpIr),
    /// Operation corresponding to [matmul](crate::ops::IntTensorOps::int_matmul).
    Matmul(MatmulOpIr),
}

/// Operation intermediate representation specific to a bool tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum BoolOperationIr {
    /// Operation corresponding to [into float](crate::ops::BoolTensorOps::bool_into_float).
    IntoFloat(CastOpIr),
    /// Operation corresponding to [into int](crate::ops::BoolTensorOps::bool_into_int).
    IntoInt(CastOpIr),
    /// Operation corresponding to [not](crate::ops::BoolTensorOps::bool_not).
    Not(UnaryOpIr),
    /// Operation corresponding to [and](crate::ops::BoolTensorOps::bool_and).
    And(BinaryOpIr),
    /// Operation corresponding to [or](crate::ops::BoolTensorOps::bool_or).
    Or(BinaryOpIr),
}

#[cfg(feature = "graph-distributed")]
/// Operations that can be done on distributed tensors.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum DistributedOperationIr {
    /// Operation corresponding to:
    /// [all_reduce](crate::distributed::DistributedBackend::all_reduce).
    AllReduce(AllReduceOpIr),
}
