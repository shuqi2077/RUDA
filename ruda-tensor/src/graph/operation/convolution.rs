use super::*;


#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv1dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: Conv1dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv1dXBackwardOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub output_grad: TensorIr,
    pub options: Conv1dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv1dWeightBackwardOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub output_grad: TensorIr,
    pub options: Conv1dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv1dBiasBackwardOpIr {
    pub x: TensorIr,
    pub bias: TensorIr,
    pub output_grad: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv2dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: Conv2dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv2dXBackwardOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub output_grad: TensorIr,
    pub options: Conv2dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv2dWeightBackwardOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub output_grad: TensorIr,
    pub options: Conv2dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv2dBiasBackwardOpIr {
    pub x: TensorIr,
    pub bias: TensorIr,
    pub output_grad: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DeformConv2dOpIr {
    pub x: TensorIr,
    pub offset: TensorIr,
    pub weight: TensorIr,
    pub mask: Option<TensorIr>,
    pub bias: Option<TensorIr>,
    pub options: DeformableConv2dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DeformConv2dBackwardOpIr {
    pub x: TensorIr,
    pub offset: TensorIr,
    pub weight: TensorIr,
    pub mask: Option<TensorIr>,
    pub bias: Option<TensorIr>,
    pub out_grad: TensorIr,
    pub options: DeformableConv2dOptionsIr,
    pub input_grad: TensorIr,
    pub offset_grad: TensorIr,
    pub weight_grad: TensorIr,
    pub mask_grad: Option<TensorIr>,
    pub bias_grad: Option<TensorIr>,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv3dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: Conv3dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv3dXBackwardOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub output_grad: TensorIr,
    pub options: Conv3dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv3dWeightBackwardOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub output_grad: TensorIr,
    pub options: Conv3dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv3dBiasBackwardOpIr {
    pub x: TensorIr,
    pub bias: TensorIr,
    pub output_grad: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose1dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: ConvTranspose1dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose2dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: ConvTranspose2dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose3dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: ConvTranspose3dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv1dOptionsIr {
    pub stride: [usize; 1],
    pub padding: [usize; 1],
    pub dilation: [usize; 1],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv2dOptionsIr {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DeformableConv2dOptionsIr {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub weight_groups: usize,
    pub offset_groups: usize,
    #[serde(default)]
    pub padding_end: Option<[usize; 2]>,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv3dOptionsIr {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose1dOptionsIr {
    pub stride: [usize; 1],
    pub padding: [usize; 1],
    pub padding_out: [usize; 1],
    pub dilation: [usize; 1],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose2dOptionsIr {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub padding_out: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose3dOptionsIr {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub padding_out: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
}

impl From<ConvOptions<1>> for Conv1dOptionsIr {
    fn from(value: ConvOptions<1>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<ConvOptions<2>> for Conv2dOptionsIr {
    fn from(value: ConvOptions<2>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<ConvOptions<3>> for Conv3dOptionsIr {
    fn from(value: ConvOptions<3>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<DeformConvOptions<2>> for DeformableConv2dOptionsIr {
    fn from(value: DeformConvOptions<2>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            weight_groups: value.weight_groups,
            offset_groups: value.offset_groups,
            padding_end: value.padding_end,
        }
    }
}

impl From<ConvTransposeOptions<1>> for ConvTranspose1dOptionsIr {
    fn from(value: ConvTransposeOptions<1>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            padding_out: value.padding_out,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<ConvTransposeOptions<2>> for ConvTranspose2dOptionsIr {
    fn from(value: ConvTransposeOptions<2>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            padding_out: value.padding_out,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<ConvTransposeOptions<3>> for ConvTranspose3dOptionsIr {
    fn from(value: ConvTransposeOptions<3>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            padding_out: value.padding_out,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<Conv1dOptionsIr> for ConvOptions<1> {
    fn from(val: Conv1dOptionsIr) -> Self {
        ConvOptions {
            stride: val.stride,
            padding: val.padding,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<Conv2dOptionsIr> for ConvOptions<2> {
    fn from(val: Conv2dOptionsIr) -> Self {
        ConvOptions {
            stride: val.stride,
            padding: val.padding,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<Conv3dOptionsIr> for ConvOptions<3> {
    fn from(val: Conv3dOptionsIr) -> Self {
        ConvOptions {
            stride: val.stride,
            padding: val.padding,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<DeformableConv2dOptionsIr> for DeformConvOptions<2> {
    fn from(value: DeformableConv2dOptionsIr) -> Self {
        DeformConvOptions {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            weight_groups: value.weight_groups,
            offset_groups: value.offset_groups,
            padding_end: value.padding_end,
        }
    }
}

impl From<ConvTranspose1dOptionsIr> for ConvTransposeOptions<1> {
    fn from(val: ConvTranspose1dOptionsIr) -> Self {
        ConvTransposeOptions {
            stride: val.stride,
            padding: val.padding,
            padding_out: val.padding_out,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<ConvTranspose2dOptionsIr> for ConvTransposeOptions<2> {
    fn from(val: ConvTranspose2dOptionsIr) -> Self {
        ConvTransposeOptions {
            stride: val.stride,
            padding: val.padding,
            padding_out: val.padding_out,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<ConvTranspose3dOptionsIr> for ConvTransposeOptions<3> {
    fn from(val: ConvTranspose3dOptionsIr) -> Self {
        ConvTransposeOptions {
            stride: val.stride,
            padding: val.padding,
            padding_out: val.padding_out,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}
