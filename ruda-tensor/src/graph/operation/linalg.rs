use super::*;


#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MatmulOpIr {
    pub lhs: TensorIr,
    pub rhs: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct CrossOpIr {
    pub lhs: TensorIr,
    pub rhs: TensorIr,
    pub out: TensorIr,
    pub dim: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct LinearOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct LinearXBackwardOpIr {
    pub weight: TensorIr,
    pub output_grad: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct LinearWeightBackwardOpIr {
    pub x: TensorIr,
    pub output_grad: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct LinearBiasBackwardOpIr {
    pub output_grad: TensorIr,
    pub out: TensorIr,
}
