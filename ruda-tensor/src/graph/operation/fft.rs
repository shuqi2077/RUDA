use super::*;


#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct RfftOpIr {
    pub signal: TensorIr,
    pub dim: usize,
    pub n: Option<usize>,
    pub out_re: TensorIr,
    pub out_im: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct IRfftOpIr {
    pub input_re: TensorIr,
    pub input_im: TensorIr,
    pub dim: usize,
    pub n: Option<usize>,
    pub out_signal: TensorIr,
}
