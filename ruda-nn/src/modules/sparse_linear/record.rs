use ruda_model::module::Param;
use ruda_model::record::{CsrTensorRecord, PrecisionSettings, Record};
use ruda_model::tensor::{Tensor, ops::SparseOps};
use serde::{Serialize, de::DeserializeOwned};

#[derive(Clone, Debug)]
pub struct SparseLinearRecord<B: SparseOps> {
    pub weight: CsrTensorRecord<B>,
    pub weight_id: u64,
    pub bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: SparseOps> Record<B> for SparseLinearRecord<B>
where
    B::CsrData: Serialize + DeserializeOwned,
{
    type Item<S: PrecisionSettings> = (
        B::CsrData,
        u64,
        <Option<Param<Tensor<B, 1>>> as Record<B>>::Item<S>,
    );

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        (self.weight.into_item::<S>(), self.weight_id, self.bias.into_item::<S>())
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            weight: CsrTensorRecord::from_item::<S>(item.0, device),
            weight_id: item.1,
            bias: Record::from_item::<S>(item.2, device),
        }
    }
}
