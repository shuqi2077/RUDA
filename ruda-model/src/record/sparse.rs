use core::marker::PhantomData;
use ruda_tensor::{api::sparse::CsrTensor, ops::SparseOps};
use serde::{Serialize, de::DeserializeOwned};
use super::{PrecisionSettings, Record};

#[derive(Clone, Debug)]
pub struct CsrTensorRecord<B: SparseOps> {
    data: B::CsrData,
    marker: PhantomData<fn() -> B>,
}

impl<B: SparseOps> CsrTensorRecord<B> {
    pub fn from_data(data: B::CsrData) -> Self {
        Self { data, marker: PhantomData }
    }

    pub fn into_data(self) -> B::CsrData {
        self.data
    }

    pub async fn capture(tensor: &CsrTensor<B>) -> Result<Self, B::SparseError> {
        tensor.to_data().await.map(Self::from_data)
    }

    pub fn restore(&self, device: &B::Device) -> Result<CsrTensor<B>, B::SparseError> {
        CsrTensor::from_data(&self.data, device)
    }
}

impl<B: SparseOps> Record<B> for CsrTensorRecord<B>
where
    B::CsrData: Serialize + DeserializeOwned,
{
    type Item<S: PrecisionSettings> = B::CsrData;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self.data
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        Self::from_data(item)
    }
}
