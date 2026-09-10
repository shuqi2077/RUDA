use super::*;
use ruda_kernel::tensor::{readback::into_data, transfer::to_device};

impl<R: Runtime> CsrTensor<R> {
    pub fn device(&self) -> &R::Device {
        &self.values.device
    }

    pub fn to_device(&self, device: &R::Device) -> Self
    where
        R::Device: PartialEq,
    {
        Self {
            rows: self.rows,
            columns: self.columns,
            nnz: self.nnz,
            base: self.base,
            offsets: to_device(self.offsets.clone(), device),
            indices: to_device(self.indices.clone(), device),
            values: to_device(self.values.clone(), device),
            host_offsets: self.host_offsets.clone(),
            host_indices: self.host_indices.clone(),
        }
    }

    pub async fn to_csr(&self) -> Result<CsrMatrixOwned, SparseError> {
        let values = into_data(self.values.clone()).await
            .map_err(SparseError::TensorExecution)?
            .into_vec::<f32>().map_err(SparseError::TensorData)?;
        CsrMatrixOwned::new(
            self.rows,
            self.columns,
            self.host_offsets.to_vec(),
            self.host_indices.to_vec(),
            values,
            self.base,
        )
    }

    pub async fn to_runtime_via_host<S: Runtime>(
        &self,
        device: &S::Device,
    ) -> Result<CsrTensor<S>, SparseError> {
        let matrix = self.to_csr().await?;
        CsrTensor::from_csr(matrix.as_ref(), Operation::None, device)
    }
}
