use super::{Tensor, TensorPrimitive};
use crate::{DType, TensorData, ops::SparseOps, tensor::{FloatTensor, Int}};

#[derive(Clone, Debug)]
pub struct CsrTensor<B: SparseOps> {
    handle: B::CsrHandle,
    values: Tensor<B, 1>,
}

impl<B: SparseOps> CsrTensor<B> {
    pub fn from_data(data: &B::CsrData, device: &B::Device) -> Result<Self, B::SparseError> {
        B::csr_from_data(data, device).map(Self::from_handle)
    }

    pub fn from_handle(handle: B::CsrHandle) -> Self {
        let values = Tensor::from_primitive(TensorPrimitive::Float(B::csr_values(&handle)));
        Self { handle, values }
    }

    pub fn from_parts(handle: B::CsrHandle, values: Tensor<B, 1>) -> Result<Self, B::SparseError> {
        B::csr_validate_values(&handle, &unquantized(values.clone()))?;
        Ok(Self { handle, values })
    }

    pub fn into_parts(self) -> (B::CsrHandle, Tensor<B, 1>) {
        (self.handle, self.values)
    }

    pub fn shape(&self) -> [usize; 2] {
        B::csr_shape(&self.handle)
    }

    pub fn values(&self) -> Tensor<B, 1> {
        self.values.clone()
    }

    pub fn device(&self) -> B::Device {
        self.values.device()
    }

    pub fn to_device(&self, device: &B::Device) -> Self {
        Self {
            handle: B::csr_to_device(&self.handle, device),
            values: self.values.clone().to_device(device),
        }
    }

    pub async fn to_data(&self) -> Result<B::CsrData, B::SparseError> {
        B::csr_to_data(&self.handle, unquantized(self.values.clone())).await
    }

    pub fn with_values(&self, values: Tensor<B, 1>) -> Result<Self, B::SparseError> {
        let primitive = unquantized(values.clone());
        B::csr_validate_values(&self.handle, &primitive)?;
        Ok(Self { handle: self.handle.clone(), values })
    }

    pub fn matmul(&self, rhs: Tensor<B, 2>) -> Result<Tensor<B, 2>, B::SparseError> {
        self.matmul_impl(rhs, false)
    }

    pub fn gather(&self, dense: Tensor<B, 2>) -> Result<Tensor<B, 1>, B::SparseError> {
        let output = B::csr_gather(&self.handle, unquantized(dense))?;
        Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }

    pub fn scatter_add(&self) -> Result<Tensor<B, 2>, B::SparseError> {
        let output = B::csr_scatter_add(&self.handle, unquantized(self.values.clone()))?;
        Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }

    pub fn mul_dense(&self, rhs: Tensor<B, 2>) -> Result<Self, B::SparseError> {
        self.with_values(self.values.clone() * self.gather(rhs)?)
    }

    pub fn to_dense(&self) -> Result<Tensor<B, 2>, B::SparseError> {
        let output = B::csr_to_dense(&self.handle, unquantized(self.values.clone()))?;
        Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }

    pub fn transpose_matmul(&self, rhs: Tensor<B, 2>) -> Result<Tensor<B, 2>, B::SparseError> {
        self.matmul_impl(rhs, true)
    }

    pub fn transpose(&self) -> Result<Self, B::SparseError> {
        let (handle, permutation) = B::csr_transpose_with_permutation(&self.handle)?;
        let nnz = permutation.len();
        let indices = Tensor::<B, 1, Int>::from_data(
            TensorData::new(permutation, [nnz]), (&self.device(), DType::I64),
        );
        Self::from_parts(handle, self.values.clone().select(0, indices))
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, B::SparseError> {
        self.add_scaled(rhs, 1.0, 1.0)
    }

    pub fn sparse_matmul(&self, rhs: &Self) -> Result<Self, B::SparseError> {
        let handle = B::csr_product_pattern(&self.handle, &rhs.handle)?;
        let values = B::csr_sampled_sparse_matmul(
            &handle, &self.handle, unquantized(self.values.clone()),
            &rhs.handle, unquantized(rhs.values.clone()), false, false,
        )?;
        Self::from_parts(handle, Tensor::from_primitive(TensorPrimitive::Float(values)))
    }

    pub fn sampled_sparse_matmul(&self, lhs: &Self, rhs: &Self) -> Result<Self, B::SparseError> {
        let values = B::csr_sampled_sparse_matmul(
            &self.handle, &lhs.handle, unquantized(lhs.values.clone()),
            &rhs.handle, unquantized(rhs.values.clone()), false, false,
        )?;
        Self::from_parts(self.handle.clone(), Tensor::from_primitive(TensorPrimitive::Float(values)))
    }

    pub fn add_scaled(&self, rhs: &Self, alpha: f32, beta: f32) -> Result<Self, B::SparseError> {
        let plan = B::csr_add_prepare(&self.handle, &rhs.handle)?;
        let values = B::csr_add(&plan, unquantized(self.values.clone()), unquantized(rhs.values.clone()), alpha, beta)?;
        Self::from_parts(plan.output, Tensor::from_primitive(TensorPrimitive::Float(values)))
    }

    fn matmul_impl(&self, rhs: Tensor<B, 2>, transpose: bool) -> Result<Tensor<B, 2>, B::SparseError> {
        let output = B::csr_matmul(
            &self.handle, unquantized(self.values.clone()), unquantized(rhs), transpose,
        )?;
        Ok(Tensor::from_primitive(TensorPrimitive::Float(output)))
    }

    pub fn sampled_matmul(
        &self,
        lhs: Tensor<B, 2>,
        rhs: Tensor<B, 2>,
    ) -> Result<Self, B::SparseError> {
        let values = B::csr_sampled_matmul(&self.handle, unquantized(lhs), unquantized(rhs))?;
        Ok(Self {
            handle: self.handle.clone(),
            values: Tensor::from_primitive(TensorPrimitive::Float(values)),
        })
    }
}

fn unquantized<B: SparseOps, const D: usize>(tensor: Tensor<B, D>) -> FloatTensor<B> {
    match tensor.into_primitive() {
        TensorPrimitive::Float(tensor) => tensor,
        TensorPrimitive::QFloat(_) => panic!("Sparse matrix operations require unquantized tensors"),
    }
}
