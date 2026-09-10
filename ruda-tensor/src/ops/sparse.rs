use crate::{Backend, TensorMetadata, tensor::FloatTensor};
use core::fmt::{Debug, Display};
use alloc::vec::Vec;

#[derive(Clone, Debug)]
pub struct CsrAddition<H> {
    pub left: H,
    pub right: H,
    pub output: H,
    pub left_entries: Vec<u32>,
    pub right_entries: Vec<u32>,
}

pub trait SparseOps: Backend {
    type CsrHandle: Clone + Debug + Send + 'static;
    type CsrData: Clone + Debug + Send + 'static;
    type SparseError: Debug + Display;

    fn csr_from_data(data: &Self::CsrData, device: &Self::Device) -> Result<Self::CsrHandle, Self::SparseError>;
    fn csr_to_device(matrix: &Self::CsrHandle, device: &Self::Device) -> Self::CsrHandle;
    fn csr_transpose_with_permutation(
        matrix: &Self::CsrHandle,
    ) -> Result<(Self::CsrHandle, Vec<u32>), Self::SparseError>;
    async fn csr_to_data(
        matrix: &Self::CsrHandle,
        values: FloatTensor<Self>,
    ) -> Result<Self::CsrData, Self::SparseError>;

    fn csr_shape(matrix: &Self::CsrHandle) -> [usize; 2];
    fn csr_nnz(matrix: &Self::CsrHandle) -> usize;
    fn csr_add_prepare(
        left: &Self::CsrHandle, right: &Self::CsrHandle,
    ) -> Result<CsrAddition<Self::CsrHandle>, Self::SparseError>;
    fn csr_add(
        plan: &CsrAddition<Self::CsrHandle>, left: FloatTensor<Self>, right: FloatTensor<Self>,
        alpha: f32, beta: f32,
    ) -> Result<FloatTensor<Self>, Self::SparseError>;
    fn csr_product_pattern(
        left: &Self::CsrHandle, right: &Self::CsrHandle,
    ) -> Result<Self::CsrHandle, Self::SparseError>;
    fn csr_sampled_sparse_matmul(
        pattern: &Self::CsrHandle,
        left: &Self::CsrHandle, left_values: FloatTensor<Self>,
        right: &Self::CsrHandle, right_values: FloatTensor<Self>,
        transpose_left: bool, transpose_right: bool,
    ) -> Result<FloatTensor<Self>, Self::SparseError>;
    fn csr_validate_operand<T: TensorMetadata>(
        matrix: &Self::CsrHandle,
        operand: &T,
        device: &Self::Device,
        shape: &[usize],
    ) -> Result<(), Self::SparseError>;
    fn csr_values(matrix: &Self::CsrHandle) -> FloatTensor<Self>;
    fn csr_validate_values(
        matrix: &Self::CsrHandle,
        values: &FloatTensor<Self>,
    ) -> Result<(), Self::SparseError>;

    fn csr_gather(
        matrix: &Self::CsrHandle, dense: FloatTensor<Self>,
    ) -> Result<FloatTensor<Self>, Self::SparseError>;

    fn csr_scatter_add(
        matrix: &Self::CsrHandle, values: FloatTensor<Self>,
    ) -> Result<FloatTensor<Self>, Self::SparseError>;

    fn csr_to_dense(
        matrix: &Self::CsrHandle,
        values: FloatTensor<Self>,
    ) -> Result<FloatTensor<Self>, Self::SparseError>;

    fn csr_to_dense_backward(
        matrix: &Self::CsrHandle,
        grad: FloatTensor<Self>,
    ) -> Result<FloatTensor<Self>, Self::SparseError>;

    fn csr_matmul(
        matrix: &Self::CsrHandle,
        values: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        transpose: bool,
    ) -> Result<FloatTensor<Self>, Self::SparseError>;

    fn csr_sampled_matmul(
        matrix: &Self::CsrHandle,
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
    ) -> Result<FloatTensor<Self>, Self::SparseError>;
}
