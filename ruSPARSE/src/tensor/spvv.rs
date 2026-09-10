use super::*;
use crate::SparseVector;

/// A sparse vector using the same device storage and index contract as one CSR row.
#[derive(Clone, Debug)]
pub struct SparseVectorTensor<R: Runtime> {
    matrix: CsrTensor<R>,
}

impl<R: Runtime> SparseVectorTensor<R> {
    pub fn from_sparse(vector: SparseVector<'_>, device: &R::Device) -> Result<Self, SparseError> {
        let base = vector.index_base().value();
        let end = dimension(vector.nnz(), "SpVV nnz")?
            .checked_add(base)
            .ok_or(SparseError::SizeOverflow("SpVV index range"))?;
        let offsets = [base, end];
        let matrix = CsrMatrix::new(
            1,
            vector.size(),
            &offsets,
            vector.indices(),
            vector.values(),
            vector.index_base(),
        )?;
        Ok(Self {
            matrix: CsrTensor::from_csr(matrix, Operation::None, device)?,
        })
    }

    pub fn size(&self) -> usize {
        self.matrix.columns
    }
    pub fn nnz(&self) -> usize {
        self.matrix.nnz
    }
    pub fn index_base(&self) -> IndexBase {
        self.matrix.base
    }

    pub fn device(&self) -> &R::Device {
        self.matrix.device()
    }

    pub fn indices(&self) -> RudaTensor<R> {
        self.matrix.column_indices()
    }

    pub fn values(&self) -> RudaTensor<R> {
        self.matrix.values()
    }

    pub fn with_values(&self, values: RudaTensor<R>) -> Result<Self, SparseError> {
        Ok(Self { matrix: self.matrix.with_values(values)? })
    }

    pub fn to_device(&self, device: &R::Device) -> Self
    where
        R::Device: PartialEq,
    {
        Self { matrix: self.matrix.to_device(device) }
    }

    pub async fn to_sparse(&self) -> Result<crate::SparseVectorOwned, SparseError> {
        let matrix = self.matrix.to_csr().await?;
        let matrix = matrix.as_ref();
        crate::SparseVectorOwned::new(
            matrix.columns(),
            matrix.column_indices().to_vec(),
            matrix.values().to_vec(),
            matrix.index_base(),
        )
    }

    pub async fn to_runtime_via_host<S: Runtime>(
        &self,
        device: &S::Device,
    ) -> Result<SparseVectorTensor<S>, SparseError> {
        Ok(SparseVectorTensor { matrix: self.matrix.to_runtime_via_host::<S>(device).await? })
    }
}

/// Sparse/dense dot product. Real FP32 conjugation is the identity; duplicates are accumulated.
pub fn spvv<R: Runtime>(
    _operation: Operation,
    x: &SparseVectorTensor<R>,
    y: RudaTensor<R>,
) -> Result<RudaTensor<R>, SparseError> {
    let matrix = &x.matrix;
    matrix.validate_dense(&y, &[matrix.columns])?;
    matrix.grid(1)?;
    let output = empty_device_contiguous_dtype(
        matrix.values.client.clone(),
        matrix.values.device.clone(),
        [1].into(),
        DType::F32,
    );
    dot::launch::<R>(
        &matrix.values.client,
        RudaCount::Static(1, 1, 1),
        RudaDim::new_1d(32),
        matrix.indices.clone().into_array_arg(),
        matrix.values.clone().into_array_arg(),
        into_contiguous(y).into_array_arg(),
        output.clone().into_array_arg(),
        matrix.nnz as u32,
        matrix.base.value(),
        include_str!("spvv.rs").to_owned(),
    );
    Ok(output)
}

#[ruda(launch)]
fn dot(
    indices: &Array<u32>,
    values: &Array<f32>,
    y: &Array<f32>,
    output: &mut Array<f32>,
    nnz: u32,
    base: u32,
    #[comptime] _source: String,
) {
    let lane = UNIT_POS as usize;
    let mut entry = lane;
    let mut sum = 0f32;
    while entry < nnz as usize {
        sum = fma(values[entry], y[(indices[entry] - base) as usize], sum);
        entry += 32;
    }
    sum = plane_sum(sum);
    if lane == 0 {
        output[0] = sum;
    }
}
