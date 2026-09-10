use super::*;
use crate::symbolic::{symbolic_product, symbolic_sum};

pub(super) mod kernel;

impl<R: Runtime> CsrTensor<R> {
    pub fn product_pattern(&self, rhs: &Self) -> Result<Self, SparseError> {
        if self.values.device.to_id() != rhs.values.device.to_id() {
            return Err(SparseError::Device("CSR product operands must use the same device"));
        }
        if self.columns != rhs.rows {
            return Err(SparseError::DimensionMismatch("CSR product inner dimensions differ"));
        }
        let symbolic = symbolic_product(self.pattern(), rhs.pattern())?;
        let mut output = self.from_pattern(self.rows, rhs.columns, symbolic.row_offsets, symbolic.column_indices)?;
        output.values = ruda_kernel::tensor::initialization::zeros(
            self.values.device.clone(), [output.nnz].into(), DType::F32,
        );
        Ok(output)
    }

    pub fn sum_pattern(&self, rhs: &Self) -> Result<(Self, Vec<u32>, Vec<u32>), SparseError> {
        if self.values.device.to_id() != rhs.values.device.to_id() {
            return Err(SparseError::Device("CSR sum operands must use the same device"));
        }
        if self.rows != rhs.rows || self.columns != rhs.columns {
            return Err(SparseError::DimensionMismatch("CSR sum operands must have the same shape"));
        }
        let symbolic = symbolic_sum(self.pattern(), rhs.pattern())?;
        let left = crate::symbolic::sum_entry_mapping(self.pattern(), &symbolic, self.base)?;
        let right = crate::symbolic::sum_entry_mapping(rhs.pattern(), &symbolic, self.base)?;
        let mut output = self.from_pattern(self.rows, self.columns, symbolic.row_offsets, symbolic.column_indices)?;
        output.values = ruda_kernel::tensor::initialization::zeros(
            self.values.device.clone(), [output.nnz].into(), DType::F32,
        );
        Ok((output, left, right))
    }
}

pub(super) fn operands<R: Runtime>(
    operation_a: Operation,
    operation_b: Operation,
    a: &CsrTensor<R>,
    b: &CsrTensor<R>,
) -> Result<(CsrTensor<R>, CsrTensor<R>), SparseError> {
    if a.values.device.to_id() != b.values.device.to_id() {
        return Err(SparseError::Device(
            "sparse operands must use the same device",
        ));
    }
    let a = if operation_a == Operation::None {
        a.clone()
    } else {
        a.transpose()?
    };
    let b = if operation_b == Operation::None {
        b.clone()
    } else {
        b.transpose()?
    };
    Ok((a, b))
}

/// `alpha * op(A) + beta * op(B)`. The shared host symbolic phase keeps the sorted union,
/// including explicit zeros; numeric duplicate accumulation runs on the device.
pub fn csrgeam<R: Runtime>(
    operation_a: Operation,
    operation_b: Operation,
    alpha: f32,
    a: &CsrTensor<R>,
    beta: f32,
    b: &CsrTensor<R>,
) -> Result<CsrTensor<R>, SparseError> {
    let (a, b) = operands(operation_a, operation_b, a, b)?;
    if a.rows != b.rows || a.columns != b.columns {
        return Err(SparseError::DimensionMismatch(
            "op(A) and op(B) must have the same shape",
        ));
    }
    let symbolic = symbolic_sum(a.pattern(), b.pattern())?;
    let grid = a.grid(symbolic.column_indices.len())?;
    let output = a.from_pattern(
        a.rows,
        a.columns,
        symbolic.row_offsets,
        symbolic.column_indices,
    )?;
    if output.nnz == 0 {
        return Ok(output);
    }
    let rows: RudaTensor<R> = from_data(
        TensorData::new(symbolic.entry_rows, [output.nnz]),
        &a.values.device,
    );
    kernel::sum::launch::<R>(
        &a.values.client,
        grid,
        RudaDim::new_1d(128),
        a.offsets.clone().into_array_arg(),
        a.indices.clone().into_array_arg(),
        a.values.clone().into_array_arg(),
        b.offsets.clone().into_array_arg(),
        b.indices.clone().into_array_arg(),
        b.values.clone().into_array_arg(),
        rows.into_array_arg(),
        output.indices.clone().into_array_arg(),
        output.values.clone().into_array_arg(),
        output.nnz as u32,
        a.base.value(),
        b.base.value(),
        alpha,
        beta,
        include_str!("sparse_binary/kernel.rs").to_owned(),
    );
    Ok(output)
}

/// `alpha * op(A) * op(B)`. Symbolic structure is shared with the native path;
/// FP32 FMA over duplicate products and 32-lane reduction run on the device.
pub fn csrgemm<R: Runtime>(
    operation_a: Operation,
    operation_b: Operation,
    alpha: f32,
    a: &CsrTensor<R>,
    b: &CsrTensor<R>,
) -> Result<CsrTensor<R>, SparseError> {
    let (a, b) = operands(operation_a, operation_b, a, b)?;
    if a.columns != b.rows {
        return Err(SparseError::DimensionMismatch(
            "op(A) columns must equal op(B) rows",
        ));
    }
    let symbolic = symbolic_product(a.pattern(), b.pattern())?;
    let grid = a.grid(symbolic.column_indices.len())?;
    let output = a.from_pattern(
        a.rows,
        b.columns,
        symbolic.row_offsets,
        symbolic.column_indices,
    )?;
    if output.nnz == 0 {
        return Ok(output);
    }
    let rows: RudaTensor<R> = from_data(
        TensorData::new(symbolic.entry_rows, [output.nnz]),
        &a.values.device,
    );
    kernel::product::launch::<R>(
        &a.values.client,
        grid,
        RudaDim::new_1d(128),
        a.offsets.clone().into_array_arg(),
        a.indices.clone().into_array_arg(),
        a.values.clone().into_array_arg(),
        b.offsets.clone().into_array_arg(),
        b.indices.clone().into_array_arg(),
        b.values.clone().into_array_arg(),
        rows.into_array_arg(),
        output.indices.clone().into_array_arg(),
        output.values.clone().into_array_arg(),
        output.nnz as u32,
        a.base.value(),
        b.base.value(),
        alpha,
        include_str!("sparse_binary/kernel.rs").to_owned(),
        output.base.value(),
    );
    Ok(output)
}
