use super::*;

pub fn sampled_csrgemm<R: Runtime>(
    pattern: &CsrTensor<R>, operation_a: Operation, operation_b: Operation,
    a: &CsrTensor<R>, b: &CsrTensor<R>,
) -> Result<RudaTensor<R>, SparseError> {
    let (a, b) = super::sparse_binary::operands(operation_a, operation_b, a, b)?;
    if pattern.values.device.to_id() != a.values.device.to_id() {
        return Err(SparseError::Device("sampled sparse product pattern must use the operand device"));
    }
    if a.columns != b.rows || pattern.rows != a.rows || pattern.columns != b.columns {
        return Err(SparseError::DimensionMismatch("sampled sparse product dimensions differ from pattern"));
    }
    let grid = pattern.grid(pattern.nnz)?;
    let output = empty_device_contiguous_dtype(
        a.values.client.clone(), a.values.device.clone(), [pattern.nnz].into(), DType::F32,
    );
    if pattern.nnz == 0 { return Ok(output); }
    let mut rows = vec![0u32; pattern.nnz];
    let base = pattern.base.value();
    for row in 0..pattern.rows {
        let start = (pattern.host_offsets[row] - base) as usize;
        let end = (pattern.host_offsets[row + 1] - base) as usize;
        rows[start..end].fill(dimension(row, "sampled sparse product row")?);
    }
    let rows: RudaTensor<R> = from_data(TensorData::new(rows, [pattern.nnz]), &a.values.device);
    super::sparse_binary::kernel::product::launch::<R>(
        &a.values.client, grid, RudaDim::new_1d(128),
        a.offsets.clone().into_array_arg(), a.indices.clone().into_array_arg(), a.values.clone().into_array_arg(),
        b.offsets.clone().into_array_arg(), b.indices.clone().into_array_arg(), b.values.clone().into_array_arg(),
        rows.into_array_arg(), pattern.indices.clone().into_array_arg(), output.clone().into_array_arg(),
        pattern.nnz as u32, a.base.value(), b.base.value(), 1.0,
        include_str!("sparse_binary/kernel.rs").to_owned(), base,
    );
    Ok(output)
}
