use super::*;
use ruda_kernel::{dsl::calculate_ruda_count_elemwise, tensor::initialization::zeros};

pub fn csr_gather<R: Runtime>(matrix: &CsrTensor<R>, dense: RudaTensor<R>) -> Result<RudaTensor<R>, SparseError> {
    matrix.validate_dense(&dense, &[matrix.rows, matrix.columns])?;
    let elements = matrix.rows.checked_mul(matrix.columns)
        .ok_or(SparseError::SizeOverflow("CSR gather input"))?;
    dimension(elements, "CSR gather input")?;
    let output = empty_device_contiguous_dtype(
        matrix.values.client.clone(), matrix.values.device.clone(), [matrix.nnz].into(), DType::F32,
    );
    if matrix.nnz == 0 { return Ok(output); }
    let ruda_dim = RudaDim::new(matrix.values.client.properties(), matrix.nnz);
    let ruda_count = calculate_ruda_count_elemwise(&matrix.values.client, matrix.nnz, ruda_dim);
    gather::launch::<R>(
        &matrix.values.client, ruda_count, ruda_dim,
        matrix.offsets.clone().into_array_arg(), matrix.indices.clone().into_array_arg(),
        into_contiguous(dense).into_array_arg(), output.clone().into_array_arg(),
        matrix.rows as u32, matrix.columns as u32, matrix.nnz as u32, matrix.base.value(),
        include_str!("indexing.rs").to_owned(),
    );
    Ok(output)
}

pub fn csr_scatter_add<R: Runtime>(matrix: &CsrTensor<R>) -> Result<RudaTensor<R>, SparseError> {
    let elements = matrix.rows.checked_mul(matrix.columns)
        .ok_or(SparseError::SizeOverflow("CSR scatter output"))?;
    dimension(elements, "CSR scatter output")?;
    let output = zeros::<R>(matrix.values.device.clone(), [matrix.rows, matrix.columns].into(), DType::F32);
    if elements == 0 || matrix.nnz == 0 { return Ok(output); }
    let ruda_dim = RudaDim::new(matrix.values.client.properties(), matrix.rows);
    let ruda_count = calculate_ruda_count_elemwise(&matrix.values.client, matrix.rows, ruda_dim);
    scatter_add::launch::<R>(
        &matrix.values.client, ruda_count, ruda_dim,
        matrix.offsets.clone().into_array_arg(), matrix.indices.clone().into_array_arg(),
        matrix.values.clone().into_array_arg(), output.clone().into_array_arg(),
        matrix.rows as u32, matrix.columns as u32, matrix.base.value(),
        include_str!("indexing.rs").to_owned(),
    );
    Ok(output)
}

#[ruda(launch)]
fn gather(
    offsets: &Array<u32>, indices: &Array<u32>, dense: &Array<f32>, output: &mut Array<f32>,
    rows: u32, columns: u32, nnz: u32, base: u32, #[comptime] _source: String,
) {
    let entry = ABSOLUTE_POS;
    if entry < nnz as usize {
        let mut low = 0usize;
        let mut high = rows as usize;
        while low < high {
            let middle = low + (high - low) / 2;
            if (offsets[middle + 1] - base) as usize <= entry {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        output[entry] = dense[low * columns as usize + (indices[entry] - base) as usize];
    }
}

#[ruda(launch)]
fn scatter_add(
    offsets: &Array<u32>, indices: &Array<u32>, values: &Array<f32>, output: &mut Array<f32>,
    rows: u32, columns: u32, base: u32, #[comptime] _source: String,
) {
    let row = ABSOLUTE_POS;
    if row < rows as usize {
        let mut entry = (offsets[row] - base) as usize;
        let end = (offsets[row + 1] - base) as usize;
        while entry < end {
            let destination = row * columns as usize + (indices[entry] - base) as usize;
            output[destination] = output[destination] + values[entry];
            entry += 1;
        }
    }
}
