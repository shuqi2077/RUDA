use super::*;

/// Sample `alpha * op(A) * op(B) + beta * C` at every stored entry of C.
/// CSR structure, duplicate entries and index base are preserved; inputs are not overwritten.
pub fn sddmm<R: Runtime>(
    operation_a: Operation,
    operation_b: Operation,
    alpha: f32,
    mut a: RudaTensor<R>,
    mut b: RudaTensor<R>,
    beta: f32,
    c: &CsrTensor<R>,
) -> Result<CsrTensor<R>, SparseError> {
    if a.meta.rank() != 2 || b.meta.rank() != 2 {
        return Err(SparseError::DimensionMismatch(
            "SDDMM dense operands must have two axes",
        ));
    }
    if operation_a != Operation::None {
        a.meta.swap(0, 1);
    }
    if operation_b != Operation::None {
        b.meta.swap(0, 1);
    }
    let inner = a.meta.shape()[1];
    dimension(inner, "SDDMM inner dimension")?;
    c.validate_dense(&a, &[c.rows, inner])?;
    c.validate_dense(&b, &[inner, c.columns])?;
    let grid = c.grid(c.nnz)?;
    let mut output = c.clone();
    output.values = empty_device_contiguous_dtype(
        c.values.client.clone(),
        c.values.device.clone(),
        [c.nnz].into(),
        DType::F32,
    );
    if c.nnz == 0 {
        return Ok(output);
    }
    sampled_dot::launch::<R>(
        &c.values.client,
        grid,
        RudaDim::new_1d(128),
        c.offsets.clone().into_array_arg(),
        c.indices.clone().into_array_arg(),
        c.values.clone().into_array_arg(),
        into_contiguous(a).into_array_arg(),
        into_contiguous(b).into_array_arg(),
        output.values.clone().into_array_arg(),
        c.rows as u32,
        c.columns as u32,
        c.nnz as u32,
        inner as u32,
        c.base.value(),
        alpha,
        beta,
        include_str!("sddmm.rs").to_owned(),
    );
    Ok(output)
}

#[ruda(launch)]
fn sampled_dot(
    offsets: &Array<u32>,
    indices: &Array<u32>,
    values: &Array<f32>,
    a: &Array<f32>,
    b: &Array<f32>,
    output: &mut Array<f32>,
    rows: u32,
    columns: u32,
    nnz: u32,
    inner: u32,
    base: u32,
    alpha: f32,
    beta: f32,
    #[comptime] _source: String,
) {
    let entry = ABSOLUTE_POS / 32;
    let lane = (UNIT_POS % 32) as usize;
    if entry < nnz as usize {
        // Upper bound on offsets finds the owner even across empty rows.
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
        let row = low;
        let column = (indices[entry] - base) as usize;
        let mut k = lane;
        let mut sum = 0f32;
        while k < inner as usize {
            sum = fma(
                a[row * inner as usize + k],
                b[k * columns as usize + column],
                sum,
            );
            k += 32;
        }
        sum = plane_sum(sum);
        if lane == 0 {
            output[entry] = fma(beta, values[entry], alpha * sum);
        }
    }
}
