use super::*;

impl<R: Runtime> CsrTensor<R> {
    /// Transpose the host index pattern and gather current values on the device.
    pub fn transpose(&self) -> Result<Self, SparseError> {
        self.transpose_impl(false).map(|(matrix, _)| matrix)
    }

    pub fn transpose_with_permutation(&self) -> Result<(Self, Vec<u32>), SparseError> {
        self.transpose_impl(true)
    }

    fn transpose_impl(&self, capture_permutation: bool) -> Result<(Self, Vec<u32>), SparseError> {
        let base = self.base.value();
        let mut offsets = vec![0u32; self.columns + 1];
        for &column in self.host_indices.iter() {
            offsets[(column - base) as usize + 1] += 1;
        }
        offsets[0] = base;
        for row in 0..self.columns {
            offsets[row + 1] += offsets[row];
        }
        let mut cursors = offsets[..self.columns].to_vec();
        let mut indices = vec![0u32; self.nnz];
        let mut permutation = vec![0u32; self.nnz];
        for row in 0..self.rows {
            let start = (self.host_offsets[row] - base) as usize;
            let end = (self.host_offsets[row + 1] - base) as usize;
            for entry in start..end {
                let column = (self.host_indices[entry] - base) as usize;
                let destination = (cursors[column] - base) as usize;
                indices[destination] = row as u32 + base;
                permutation[destination] = entry as u32;
                cursors[column] += 1;
            }
        }
        self.grid(1)?;
        let grid = RudaCount::Static((self.nnz as u32).div_ceil(128), 1, 1);
        let output = self.from_pattern(self.columns, self.rows, offsets, indices)?;
        let captured = if capture_permutation { permutation.clone() } else { Vec::new() };
        if self.nnz == 0 {
            return Ok((output, captured));
        }
        let permutation: RudaTensor<R> = from_data(
            TensorData::new(permutation, [self.nnz]),
            &self.values.device,
        );
        gather::launch::<R>(
            &self.values.client,
            grid,
            RudaDim::new_1d(128),
            permutation.into_array_arg(),
            self.values.clone().into_array_arg(),
            output.values.clone().into_array_arg(),
            self.nnz as u32,
            include_str!("transpose.rs").to_owned(),
        );
        Ok((output, captured))
    }
}

#[ruda(launch)]
fn gather(
    permutation: &Array<u32>,
    values: &Array<f32>,
    output: &mut Array<f32>,
    nnz: u32,
    #[comptime] _source: String,
) {
    let entry = ABSOLUTE_POS;
    if entry < nnz as usize {
        output[entry] = values[permutation[entry] as usize];
    }
}
