use super::{GatedDeltaError, GatedDeltaInput, GatedDeltaOutput, chunk_kernel as kernel, validate};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::{RudaCount, RudaDim}},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous,
        permutation::swap_dims, reshape::reshape},
};
use rublas::tensor_matmul::{MatmulStrategy, matmul};

pub(super) fn product<R: Runtime>(a: RudaTensor<R>, b: RudaTensor<R>) -> Result<RudaTensor<R>, GatedDeltaError> {
    let shape = ruda_core::tensor::calculate_matmul_output(a.meta.shape(), b.meta.shape())
        .map_err(|_| GatedDeltaError("incompatible chunk matrix shapes"))?;
    let out = empty_device_contiguous_dtype(a.client.clone(), a.device.clone(), shape, DType::F32);
    // Elementwise consumers use flat contiguous arrays. Keep FP32 products on
    // the FP32 register path rather than a reduced-precision tensor-core path.
    matmul(a, b, Some(out), MatmulStrategy::Naive, DType::F32)
        .map_err(|_| GatedDeltaError("chunk gated delta FP32 matmul setup failed"))
}

pub(super) fn prefix_tile(heads: usize, sequence: usize, chunk: usize) -> usize {
    let rows = heads * sequence.div_ceil(chunk);
    let log_chunk = (usize::BITS - (chunk - 1).leading_zeros()) as i32;
    let log_rows = (usize::BITS - (rows - 1).leading_zeros()) as i32;
    let log_threads = ((9 + log_chunk - log_rows) / 2).clamp(4, 9);
    (2usize << log_threads).min(chunk.next_power_of_two())
}

/// Chunked gated delta prefill with FP32 intermediate matrices and state.
/// Layout, normalization and ownership follow `gated_delta_rule`. The last
/// chunk is zero padded; output excludes padding. `chunk_size` must be positive
/// and its triangular workspace must fit the device's per-block shared memory.
pub fn chunk_gated_delta_rule<R: Runtime>(
    input: GatedDeltaInput<R>, chunk_size: usize,
) -> Result<GatedDeltaOutput<R>, GatedDeltaError> {
    chunk_impl(input, chunk_size, #[cfg(test)] None)
}

pub(super) fn chunk_impl<R: Runtime>(
    input: GatedDeltaInput<R>, chunk_size: usize,
    #[cfg(test)] mut observe: Option<&mut dyn FnMut(usize, &str, RudaTensor<R>)>,
) -> Result<GatedDeltaOutput<R>, GatedDeltaError> {
    let [batch, heads, sequence, kd, vd] = validate(&input)?;
    let h = batch * heads;
    let c = chunk_size;
    if c == 0 || c > u32::MAX as usize {
        return Err(GatedDeltaError("chunk size must be positive and fit U32 indexing"));
    }
    let shared = c.checked_mul(c).and_then(|n| n.checked_add(c)).and_then(|n| n.checked_mul(4));
    if shared.is_none_or(|n| n > input.query.client.properties().hardware.max_shared_memory_size as usize) {
        return Err(GatedDeltaError("chunk triangular workspace exceeds device shared memory"));
    }
    for width in [c, kd, vd] {
        if h.checked_mul(c).and_then(|n| n.checked_mul(width)).is_none_or(|n| n > u32::MAX as usize) {
            return Err(GatedDeltaError("chunk workspace exceeds U32 indexing"));
        }
    }
    if sequence == 0 { return super::gated_delta_rule(input); }
    let client = input.query.client.clone();
    let device = input.query.device.clone();
    let dtype = input.query.dtype;
    let allocate = |shape: Shape, dtype| empty_device_contiguous_dtype(client.clone(), device.clone(), shape, dtype);
    let matrix = |rows, cols| allocate([h, rows, cols].into(), DType::F32);
    let launch = |elements| {
        let dim = RudaDim::new(client.properties(), elements);
        (calculate_ruda_count_elemwise(&client, elements, dim), dim)
    };
    let source = || include_str!("chunk_kernel.rs").to_owned();
    let q = into_contiguous(input.query);
    let k = into_contiguous(input.key);
    let v = into_contiguous(input.value);
    let beta = into_contiguous(input.beta);
    let decay = into_contiguous(input.log_decay);
    let mut state = reshape(into_contiguous(input.initial_state), [h, kd, vd].into());
    let output = allocate([batch, heads, sequence, vd].into(), dtype);
    let tile = prefix_tile(h, sequence, c);
    for start in (0..sequence).step_by(c) {
        let cumulative = matrix(c, 1);
        kernel::prefix::launch::<R>(&client, RudaCount::Static(h as u32, 1, 1),
            RudaDim::new_1d((tile / 2).max(1) as u32),
            decay.clone().into_array_arg(), cumulative.clone().into_array_arg(),
            sequence as u32, start as u32, c, tile, source());
        let prepare = |tensor: &RudaTensor<R>, width, scale, mode| {
            let result = matrix(c, width);
            let (count, dim) = launch(h * c * width);
            kernel::prepare::launch::<R>(&client, count, dim,
                tensor.clone().into_array_arg(), beta.clone().into_array_arg(), cumulative.clone().into_array_arg(),
                result.clone().into_array_arg(), sequence as u32, start as u32, c as u32, width as u32,
                scale, mode, source(), dtype.into());
            result
        };
        let mask = |products: RudaTensor<R>, strict| {
            let result = matrix(c, c);
            let (count, dim) = launch(h * c * c);
            kernel::decay_mask::launch::<R>(&client, count, dim,
                products.into_array_arg(), cumulative.clone().into_array_arg(), result.clone().into_array_arg(),
                c as u32, strict, source());
            result
        };
        let query = prepare(&q, kd, input.query_scale, 0);
        let key = prepare(&k, kd, 1.0, 0);
        let key_beta = prepare(&k, kd, 1.0, 1);
        let value_beta = prepare(&v, vd, 1.0, 1);
        #[cfg(test)]
        if let Some(observe) = &mut observe {
            for (name, tensor) in [("query", &query), ("key", &key), ("key-beta", &key_beta),
                ("value-beta", &value_beta), ("cumulative", &cumulative)] {
                observe(start / c, name, tensor.clone());
            }
        }
        let triangular = mask(product(key_beta, swap_dims(key.clone(), 1, 2))?, true);
        let inverse = matrix(c, c);
        let plane = client.properties().hardware.plane_size_max;
        kernel::triangular_inverse::launch::<R>(&client, RudaCount::Static(h as u32, 1, 1),
            RudaDim::new_1d(plane), triangular.into_array_arg(), inverse.clone().into_array_arg(), c, source());
        #[cfg(test)]
        if let Some(observe) = &mut observe { observe(start / c, "inverse", inverse.clone()); }
        let value = product(inverse.clone(), value_beta)?;
        let key_cumulative = product(inverse, prepare(&k, kd, 1.0, 2))?;
        #[cfg(test)]
        if let Some(observe) = &mut observe {
            observe(start / c, "value", value.clone());
            observe(start / c, "key-cumulative", key_cumulative.clone());
        }
        let prediction = product(key_cumulative, state.clone())?;
        #[cfg(test)]
        if let Some(observe) = &mut observe { observe(start / c, "prediction", prediction.clone()); }
        let residual = matrix(c, vd);
        let (count, dim) = launch(h * c * vd);
        kernel::subtract::launch::<R>(&client, count.clone(), dim, value.into_array_arg(),
            prediction.into_array_arg(), residual.clone().into_array_arg(), source());
        let attention = mask(product(query, swap_dims(key, 1, 2))?, false);
        let inter = product(prepare(&q, kd, input.query_scale, 3), state.clone())?;
        #[cfg(test)]
        if let Some(observe) = &mut observe {
            for (name, tensor) in [("attention", &attention), ("inter", &inter), ("residual", &residual)] {
                observe(start / c, name, tensor.clone());
            }
        }
        let intra = product(attention, residual.clone())?;
        kernel::write_output::launch::<R>(&client, count, dim,
            inter.into_array_arg(), intra.into_array_arg(), output.clone().into_array_arg(),
            sequence as u32, start as u32, c as u32, vd as u32, source(), dtype.into());
        let update = product(swap_dims(prepare(&k, kd, 1.0, 4), 1, 2), residual)?;
        let next = matrix(kd, vd);
        let (count, dim) = launch(h * kd * vd);
        kernel::update_state::launch::<R>(&client, count, dim, state.into_array_arg(),
            update.into_array_arg(), cumulative.into_array_arg(), next.clone().into_array_arg(),
            c as u32, (kd * vd) as u32, source());
        state = next;
    }
    Ok(GatedDeltaOutput { output, final_state: reshape(state, [batch, heads, kd, vd].into()) })
}
