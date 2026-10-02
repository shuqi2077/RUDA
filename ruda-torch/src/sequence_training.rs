use super::*;

#[ruda(launch)]
fn triangular<F: Float + RudaElement>(
    a: &Tensor<F>, b: &Tensor<F>, out: &mut Tensor<F>,
    n: u32, columns: u32, #[comptime] upper: bool, #[comptime] unit: bool,
) {
    let batch = RUDA_POS_Y as usize;
    let column = RUDA_POS_X as usize * 4 + UNIT_POS as usize / 32;
    let lane = UNIT_POS as usize % 32;
    let width = n as usize;
    let cols = columns as usize;
    for offset in 0..width {
        let mut row = offset;
        if comptime!(upper) { row = width - 1 - offset; }
        let mut sum = 0.0f32;
        let mut k = lane;
        while k < width {
            let mut used = k < row;
            if comptime!(upper) { used = k > row; }
            if used && column < cols {
                sum += f32::cast_from(a[(batch * width + row) * width + k])
                    * f32::cast_from(out[(batch * width + k) * cols + column]);
            }
            k += 32;
        }
        let dot = plane_sum(sum);
        if lane == 0 && column < cols {
            let mut result = f32::cast_from(b[(batch * width + row) * cols + column]) - dot;
            if comptime!(!unit) {
                result /= f32::cast_from(a[(batch * width + row) * width + row]);
            }
            out[(batch * width + row) * cols + column] = F::cast_from(result);
        }
        sync_ruda();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_sequence_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_delta_forward(
    descriptors: *const Descriptor, count: usize, scale: f32, chunk: u32,
) -> i32 {
    checked(|| {
        assert!(!descriptors.is_null() && count == 8 && scale.is_finite() && chunk > 0);
        let ds = unsafe { std::slice::from_raw_parts(descriptors, count) };
        let v: Vec<_> = ds.iter().map(|d| unsafe { View::read(d) }).collect();
        assert!(v.iter().all(|x| x.dtype == 0));
        let input = rudnn::gated_delta::GatedDeltaInput {
            query: primitives::tensor(&v[0]), key: primitives::tensor(&v[1]),
            value: primitives::tensor(&v[2]), beta: primitives::tensor(&v[3]),
            log_decay: primitives::tensor(&v[4]), initial_state: primitives::tensor(&v[5]), query_scale: scale,
        };
        let result = rudnn::gated_delta::chunk_gated_delta_rule(input, chunk as usize)
            .expect("RUDA gated-delta forward failed");
        assert_eq!(result.output.meta.shape()[..], v[6].shape[..]);
        assert_eq!(result.final_state.meta.shape()[..], v[7].shape[..]);
        primitives::store(result.output, &v[6]); primitives::store(result.final_state, &v[7]);
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_triangular_solve(
    descriptors: *const Descriptor, count: usize, upper: bool, unit: bool,
) -> i32 {
    checked(|| {
        assert!(!descriptors.is_null() && count == 3);
        let ds = unsafe { std::slice::from_raw_parts(descriptors, count) };
        let v: Vec<_> = ds.iter().map(|d| unsafe { View::read(d) }).collect();
        let (a, b, out) = (&v[0], &v[1], &v[2]);
        assert!(a.shape.len() >= 2 && a.dtype <= 2);
        assert_eq!(a.dtype, b.dtype); assert_eq!(a.dtype, out.dtype);
        let rank = a.shape.len(); let n = a.shape[rank - 1];
        assert_eq!(a.shape[rank - 2], n);
        assert_eq!(b.shape.len(), rank); assert_eq!(b.shape[rank - 2], n);
        assert_eq!(a.shape[..rank - 2], b.shape[..rank - 2]);
        assert_eq!(b.shape, out.shape);
        for view in &v {
            let mut stride = 1;
            for axis in (0..view.shape.len()).rev() {
                if view.shape[axis] > 1 { assert_eq!(view.strides[axis], stride); }
                stride *= view.shape[axis];
            }
        }
        if out.len == 0 { return; }
        let columns = b.shape[rank - 1];
        let batches = a.len / (n * n);
        assert!(a.len <= u32::MAX as usize && b.len <= u32::MAX as usize);
        let c = client();
        let grid = RudaCount::Static(u32::try_from(columns.div_ceil(4)).unwrap(), u32::try_from(batches).unwrap(), 1);
        macro_rules! launch {
            ($f:ty) => { unsafe { triangular::launch::<$f, CudaRuntime>(
                &c, grid, RudaDim::new_1d(128), a.arg(), b.arg(), out.arg(),
                n as u32, columns as u32, upper, unit) } };
        }
        match a.dtype { 0 => launch!(f32), 1 => launch!(f16), 2 => launch!(bf16), _ => unreachable!() }
        LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
    })
}
