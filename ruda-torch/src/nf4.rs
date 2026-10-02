//! Optional NF4 decoder and fused FP16/BF16 Tensor Core GEMM. The fused path
//! decodes into shared-memory tiles, never a persistent dense weight.
use super::*;

#[ruda(launch)]
fn decode<F: Float + RudaElement>(
    packed: &Tensor<u8>, scales: &Tensor<f32>, table: &Tensor<f32>,
    output: &mut Tensor<F>, start: u32, block: u32,
) {
    let position = ABSOLUTE_POS as u32;
    if (position as usize) < output.len() {
        let index = position + start;
        let byte = u32::cast_from(packed[(index / 2) as usize]);
        let mut code = byte & 15;
        if index % 2 == 0 { code = byte >> 4; }
        output[position as usize] = F::cast_from(table[code as usize] * scales[(index / block) as usize]);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_nf4_api_version() -> u32 { 1 }

#[ruda(launch)]
fn gemm<F: Float + RudaElement, O: Float + RudaElement>(
    input: &Tensor<F>, packed: &Tensor<u8>, scales: &Tensor<f32>, table: &Tensor<f32>,
    output: &mut Tensor<O>, rows: u32, columns: u32,
    width: u32, block: u32, #[comptime] backward: bool,
) {
    let lane = UNIT_POS as usize;
    let row_base = RUDA_POS_Y as usize * 16;
    let column_base = RUDA_POS_X as usize * 16;
    let k = width as usize;
    let n = columns as usize;
    let mut inner = k;
    let mut cols = n;
    if comptime!(backward) { inner = n; cols = k; }
    let mut left = SharedMemory::<F>::new_aligned(256usize, 32usize);
    let mut right = SharedMemory::<F>::new_aligned(256usize, 32usize);
    let mut result = SharedMemory::<f32>::new_aligned(256usize, 32usize);
    let acc = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator,
        16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let mut base = 0usize;
    while base < inner {
        #[unroll]
        for i in 0usize..8usize {
            let t = lane + i * 32;
            let tr = t / 16;
            let tc = t % 16;
            let row = row_base + tr;
            let source_col = base + tc;
            let mut a = F::cast_from(0.0f32);
            if row < rows as usize && source_col < inner { a = input[row * inner + source_col]; }
            left[t] = a;
            let mut wr = column_base + tr;
            let mut wc = base + tc;
            if comptime!(backward) { wr = base + tr; wc = column_base + tc; }
            let mut value = F::cast_from(0.0f32);
            if wr < n && wc < k {
                let index = wr * k + wc;
                let byte = u32::cast_from(packed[index / 2]);
                let mut code = byte & 15;
                if index % 2 == 0 { code = byte >> 4; }
                value = F::cast_from(table[code as usize] * scales[index / block as usize]);
            }
            if comptime!(backward) { right[tc * 16 + tr] = value; }
            else { right[t] = value; }
        }
        sync_ruda();
        let a = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A,
            16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &left.to_slice(), 16);
        let b = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B,
            16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor, &right.to_slice(), 16);
        cmma::execute::<F,F,f32,f32>(&a, &b, &acc, &acc);
        sync_ruda(); base += 16;
    }
    cmma::store(&mut result.to_slice_mut(), &acc, 16, cmma::MatrixLayout::RowMajor);
    sync_ruda();
    #[unroll]
    for i in 0usize..8usize {
        let t = lane + i * 32;
        let row = row_base + t / 16;
        let column = column_base + t % 16;
        if row < rows as usize && column < cols {
            output[row * cols + column] = O::cast_from(result[t]);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_nf4_matmul_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_nf4_matmul(
    descriptors: *const Descriptor, count: usize, columns: u32, width: u32, block: u32, backward: bool,
) -> i32 {
    checked(|| {
        assert!(!descriptors.is_null() && count == 5 && columns > 0 && width > 0 && block > 0 && block % 2 == 0);
        let ds = unsafe { std::slice::from_raw_parts(descriptors, count) };
        let v: Vec<_> = ds.iter().map(|d| unsafe { View::read(d) }).collect();
        let (input, packed, scales, table, out) = (&v[0], &v[1], &v[2], &v[3], &v[4]);
        assert!(input.dtype == 1 || input.dtype == 2);
        assert_eq!((packed.dtype, scales.dtype, table.dtype, table.len), (8, 0, 0, 16));
        let size = (columns as usize).checked_mul(width as usize).expect("NF4 matrix size overflow");
        assert!(size <= u32::MAX as usize && packed.len == size.div_ceil(2) && scales.len == size.div_ceil(block as usize));
        assert_eq!(input.shape.len(), 2); assert_eq!(out.shape.len(), 2);
        let inner = if backward { columns } else { width } as usize;
        let cols = if backward { width } else { columns } as usize;
        assert_eq!(input.shape[1], inner); assert_eq!(out.shape, vec![input.shape[0], cols]);
        assert_eq!(out.dtype, if backward { 0 } else { input.dtype });
        for view in &v {
            let mut stride = 1;
            for axis in (0..view.shape.len()).rev() {
                if view.shape[axis] > 1 { assert_eq!(view.strides[axis], stride); }
                stride *= view.shape[axis];
            }
            assert!(view.len <= u32::MAX as usize);
        }
        if input.shape[0] == 0 { return; }
        let c = client();
        let grid = RudaCount::Static(u32::try_from(cols.div_ceil(16)).unwrap(), u32::try_from(input.shape[0].div_ceil(16)).unwrap(), 1);
        macro_rules! launch {
            ($f:ty, $o:ty) => { unsafe { gemm::launch::<$f, $o, CudaRuntime>(
                &c, grid, RudaDim::new_1d(32), input.arg(), packed.arg(), scales.arg(), table.arg(),
                out.arg(), input.shape[0] as u32, columns, width, block, backward) } };
        }
        match (input.dtype, backward) {
            (1, false) => launch!(f16, f16), (1, true) => launch!(f16, f32),
            (2, false) => launch!(bf16, bf16), (2, true) => launch!(bf16, f32), _ => unreachable!()
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed); finish_dispatch(&c);
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_nf4_decode(
    descriptors: *const Descriptor, count: usize, start: u32, block: u32,
) -> i32 {
    checked(|| {
        assert!(!descriptors.is_null());
        assert_eq!(count, 4);
        assert!(block > 0 && block % 2 == 0);
        let ds = unsafe { std::slice::from_raw_parts(descriptors, count) };
        let views: Vec<_> = ds.iter().map(|d| unsafe { View::read(d) }).collect();
        let (packed, scales, table, output) = (&views[0], &views[1], &views[2], &views[3]);
        assert_eq!((packed.dtype, scales.dtype, table.dtype), (8, 0, 0));
        assert!(output.dtype <= 2 && output.len > 0);
        assert_eq!(table.len, 16);
        for v in &views {
            let mut stride = 1;
            for axis in (0..v.shape.len()).rev() {
                assert_eq!(v.strides[axis], stride);
                stride *= v.shape[axis];
            }
        }
        let end = (start as usize).checked_add(output.len).expect("NF4 index overflow");
        assert!(end <= u32::MAX as usize);
        assert!(end.div_ceil(2) <= packed.len && end.div_ceil(block as usize) <= scales.len);
        let c = client();
        let grid = RudaCount::Static(u32::try_from(output.len.div_ceil(128)).unwrap(), 1, 1);
        macro_rules! launch {
            ($f:ty) => { unsafe { decode::launch::<$f, CudaRuntime>(
                &c, grid, RudaDim::new_1d(128), packed.arg(), scales.arg(),
                table.arg(), output.arg(), start, block) } };
        }
        match output.dtype {
            0 => launch!(f32), 1 => launch!(f16), 2 => launch!(bf16),
            _ => unreachable!(),
        }
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        finish_dispatch(&c);
    })
}
