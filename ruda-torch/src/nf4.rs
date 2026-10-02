//! Optional NF4 tile decoder. The matrix product reuses the existing ruBLAS
//! bridge; only a bounded row tile is decoded, never a persistent dense weight.
use super::*;

#[ruda(launch)]
fn decode<F: Float + RudaElement>(
    packed: &Tensor<u8>, scales: &Tensor<f32>, table: &Tensor<f32>,
    output: &mut Tensor<F>, start: u32, block: u32,
) {
    let position = ABSOLUTE_POS;
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
