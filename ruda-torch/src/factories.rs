use super::*;

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_factory_api_version() -> u32 { 1 }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_arange(out: *const Descriptor, start: u64, step: u64) -> i32 {
    checked(|| {
        assert!(!out.is_null());
        let out = unsafe { View::read(&*out) };
        assert!(out.dtype != 3, "arange does not support bool storage");
        if out.len == 0 { return; }
        let client = client();
        let count = u32::try_from(out.len.div_ceil(128)).expect("arange grid overflow");
        macro_rules! run {
            ($o:ty, $a:ty, $start:expr, $step:expr) => {
                unsafe { kernels::arange::launch::<$o, $a, CudaRuntime>(
                    &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                    out.arg(), $start, $step) }
            };
        }
        let float_start = f32::from_bits(start as u32);
        let float_step = f32::from_bits(step as u32);
        match out.dtype {
            0 => run!(f32, f32, float_start, float_step),
            1 => run!(f16, f32, float_start, float_step),
            2 => run!(bf16, f32, float_start, float_step),
            4 => run!(i64, i64, start as i64, step as i64),
            5 => run!(i32, i64, start as i64, step as i64),
            6 => run!(i16, i64, start as i64, step as i64),
            7 => run!(i8, i64, start as i64, step as i64),
            8 => run!(u8, i64, start as i64, step as i64),
            _ => panic!("unsupported arange dtype"),
        }
        finish_dispatch(&client);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_normal(
    out: *const Descriptor, mean: f32, std: f32, seeds: *const u32,
) -> i32 {
    checked(|| {
        assert!(!out.is_null() && !seeds.is_null());
        let out = unsafe { View::read(&*out) };
        assert!(out.dtype <= 2 && std >= 0.0, "normal requires floating storage and nonnegative std");
        if out.len == 0 { return; }
        let seeds: [u32; 4] = unsafe { std::slice::from_raw_parts(seeds, 4) }.try_into().unwrap();
        let client = client();
        let tensor = primitives::tensor(&out);
        let dtype = tensor.dtype.into();
        rurand::random_normal_seeded(&client, mean, std, tensor.binding(), dtype, seeds)
            .expect("RUDA normal launch failed");
        finish_dispatch(&client);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    })
}
