use half::{bf16, f16};
use ruda_driver_cuda::CudaRuntime;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch, address_type = "dynamic")]
fn half_ops<F: Float + RudaElement>(
    input: &Array<F>,
    rhs: &Array<F>,
    float_input: &Array<f32>,
    output: &mut Array<F>,
    widened: &mut Array<f32>,
    bias: F,
) {
    let i = ABSOLUTE_POS as usize;
    let tid = UNIT_POS as usize;
    let mut shared = SharedMemory::<F>::new(64usize);
    shared[tid] = input[i];
    sync_ruda();
    if i < input.len() {
        let a = input[i];
        let b = rhs[i];
        let n = input.len();
        output[i] = shared[63usize - tid];
        output[n + i] = a + b;
        output[n * 2 + i] = a - b;
        output[n * 3 + i] = a * b;
        output[n * 4 + i] = fma(a, b, bias);
        output[n * 5 + i] = select(a != b, a, F::new(1.25f32));
        output[n * 6 + i] = F::cast_from(float_input[i]);
        widened[i] = f32::cast_from(a);
    }
}

trait HalfTest: Float + RudaElement {
    fn raw(bits: u16) -> Self;
    fn bits(self) -> u16;
    fn wide(self) -> f64;
    fn rounded(value: f64) -> Self;
}
macro_rules! half_test {
    ($ty:ty, $fraction:expr, $bias:expr) => {
        impl HalfTest for $ty {
            fn raw(bits: u16) -> Self {
                Self::from_bits(bits)
            }
            fn bits(self) -> u16 {
                self.to_bits()
            }
            fn wide(self) -> f64 {
                self.to_f64()
            }
            fn rounded(value: f64) -> Self {
                Self::from_bits(round_binary64(value, $fraction, $bias))
            }
        }
    };
}
half_test!(f16, 10, 15);
half_test!(bf16, 7, 127);

fn round_binary64(value: f64, fraction: u32, bias: i32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = bits & ((1u64 << 52) - 1);
    let limit = bias * 2 + 1;
    let infinity = (limit as u16) << fraction;
    if exp == 0x7ff {
        return sign | infinity | if mantissa == 0 { 0 } else { 1 };
    }
    if exp == 0 {
        return sign;
    }
    let target_exp = exp - 1023 + bias;
    if target_exp >= limit {
        return sign | infinity;
    }
    let significand = mantissa | (1u64 << 52);
    let shift = 52 - fraction + (1 - target_exp).max(0) as u32;
    if shift > 53 {
        return sign;
    }
    let integer = significand >> shift;
    let remainder = significand & ((1u64 << shift) - 1);
    let midpoint = 1u64 << (shift - 1);
    let rounded =
        integer + u64::from(remainder > midpoint || (remainder == midpoint && integer & 1 != 0));
    if target_exp <= 0 {
        sign | rounded as u16
    } else {
        sign | (((target_exp as u16) << fraction) + rounded as u16 - (1 << fraction))
    }
}

fn equal<F: HalfTest>(actual: F, expected: F) -> bool {
    actual.bits() == expected.bits() || (actual.wide().is_nan() && expected.wide().is_nan())
}

fn check<F: HalfTest>(
    client: &ComputeClient<CudaRuntime>,
    backend: &str,
    address: AddressType,
    count: usize,
) {
    let input: Vec<F> = (0..count).map(|i| F::raw((i / 3) as u16)).collect();
    let rhs: Vec<F> = (0..count)
        .map(|i| F::raw((i as u16).wrapping_mul(40503).wrapping_add(17)))
        .collect();
    let floats: Vec<f32> = (0..count)
        .map(|i| {
            let a = input[i].wide();
            let b = F::raw(((i / 3) as u16).wrapping_add(1)).wide();
            if a.is_finite() && b.is_finite() {
                let mid = ((a + b) * 0.5) as f32;
                f32::from_bits(mid.to_bits().wrapping_add((i % 3) as u32).wrapping_sub(1))
            } else {
                f32::from_bits((i as u32).wrapping_mul(2654435761))
            }
        })
        .collect();
    let sentinel = F::rounded(247.0);
    let input_handle = client.create_from_slice(F::as_bytes(&input));
    let rhs_handle = client.create_from_slice(F::as_bytes(&rhs));
    let floats_handle = client.create_from_slice(f32::as_bytes(&floats));
    let output = client.create_from_slice(F::as_bytes(&vec![sentinel; count * 7 + 16]));
    let widened = client.create_from_slice(f32::as_bytes(&vec![247.0; count + 16]));
    // SAFETY: Checked global reads initialize all 64 shared lanes, including tail
    // threads; every thread reaches the barrier before the in-length output guard.
    unsafe {
        half_ops::launch::<F, CudaRuntime>(
            client,
            RudaCount::Static(count.div_ceil(64) as u32, 1, 1),
            RudaDim::new_1d(64),
            address,
            ArrayArg::from_raw_parts(input_handle, count),
            ArrayArg::from_raw_parts(rhs_handle, count),
            ArrayArg::from_raw_parts(floats_handle, count),
            ArrayArg::from_raw_parts(output.clone(), count * 7),
            ArrayArg::from_raw_parts(widened.clone(), count),
            F::rounded(-0.75),
        );
    }
    let bytes = client.read_one(output).unwrap();
    let actual = F::from_bytes(&bytes);
    let float_bytes = client.read_one(widened).unwrap();
    let actual_float = f32::from_bytes(&float_bytes);
    for i in 0..count {
        let reversed = i / 64 * 64 + 63 - i % 64;
        assert_eq!(
            actual[i].bits(),
            input
                .get(reversed)
                .copied()
                .unwrap_or_else(|| F::raw(0))
                .bits()
        );
        let a = input[i].wide();
        let b = rhs[i].wide();
        for (column, value) in [(1, a + b), (2, a - b), (3, a * b), (6, floats[i] as f64)] {
            assert!(
                equal(actual[column * count + i], F::rounded(value)),
                "{backend} {} {address:?} column={column} i={i} actual={:04x} expected={:04x}",
                core::any::type_name::<F>(),
                actual[column * count + i].bits(),
                F::rounded(value).bits()
            );
        }
        let selected = if a != b { input[i] } else { F::rounded(1.25) };
        assert_eq!(actual[count * 5 + i].bits(), selected.bits());
        assert!(
            actual_float[i].to_bits() == (a as f32).to_bits()
                || (actual_float[i].is_nan() && a.is_nan())
        );
    }
    assert!(
        actual[count * 7..]
            .iter()
            .all(|x| x.bits() == sentinel.bits())
    );
    assert!(actual_float[count..].iter().all(|&x| x == 247.0));
    let root = std::path::PathBuf::from(
        std::env::var("RUDA_HALF_REFERENCE").expect("explicit reference directory required"),
    );
    std::fs::create_dir_all(&root).unwrap();
    let name = core::any::type_name::<F>().rsplit("::").next().unwrap();
    let path = root.join(format!("{name}-{address:?}-{count}.bin"));
    if backend == "nvrtc" && !path.exists() {
        std::fs::write(&path, &*bytes).unwrap();
    } else {
        let reference_bytes =
            std::fs::read(&path).expect("NVRTC reference must be generated first");
        let reference = F::from_bytes(&reference_bytes);
        assert_eq!(actual.len(), reference.len());
        for (i, (&a, &b)) in actual.iter().zip(reference).enumerate() {
            assert!(
                equal(a, b),
                "NVRTC mismatch {name} {address:?} element={i} actual={:04x} reference={:04x}",
                a.bits(),
                b.bits()
            );
        }
    }
    println!(
        "PASS {backend} half {name} {address:?} elements={count} outputs={} CPU add/sub/mul/conversion and NVRTC FMA reference={}",
        count * 8,
        path.display()
    );
}

pub fn run(client: &ComputeClient<CudaRuntime>, backend: &str) {
    let count: usize = std::env::var("RUDA_HALF_TEST_COUNT")
        .map(|s| s.parse().unwrap())
        .unwrap_or(65536 * 3);
    assert!(count > 0 && count <= 65536 * 3);
    for address in [AddressType::U32, AddressType::U64] {
        check::<f16>(client, backend, address, count);
        check::<bf16>(client, backend, address, count);
    }
}
