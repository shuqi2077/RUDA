use half::{bf16, f16};
use ruda_driver_cuda::CudaRuntime;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch, address_type = "dynamic")]
fn trig<F: Float + RudaElement>(input: &Array<F>, sine: &mut Array<F>, cosine: &mut Array<F>) {
    let i = ABSOLUTE_POS as usize;
    if i < input.len() {
        sine[i] = input[i].sin();
        cosine[i] = input[i].cos();
    }
}

trait TrigTest: Float + RudaElement {
    const SIGN: u32;
    const ULP: u32;
    fn bits(self) -> u32;
    fn wide(self) -> f64;
    fn rounded(x: f64) -> Self;
    fn inputs() -> Vec<Self>;
}
impl TrigTest for f32 {
    const SIGN: u32 = 1 << 31;
    const ULP: u32 = 2;
    fn bits(self) -> u32 { self.to_bits() }
    fn wide(self) -> f64 { self as f64 }
    fn rounded(x: f64) -> Self { x as f32 }
    fn inputs() -> Vec<Self> {
        let mut bits = vec![0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
            0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001];
        for x in [2f32.powi(-12), 120.0, core::f32::consts::FRAC_PI_4] {
            for offset in -8i32..=8 { for sign in [0, 1 << 31] {
                bits.push(x.to_bits().wrapping_add_signed(offset) ^ sign);
            }}
        }
        for exponent in -126..=127 {
            let x = 2f32.powi(exponent);
            for offset in -2i32..=2 { for sign in [0, 1 << 31] {
                bits.push(x.to_bits().wrapping_add_signed(offset) ^ sign);
            }}
        }
        for n in -4096..=4096 {
            let x = (n as f64 * core::f64::consts::FRAC_PI_2) as f32;
            for offset in -2i32..=2 { bits.push(x.to_bits().wrapping_add_signed(offset)); }
        }
        let mut seed = 0x51c05f32u32;
        for _ in 0..262144 {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            bits.push(seed);
        }
        bits.into_iter().map(f32::from_bits).collect()
    }
}
macro_rules! half_test {
    ($ty:ty) => {
        impl TrigTest for $ty {
            const SIGN: u32 = 1 << 15;
            const ULP: u32 = 1;
            fn bits(self) -> u32 { self.to_bits() as u32 }
            fn wide(self) -> f64 { self.to_f64() }
            fn rounded(x: f64) -> Self { Self::from_f64(x) }
            fn inputs() -> Vec<Self> { (0..=u16::MAX).map(Self::from_bits).collect() }
        }
    };
}
half_test!(f16);
half_test!(bf16);

fn ordered<F: TrigTest>(x: F) -> u32 {
    let bits = x.bits();
    if bits & F::SIGN != 0 { !bits & (F::SIGN | (F::SIGN - 1)) } else { bits | F::SIGN }
}

fn check<F: TrigTest>(client: &ComputeClient<CudaRuntime>, backend: &str, address: AddressType) {
    let inputs = F::inputs();
    let count = inputs.len();
    let input = client.create_from_slice(F::as_bytes(&inputs));
    let sentinel = F::rounded(247.0);
    let sine = client.create_from_slice(F::as_bytes(&vec![sentinel; count + 16]));
    let cosine = client.create_from_slice(F::as_bytes(&vec![sentinel; count + 16]));
    // SAFETY: All buffers cover count elements, and checked launches guard the tail.
    unsafe {
        trig::launch::<F, CudaRuntime>(client, RudaCount::Static(count.div_ceil(128) as u32, 1, 1),
            RudaDim::new_1d(128), address, ArrayArg::from_raw_parts(input, count),
            ArrayArg::from_raw_parts(sine.clone(), count), ArrayArg::from_raw_parts(cosine.clone(), count));
    }
    for (name, output, reference) in [("sin", sine, f64::sin as fn(f64)->f64), ("cos", cosine, f64::cos)] {
        let bytes = client.read_one(output).unwrap();
        let actual = F::from_bytes(&bytes);
        let mut max_ulp = 0;
        for (i, (&input, &actual)) in inputs.iter().zip(actual).enumerate() {
            let expected = F::rounded(reference(input.wide()));
            if expected.wide().is_nan() {
                assert!(actual.wide().is_nan(), "{backend} {name} NaN i={i}");
            } else {
                assert!(actual.wide().is_finite(), "{backend} {name} nonfinite i={i}");
                if input.wide() == 0.0 || expected.wide() == 0.0 {
                    assert_eq!(actual.bits(), expected.bits(), "{backend} {name} signed zero i={i}");
                }
                let ulp = ordered(actual).abs_diff(ordered(expected));
                max_ulp = max_ulp.max(ulp);
                assert!(ulp <= F::ULP, "{backend} {name} {} {address:?} i={i} input={:x} actual={:x} expected={:x} ulp={ulp}",
                    core::any::type_name::<F>(), input.bits(), actual.bits(), expected.bits());
            }
        }
        assert!(actual[count..].iter().all(|x| x.bits() == sentinel.bits()));
        println!("PASS {backend} {name} {} {address:?} count={count} max_ulp={max_ulp}", core::any::type_name::<F>());
    }
}

pub fn run(client: &ComputeClient<CudaRuntime>, backend: &str) {
    for address in [AddressType::U32, AddressType::U64] {
        check::<f32>(client, backend, address);
        check::<f16>(client, backend, address);
        check::<bf16>(client, backend, address);
    }
}

#[ruda(launch)]
fn unsupported_input(input: &Array<f64>, output: &mut Array<f64>) {
    output[ABSOLUTE_POS] = input[ABSOLUTE_POS].exp();
}

pub fn autotune_input_error(client: &ComputeClient<CudaRuntime>, backend: &str) {
    use ruda::runtime::tune::{LocalTuner, Tunable, TunableSet};
    use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    assert_eq!(backend, "ptx");
    let input = client.create_from_slice(f64::as_bytes(&[1.0]));
    let output = client.empty(8);
    // SAFETY: Each buffer holds one f64, with exactly one participating thread.
    unsafe {
        unsupported_input::launch::<CudaRuntime>(client, RudaCount::Static(1, 1, 1), RudaDim::new_1d(1),
            ArrayArg::from_raw_parts(input, 1), ArrayArg::from_raw_parts(output, 1));
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let mut set = TunableSet::<String, (), ()>::new_cloning_inputs(|_: &()| "pending-input-error".into());
    for name in ["first", "second"] {
        let calls = calls.clone();
        set = set.with(Tunable::new(name, move |()| { calls.fetch_add(1, Ordering::SeqCst); Ok::<(), String>(()) }));
    }
    let tuner = LocalTuner::<String, String>::new("ptx-pending-input-error-regression");
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tuner.execute(&"cuda".into(), client, Arc::new(set), ());
    })).expect_err("Autotuning must not erase an earlier input kernel failure");
    let message = failure.downcast_ref::<String>().map(String::as_str)
        .or_else(|| failure.downcast_ref::<&str>().copied()).unwrap_or("");
    assert!(message.contains("pre-existing stream errors") && message.contains("Direct PTX"), "{message}");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    println!("PASS direct PTX input compilation failure reaches caller before any autotune candidate");
}
