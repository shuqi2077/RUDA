use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::dsl::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "ptx_runtime/tensor.rs"]
mod tensor;

#[path = "ptx_runtime/shared.rs"]
mod shared;

#[path = "ptx_runtime/half_precision.rs"]
mod half_precision;

#[path = "ptx_runtime/bitwise.rs"]
mod bitwise;

#[path = "ptx_runtime/trigonometry.rs"]
mod trigonometry;

#[path = "ptx_runtime/matrix.rs"]
mod matrix;

struct CompilationTrace;
static TRACE: CompilationTrace = CompilationTrace;
static COMPILED: AtomicUsize = AtomicUsize::new(0);
static CACHE_HITS: AtomicUsize = AtomicUsize::new(0);

impl log::Log for CompilationTrace {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record
            .target()
            .starts_with("ruda_driver_cuda::execution::context")
        {
            match record.args().to_string().as_str() {
                "Compiling kernel" => {
                    COMPILED.fetch_add(1, Ordering::Relaxed);
                }
                "Using PTX cache" => {
                    CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
        }
    }
    fn flush(&self) {}
}

#[ruda(launch)]
fn add(a: &Array<f32>, b: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

#[ruda(launch)]
fn unsupported_exp(a: &Array<f32>, output: &mut Array<f32>) {
    output[ABSOLUTE_POS] = a[ABSOLUTE_POS].exp();
}

fn main() {
    log::set_logger(&TRACE).unwrap();
    log::set_max_level(log::LevelFilter::Trace);
    if let Ok(path) = std::env::var("RUDA_PTX_TEST_CACHE") {
        use ruda::runtime::config::{RudaRuntimeConfig, RuntimeConfig, cache::CacheConfig};
        let mut config = RudaRuntimeConfig::from_current_dir().override_from_env();
        config.compilation.cache = Some(CacheConfig::File(path.into()));
        RudaRuntimeConfig::set(config);
    }
    let client = CudaRuntime::client(&CudaDevice::default());
    let backend = std::env::var("RUDA_CUDA_COMPILER").unwrap_or_else(|_| "nvrtc".into());
    if std::env::args().any(|arg| arg == "--matrix") {
        matrix::run(&client, &backend);
        return;
    }
    if std::env::args().any(|arg| arg == "--trigonometry") {
        trigonometry::run(&client, &backend);
        return;
    }
    if std::env::args().any(|arg| arg == "--autotune-input-error") {
        trigonometry::autotune_input_error(&client, &backend);
        return;
    }
    if std::env::args().any(|arg| arg == "--shared-over-limit") {
        shared::over_limit(&client, &backend);
        return;
    }
    if std::env::args().any(|arg| arg == "--unsupported") {
        let a = client.create_from_slice(f32::as_bytes(&[1.0]));
        let output = client.empty(4);
        // SAFETY: Each argument has one f32 and exactly one thread is launched.
        unsafe {
            unsupported_exp::launch::<CudaRuntime>(
                &client,
                RudaCount::Static(1, 1, 1),
                RudaDim::new_1d(1),
                ArrayArg::from_raw_parts(a, 1),
                ArrayArg::from_raw_parts(output.clone(), 1),
            )
        };
        assert_eq!(backend, "ptx");
        let error = client
            .read_one(output)
            .expect_err("unsupported PTX must not fall back to NVRTC");
        assert!(format!("{error:?}").contains("Direct PTX"), "{error:?}");
        println!("PASS direct PTX rejects unsupported IR without fallback");
        return;
    }
    for count in [1, 63, 64, 65, 257] {
        let a: Vec<f32> = (0..count).map(|i| i as f32 * 0.25).collect();
        let b: Vec<f32> = (0..count).map(|i| -(i as f32) * 0.5).collect();
        let a_handle = client.create_from_slice(f32::as_bytes(&a));
        let b_handle = client.create_from_slice(f32::as_bytes(&b));
        let output = client.create_from_slice(f32::as_bytes(&vec![247.0; count + 16]));
        for repetition in 0..2 {
            // SAFETY: Each binding contains at least `count` f32 elements.
            // Checked launch and the kernel's length guard cover tail threads.
            unsafe {
                add::launch::<CudaRuntime>(
                    &client,
                    RudaCount::Static(count.div_ceil(64) as u32, 1, 1),
                    RudaDim::new_1d(64),
                    ArrayArg::from_raw_parts(a_handle.clone(), count),
                    ArrayArg::from_raw_parts(b_handle.clone(), count),
                    ArrayArg::from_raw_parts(output.clone(), count),
                );
            }
            let bytes = client.read_one(output.clone()).unwrap();
            let actual = f32::from_bytes(&bytes);
            for i in 0..count {
                assert_eq!(actual[i].to_bits(), (a[i] + b[i]).to_bits());
            }
            assert!(actual[count..count + 16].iter().all(|&v| v == 247.0));
            println!("PASS {backend} runtime count={count} repetition={repetition}");
        }
    }
    if std::env::args().any(|arg| arg == "--tensor") {
        tensor::run(&client, &backend);
    }
    if std::env::args().any(|arg| arg == "--shared") {
        shared::run(&client, &backend);
    }
    if std::env::args().any(|arg| arg == "--half") {
        half_precision::run(&client, &backend);
    }
    if std::env::args().any(|arg| arg == "--bitwise") {
        bitwise::run(&client, &backend);
    }
    let compiled = COMPILED.load(Ordering::Relaxed);
    let hits = CACHE_HITS.load(Ordering::Relaxed);
    if std::env::args().any(|arg| arg == "--expect-cold") {
        assert!(
            compiled > 0,
            "cold run must compile rather than reuse another backend's cache"
        );
        assert_eq!(hits, 0);
    }
    if std::env::args().any(|arg| arg == "--expect-warm") {
        assert_eq!(compiled, 0, "warm process must not recompile");
        assert!(
            hits > 0,
            "warm process must actually load its persistent cache"
        );
    }
    println!("CACHE {backend} compiled={compiled} disk_hits={hits}");
}
