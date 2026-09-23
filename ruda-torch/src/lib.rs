use ruda_core::future::block_on;
use half::{bf16, f16};
use ruda::runtime::server::Handle;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::dsl::prelude::*;
use std::{cell::RefCell, ffi::CString, panic::AssertUnwindSafe, sync::{OnceLock, atomic::{AtomicU64, Ordering}}};

mod streams;
mod paged;
mod kernels;
mod matmul;
mod pointwise;
mod softmax;
mod normalization;
mod primitives;
mod spatial;

thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::default()); }
static LAUNCHES: AtomicU64 = AtomicU64::new(0);
static UPLOAD: AtomicU64 = AtomicU64::new(0);
static DOWNLOAD: AtomicU64 = AtomicU64::new(0);
// Cumulative dispatch/allocation counters, not peak memory or timing measurements.
static RUBLAS_CALLS: AtomicU64 = AtomicU64::new(0);
static SCALAR_MATMUL_CALLS: AtomicU64 = AtomicU64::new(0);
static DIRECT_POINTWISE_CALLS: AtomicU64 = AtomicU64::new(0);
static ADDMM_EPILOGUES: AtomicU64 = AtomicU64::new(0);
static ADDMM_WORKSPACE_BYTES: AtomicU64 = AtomicU64::new(0);
static LEGACY_FP32_TEMP_BYTES: AtomicU64 = AtomicU64::new(0);
static WARP_SOFTMAX_CALLS: AtomicU64 = AtomicU64::new(0);
static SCALAR_SOFTMAX_CALLS: AtomicU64 = AtomicU64::new(0);
static FUSED_LAYER_NORM_CALLS: AtomicU64 = AtomicU64::new(0);
static FUSED_RMS_NORM_CALLS: AtomicU64 = AtomicU64::new(0);
static ASYNC_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static TRUSTED_INDEX_CALLS: AtomicU64 = AtomicU64::new(0);
static STORAGE_REDUCTION_CALLS: AtomicU64 = AtomicU64::new(0);
static WARP_REDUCTION_CALLS: AtomicU64 = AtomicU64::new(0);

pub struct Allocation { handle: Handle, bytes: usize }

#[repr(C)]
pub struct Descriptor {
    allocation: *const Allocation,
    offset_bytes: usize,
    rank: usize,
    shape: *const usize,
    strides: *const usize,
    dtype: u32,
}

struct View { handle: Handle, shape: Vec<usize>, strides: Vec<usize>, len: usize, dtype: u32 }

fn element_bytes(dtype: u32) -> usize {
    match dtype { 0 | 5 => 4, 1 | 2 | 6 => 2, 3 | 7 | 8 => 1, 4 => 8, _ => panic!("unsupported RUDA dtype") }
}

impl View {
    // The C++ adapter owns the descriptor arrays and allocation for the entire call.
    unsafe fn read(desc: &Descriptor) -> Self {
        assert!(!desc.allocation.is_null());
        let allocation = unsafe { &*desc.allocation };
        let (shape, strides) = if desc.rank == 0 {
            (vec![1], vec![1])
        } else {
            assert!(!desc.shape.is_null() && !desc.strides.is_null());
            unsafe { (std::slice::from_raw_parts(desc.shape, desc.rank).to_vec(),
                      std::slice::from_raw_parts(desc.strides, desc.rank).to_vec()) }
        };
        let len = shape.iter().try_fold(1usize, |a, b| a.checked_mul(*b)).expect("shape overflow");
        let span = if len == 0 { 0 } else {
            shape.iter().zip(&strides).try_fold(1usize, |n, (&d, &s)|
                (d - 1).checked_mul(s).and_then(|v| n.checked_add(v))).expect("stride overflow")
        };
        let bytes = element_bytes(desc.dtype);
        assert_eq!(desc.offset_bytes % bytes, 0);
        assert!(span.checked_mul(bytes).and_then(|n| n.checked_add(desc.offset_bytes))
            .is_some_and(|n| n <= allocation.bytes), "tensor exceeds allocation");
        Self { handle: allocation.handle.clone().offset_start(desc.offset_bytes as u64), shape, strides, len, dtype: desc.dtype }
    }
    fn arg(&self) -> TensorArg<CudaRuntime> {
        unsafe { TensorArg::from_raw_parts(self.handle.clone(), self.strides.clone().into(), self.shape.clone().into()) }
    }
    fn byte_view(&self) -> Self {
        let bytes = element_bytes(self.dtype);
        let mut shape = self.shape.clone();
        shape.push(bytes);
        let mut strides: Vec<usize> = self.strides.iter()
            .map(|stride| stride.checked_mul(bytes).expect("byte stride overflow")).collect();
        strides.push(1);
        Self { handle: self.handle.clone(), shape, strides,
            len: self.len.checked_mul(bytes).expect("byte length overflow"), dtype: 8 }
    }
    fn packed(handle: Handle, shape: Vec<usize>, dtype: u32) -> Self {
        let mut strides = vec![1; shape.len()];
        let mut len = 1;
        for dim in (0..shape.len()).rev() { strides[dim] = len; len *= shape[dim]; }
        Self { handle, shape, strides, len, dtype }
    }
}

fn client() -> ComputeClient<CudaRuntime> { let mut c=CudaRuntime::client(&CudaDevice::default()); streams::bind(&mut c); c }
fn sync(client: &ComputeClient<CudaRuntime>) { block_on(client.sync()).expect("RUDA CUDA synchronization failed"); }
fn async_dispatch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("RUDA_TORCH_ASYNC").as_deref(), Ok("1") | Ok("true") | Ok("yes")))
}
fn finish_dispatch(client: &ComputeClient<CudaRuntime>) {
    if async_dispatch_enabled() { ASYNC_DISPATCHES.fetch_add(1, Ordering::Relaxed); } else { sync(client); }
}
fn checked(call: impl FnOnce()) -> i32 {
    match std::panic::catch_unwind(AssertUnwindSafe(call)) {
        Ok(()) => 0,
        Err(error) => {
            let message = error.downcast_ref::<String>().map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied()).unwrap_or("RUDA native error");
            ERROR.with(|slot| *slot.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap());
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_error() -> *const std::ffi::c_char { ERROR.with(|v| v.borrow().as_ptr()) }

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_abi_version() -> u32 { 9 }

// All pointer arguments below are valid, aligned, and held alive by the in-process C++ adapter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_alloc(bytes: usize, allocation: *mut *mut Allocation, ptr: *mut u64) -> i32 {
    checked(|| {
        let client = client();
        let handle = client.empty(bytes.max(4));
        let resource = client.get_resource(handle.clone()).expect("RUDA CUDA allocation failed");
        // get_resource runs on the owning device service and returns the
        // stream-ordered allocation pointer. In opt-in async mode subsequent
        // RUDA bindings/stream events protect its use; do not drain this queue
        // for every freshly allocated PyTorch output tensor.
        if !async_dispatch_enabled() { sync(&client); }
        unsafe { *ptr = resource.resource().ptr; *allocation = Box::into_raw(Box::new(Allocation { handle, bytes })); }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_free(allocation: *mut Allocation) -> i32 {
    checked(|| { if !allocation.is_null() { unsafe { drop(Box::from_raw(allocation)); } } })
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_sync() -> i32 { checked(streams::device_sync) }

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_fill(out: *const Descriptor, bits: u64) -> i32 {
    checked(|| {
        let out = unsafe { View::read(&*out) }.byte_view();
        if out.len == 0 { return; }
        let client = client();
        let count = u32::try_from(out.len.div_ceil(128)).expect("launch grid overflow");
        unsafe {
            kernels::fill_bytes::launch::<CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                out.arg(), bits as u32, (bits >> 32) as u32);
        }
        finish_dispatch(&client);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    })
}

fn launch(op: u32, a: &View, b: &View, out: &View, scalar: f32) {
    if out.len == 0 { return; }
    if op == 0 { convert(a, out); return; }
    if (89..=106).contains(&op) { primitives::launch(op, a, b, out, scalar); return; }
    if (78..=88).contains(&op) {
        let client = client();
        let work = if op == 84 { out.len.checked_mul(element_bytes(out.dtype)).expect("byte length overflow") } else { out.len };
        let count = u32::try_from(work.div_ceil(128)).expect("launch grid overflow");
        unsafe {
            if op == 84 {
                assert_eq!(a.dtype, 3);
                assert_eq!(b.dtype, out.dtype);
                kernels::select_bytes::launch::<CudaRuntime>(
                    &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                    a.arg(), b.byte_view().arg(), out.byte_view().arg());
            } else if op >= 85 {
                assert_eq!((a.dtype, b.dtype, out.dtype), (3, 3, 3));
                kernels::logical::launch::<CudaRuntime>(
                    &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                    a.arg(), b.arg(), out.arg(), op);
            } else {
                assert_eq!(a.dtype, b.dtype);
                assert_eq!(out.dtype, 3);
                macro_rules! run {
                    ($dtype:ty) => { kernels::compare_float::launch::<$dtype, CudaRuntime>(
                        &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                        a.arg(), b.arg(), out.arg(), op) };
                }
                match a.dtype {
                    0 => run!(f32), 1 => run!(f16), 2 => run!(bf16),
                    3..=8 => kernels::compare_integer::launch::<CudaRuntime>(
                        &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                        a.byte_view().arg(), b.byte_view().arg(), out.arg(),
                        (4..=7).contains(&a.dtype), a.dtype == 3, op),
                    _ => panic!("unsupported RUDA comparison dtype"),
                }
            }
        }
        finish_dispatch(&client);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        return;
    }
    assert!(a.dtype <= 2 && b.dtype <= 2 && out.dtype <= 2,
        "this RUDA arithmetic kernel requires floating tensors");
    // Dispatch before the legacy whole-tensor FP32 staging path.
    if (31..=34).contains(&op) && a.dtype == b.dtype {
        softmax::launch(op, a, b, out, scalar as usize);
        return;
    }
    if op == 7 || op == 30 { matmul::launch(op, a, b, out); return; }
    if op == 6 {
        assert_eq!(a.dtype, out.dtype);
        if out.len == 0 { return; }
        let client = client();
        let packed = |view: &View| {
            let mut expected = 1usize;
            for (&dim, &stride) in view.shape.iter().zip(&view.strides).rev() {
                if dim > 1 && stride != expected { return false; }
                expected = expected.checked_mul(dim).expect("reduction stride overflow");
            }
            true
        };
        let last_axis = !a.shape.is_empty() && a.shape.last().copied().unwrap_or(0) > 0
            && a.shape.len() == out.shape.len()
            && out.shape.last() == Some(&1)
            && a.shape[..a.shape.len()-1] == out.shape[..out.shape.len()-1]
            && packed(a) && packed(out);
        macro_rules! launch {
            ($kernel:ident, $dtype:ty, $count:expr) => { kernels::$kernel::launch::<$dtype, $dtype, CudaRuntime>(
                &client, RudaCount::Static($count, 1, 1), RudaDim::new_1d(128), a.arg(), out.arg()) };
        }
        if last_axis {
            let work = out.len.checked_mul(32).expect("reduction launch overflow");
            let count = u32::try_from(work.div_ceil(128)).expect("reduction launch grid overflow");
            unsafe { match a.dtype {
                0 => launch!(reduce_sum_last_warp, f32, count),
                1 => launch!(reduce_sum_last_warp, f16, count),
                2 => launch!(reduce_sum_last_warp, bf16, count), _ => unreachable!()
            } }
            WARP_REDUCTION_CALLS.fetch_add(1, Ordering::Relaxed);
        } else {
            let count = u32::try_from(out.len.div_ceil(128)).expect("reduction launch grid overflow");
            unsafe { match a.dtype {
                0 => launch!(reduce_sum_storage, f32, count),
                1 => launch!(reduce_sum_storage, f16, count),
                2 => launch!(reduce_sum_storage, bf16, count), _ => unreachable!()
            } }
            STORAGE_REDUCTION_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        finish_dispatch(&client);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if matches!(op, 1..=5 | 8..=29 | 35..=57 | 62..=64 | 69..=72) {
        pointwise::launch(op, a, b, out, scalar);
        return;
    }
    if (74..=77).contains(&op) {
        assert_eq!(a.dtype, out.dtype);
        let client = client();
        let count = u32::try_from(out.len.div_ceil(128)).expect("launch grid overflow");
        macro_rules! run {
            ($dtype:ty) => { kernels::adaptive_avg_pool::launch::<$dtype, CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                a.arg(), out.arg(), if op >= 76 { 3 } else { 2 }, op == 75 || op == 77) };
        }
        unsafe {
            match out.dtype {
                0 => run!(f32), 1 => run!(f16), 2 => run!(bf16),
                _ => panic!("unsupported RUDA pooling dtype"),
            }
        }
        finish_dispatch(&client);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if (58..=61).contains(&op) || (65..=68).contains(&op) || op == 73 {
        assert_eq!(a.dtype, out.dtype);
        assert_eq!(b.dtype, out.dtype);
        let client = client();
        let count = u32::try_from(out.len.div_ceil(128)).expect("launch grid overflow");
        macro_rules! run {
            ($dtype:ty) => { kernels::storage_pointwise::launch::<$dtype, CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                a.arg(), b.arg(), out.arg(), scalar, op) };
        }
        unsafe {
            match out.dtype {
                0 => run!(f32), 1 => run!(f16), 2 => run!(bf16),
                _ => panic!("unsupported RUDA storage dtype"),
            }
        }
        finish_dispatch(&client);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if a.dtype != 0 || b.dtype != 0 || out.dtype != 0 {
        let allocate = |v: &View| {
            let bytes = v.len.checked_mul(4).expect("size overflow").max(4);
            LEGACY_FP32_TEMP_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
            View::packed(client().empty(bytes), v.shape.clone(), 0)
        };
        let av = (a.dtype != 0).then(|| allocate(a));
        let bv = (b.dtype != 0).then(|| allocate(b));
        let ov = (out.dtype != 0).then(|| allocate(out));
        if op != 5 {
            if let Some(v) = &av { convert(a, v); }
            if let Some(v) = &bv { convert(b, v); }
        }
        if op == 28 || op == 29 || op == 49 || op == 54 || op == 56 || op == 63 || op == 72 {
            if let Some(v) = &ov { convert(out, v); }
        }
        launch(op, av.as_ref().unwrap_or(a), bv.as_ref().unwrap_or(b), ov.as_ref().unwrap_or(out), scalar);
        if let Some(v) = &ov { convert(v, out); }
        return;
    }
    let client = client();
    let work = if (31..=34).contains(&op) { out.len / out.shape[scalar as usize] } else { out.len };
    let count = u32::try_from(work.div_ceil(128)).expect("launch grid overflow");
    unsafe {
        match op {
            0..=5 | 8..=29 | 35..=57 | 62..=64 | 69..=72 => kernels::pointwise::launch::<f32, f32, f32, CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                a.arg(), b.arg(), out.arg(), scalar, op),
            6 => kernels::reduce::launch::<CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), out.arg()),
            7 => kernels::matmul::launch::<CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), b.arg(), out.arg()),
            30 => kernels::bmm::launch::<CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), b.arg(), out.arg()),
            31..=34 => kernels::softmax::launch::<f32, f32, CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), b.arg(), out.arg(),
                scalar as usize, op == 32 || op == 34, op >= 33),
            _ => panic!("unsupported RUDA operation {op}"),
        }
    }
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

fn convert(a: &View, out: &View) {
    assert_eq!(a.shape, out.shape);
    if out.len == 0 { return; }
    let client = client();
    let byte_work = a.dtype >= 3 && out.dtype >= 3;
    let work = if byte_work { out.len.checked_mul(element_bytes(out.dtype)).expect("byte length overflow") } else { out.len };
    let count = u32::try_from(work.div_ceil(128)).expect("launch grid overflow");
    macro_rules! run {
        ($i:ty, $o:ty) => { kernels::convert::launch::<$i, $o, CudaRuntime>(
            &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), out.arg()) };
    }
    if byte_work {
        unsafe {
            if a.dtype == out.dtype {
                kernels::copy_bytes::launch::<CudaRuntime>(
                    &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                    a.byte_view().arg(), out.byte_view().arg());
            } else {
                kernels::convert_integer::launch::<CudaRuntime>(
                    &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                    a.byte_view().arg(), out.byte_view().arg(),
                    (4..=7).contains(&a.dtype), a.dtype == 3, out.dtype == 3);
            }
        }
    } else if a.dtype == 3 || out.dtype == 3 {
        macro_rules! boolean {
            ($kernel:ident, $dtype:ty) => { kernels::$kernel::launch::<$dtype, CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), out.arg()) };
        }
        unsafe {
            match (a.dtype, out.dtype) {
                (0, 3) => boolean!(float_to_bool, f32),
                (1, 3) => boolean!(float_to_bool, f16),
                (2, 3) => boolean!(float_to_bool, bf16),
                (3, 0) => boolean!(bool_to_float, f32),
                (3, 1) => boolean!(bool_to_float, f16),
                (3, 2) => boolean!(bool_to_float, bf16),
                _ => panic!("unsupported RUDA bool conversion"),
            }
        }
    } else {
        macro_rules! convert_to {
            ($input:ty) => { match out.dtype {
                0 => run!($input, f32), 1 => run!($input, f16), 2 => run!($input, bf16),
                4 => run!($input, i64), 5 => run!($input, i32), 6 => run!($input, i16),
                7 => run!($input, i8), 8 => run!($input, u8),
                _ => panic!("unsupported RUDA conversion dtype"),
            } };
        }
        match a.dtype {
            0 => convert_to!(f32), 1 => convert_to!(f16), 2 => convert_to!(bf16),
            4 => convert_to!(i64), 5 => convert_to!(i32), 6 => convert_to!(i16),
            7 => convert_to!(i8), 8 => convert_to!(u8),
            _ => panic!("unsupported RUDA conversion dtype"),
        }
    }
    finish_dispatch(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_spatial(op: u32, a: *const Descriptor, b: *const Descriptor,
    out: *const Descriptor, params: *const i64, count: usize) -> i32 {
    checked(|| {
        assert!(!a.is_null() && !b.is_null() && !out.is_null() && !params.is_null());
        assert!(count <= 14);
        let (a, b, out, params) = unsafe {
            (View::read(&*a), View::read(&*b), View::read(&*out), std::slice::from_raw_parts(params, count))
        };
        spatial::launch(op, &a, &b, &out, params);
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_layer_norm(
    input: *const Descriptor, weight: *const Descriptor, bias: *const Descriptor,
    out: *const Descriptor, mean: *const Descriptor, rstd: *const Descriptor, epsilon: f32,
) -> i32 {
    checked(|| {
        assert!(!input.is_null() && !out.is_null() && !mean.is_null() && !rstd.is_null());
        let input = unsafe { View::read(&*input) };
        let weight = (!weight.is_null()).then(|| unsafe { View::read(&*weight) });
        let bias = (!bias.is_null()).then(|| unsafe { View::read(&*bias) });
        let out = unsafe { View::read(&*out) };
        let mean = unsafe { View::read(&*mean) };
        let rstd = unsafe { View::read(&*rstd) };
        normalization::layer_norm(&input, weight.as_ref(), bias.as_ref(), &out, &mean, &rstd, epsilon);
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_rms_norm(
    input: *const Descriptor, weight: *const Descriptor, out: *const Descriptor, epsilon: f32,
) -> i32 {
    checked(|| {
        assert!(!input.is_null() && !out.is_null());
        let input = unsafe { View::read(&*input) };
        let weight = (!weight.is_null()).then(|| unsafe { View::read(&*weight) });
        let out = unsafe { View::read(&*out) };
        normalization::rms_norm(&input, weight.as_ref(), &out, epsilon);
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_execute(op: u32, a: *const Descriptor, b: *const Descriptor, out: *const Descriptor, scalar: f32) -> i32 {
    checked(|| {
        let (a, b, out) = unsafe { (View::read(&*a), View::read(&*b), View::read(&*out)) };
        match op {
            0..=4 | 8..=29 | 35..=73 => { assert_eq!(a.shape, out.shape); assert_eq!(b.shape, out.shape); }
            78..=97 => { assert_eq!(a.shape, out.shape); assert_eq!(b.shape, out.shape); }
            98..=99 => {
                assert!(scalar >= 0.0 && scalar.fract() == 0.0 && (scalar as usize) < a.shape.len());
                let mut shape = a.shape.clone();
                shape[scalar as usize] = 1;
                assert_eq!(out.shape, shape);
            }
            100..=101 => {
                assert_eq!(a.shape, out.shape);
                assert!(scalar >= 0.0 && scalar.fract() == 0.0 && (scalar as usize) < a.shape.len());
            }
            102..=106 => {
                assert!(scalar >= 0.0 && scalar.fract() == 0.0 && (scalar as usize) < a.shape.len());
                let axis = scalar as usize;
                assert_eq!(a.shape.len(), out.shape.len());
                assert!(b.dtype == 4 || b.dtype == 5);
                if op >= 104 && a.len != 0 {
                    assert!(out.shape[axis] != 0, "RUDA index out of bounds");
                }
                if op == 102 || op == 104 || op == 106 {
                    assert_eq!(a.shape.len(), b.shape.len());
                    assert!(a.shape.iter().zip(&out.shape).enumerate().all(|(d, (x, y))| d == axis || x == y));
                    assert_eq!(b.shape, if op == 102 { &out.shape } else { &a.shape }.clone());
                } else {
                    assert_eq!(b.shape.len(), 1);
                    let mut expected = if op == 103 { a.shape.clone() } else { out.shape.clone() };
                    expected[axis] = b.len;
                    assert_eq!(expected, if op == 103 { &out.shape } else { &a.shape }.clone());
                }
            }
            5 => (),
            74..=77 => {
                let spatial = if op >= 76 { 3 } else { 2 };
                let rank = a.shape.len();
                assert!(rank == spatial + 1 || rank == spatial + 2);
                assert_eq!(rank, out.shape.len());
                assert_eq!(a.shape[..rank - spatial], out.shape[..rank - spatial]);
                assert!(a.shape[rank - spatial..].iter().all(|&size| size > 0));
                if op == 75 || op == 77 {
                    assert!(out.shape[rank - spatial..].iter().all(|&size| size > 0));
                }
            }
            6 => {
                assert_eq!(a.shape.len(), out.shape.len());
                assert!(a.shape.iter().zip(&out.shape).all(|(a, o)| *o == 1 || a == o));
            }
            7 => {
                assert_eq!(a.shape.len(), 2); assert_eq!(b.shape.len(), 2);
                assert_eq!(a.shape[1], b.shape[0]); assert_eq!(out.shape, [a.shape[0], b.shape[1]]);
            }
            30 => {
                assert_eq!(a.shape.len(), 3); assert_eq!(b.shape.len(), 3);
                assert_eq!(a.shape[0], b.shape[0]); assert_eq!(a.shape[2], b.shape[1]);
                assert_eq!(out.shape, [a.shape[0], a.shape[1], b.shape[2]]);
            }
            31..=34 => {
                assert_eq!(a.shape, out.shape); assert_eq!(b.shape, out.shape);
                assert!(scalar >= 0.0 && scalar.fract() == 0.0 && (scalar as usize) < a.shape.len());
            }
            _ => panic!("unsupported RUDA operation {op}"),
        }
        launch(op, &a, &b, &out, scalar);
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_addmm(
    bias: *const Descriptor, a: *const Descriptor, b: *const Descriptor,
    out: *const Descriptor, alpha: f32, beta: f32,
) -> i32 {
    checked(|| {
        let (bias, a, b, out) = unsafe {
            (View::read(&*bias), View::read(&*a), View::read(&*b), View::read(&*out))
        };
        matmul::addmm(&bias, &a, &b, &out, alpha, beta);
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_transfer(desc: *const Descriptor, host: *mut u8, upload: bool) -> i32 {
    checked(|| {
        let view = unsafe { View::read(&*desc) };
        if view.len == 0 { return; }
        assert!(!host.is_null());
        let client = client();
        let bytes = view.len.checked_mul(element_bytes(view.dtype)).expect("transfer size overflow");
        if upload {
            let input = client.create_from_slice(unsafe { std::slice::from_raw_parts(host, bytes) });
            let packed = View::packed(input, view.shape.clone(), view.dtype);
            launch(0, &packed, &packed, &view, 0.0);
            UPLOAD.fetch_add(bytes as u64, Ordering::Relaxed);
        } else {
            let packed = View::packed(client.empty(bytes), view.shape.clone(), view.dtype);
            launch(0, &view, &view, &packed, 0.0);
            let result = client.read_one(packed.handle).expect("RUDA GPU readback failed");
            unsafe { std::ptr::copy_nonoverlapping(result.as_ptr(), host, bytes); }
            DOWNLOAD.fetch_add(bytes as u64, Ordering::Relaxed);
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_counter(index: u32) -> u64 {
    match index {
        0 => LAUNCHES.load(Ordering::Relaxed),
        1 => UPLOAD.load(Ordering::Relaxed),
        2 => DOWNLOAD.load(Ordering::Relaxed),
        3 => RUBLAS_CALLS.load(Ordering::Relaxed),
        4 => SCALAR_MATMUL_CALLS.load(Ordering::Relaxed),
        5 => DIRECT_POINTWISE_CALLS.load(Ordering::Relaxed),
        6 => ADDMM_EPILOGUES.load(Ordering::Relaxed),
        7 => ADDMM_WORKSPACE_BYTES.load(Ordering::Relaxed),
        8 => LEGACY_FP32_TEMP_BYTES.load(Ordering::Relaxed),
        9 => WARP_SOFTMAX_CALLS.load(Ordering::Relaxed),
        10 => SCALAR_SOFTMAX_CALLS.load(Ordering::Relaxed),
        11 => FUSED_LAYER_NORM_CALLS.load(Ordering::Relaxed),
        12 => TRUSTED_INDEX_CALLS.load(Ordering::Relaxed),
        13 => STORAGE_REDUCTION_CALLS.load(Ordering::Relaxed),
        14 => WARP_REDUCTION_CALLS.load(Ordering::Relaxed),
        15 => FUSED_RMS_NORM_CALLS.load(Ordering::Relaxed),
        16 => ASYNC_DISPATCHES.load(Ordering::Relaxed),
        17 => paged::SPLIT_CALLS.load(Ordering::Relaxed),
        18 => paged::WORKSPACE_ALLOCS.load(Ordering::Relaxed),
        19 => paged::WORKSPACE_BYTES.load(Ordering::Relaxed),
        _ => 0,
    }
}

#[cfg(test)]
mod v14_gpu_tests;
#[cfg(test)]
mod v15_gpu_tests;
