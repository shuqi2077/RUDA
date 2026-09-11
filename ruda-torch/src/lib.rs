use ruda_core::future::block_on;
use half::{bf16, f16};
use ruda::runtime::server::Handle;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::dsl::prelude::*;
use std::{cell::RefCell, ffi::CString, panic::AssertUnwindSafe, sync::atomic::{AtomicU64, Ordering}};

mod kernels;

thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::default()); }
static LAUNCHES: AtomicU64 = AtomicU64::new(0);
static UPLOAD: AtomicU64 = AtomicU64::new(0);
static DOWNLOAD: AtomicU64 = AtomicU64::new(0);

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
    match dtype { 0 => 4, 1 | 2 => 2, _ => panic!("unsupported RUDA dtype") }
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
    fn packed(handle: Handle, shape: Vec<usize>, dtype: u32) -> Self {
        let mut strides = vec![1; shape.len()];
        let mut len = 1;
        for dim in (0..shape.len()).rev() { strides[dim] = len; len *= shape[dim]; }
        Self { handle, shape, strides, len, dtype }
    }
}

fn client() -> ComputeClient<CudaRuntime> { CudaRuntime::client(&CudaDevice::default()) }
fn sync(client: &ComputeClient<CudaRuntime>) { block_on(client.sync()).expect("RUDA CUDA synchronization failed"); }
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
pub extern "C" fn ruda_torch_abi_version() -> u32 { 2 }

// All pointer arguments below are valid, aligned, and held alive by the in-process C++ adapter.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_alloc(bytes: usize, allocation: *mut *mut Allocation, ptr: *mut u64) -> i32 {
    checked(|| {
        let client = client();
        let handle = client.empty(bytes.max(4));
        let resource = client.get_resource(handle.clone()).expect("RUDA CUDA allocation failed");
        sync(&client);
        unsafe { *ptr = resource.resource().ptr; *allocation = Box::into_raw(Box::new(Allocation { handle, bytes })); }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_free(allocation: *mut Allocation) -> i32 {
    checked(|| { if !allocation.is_null() { unsafe { drop(Box::from_raw(allocation)); } } })
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_sync() -> i32 { checked(|| sync(&client())) }

fn launch(op: u32, a: &View, b: &View, out: &View, scalar: f32) {
    if out.len == 0 { return; }
    if op == 0 { convert(a, out); return; }
    if a.dtype != 0 || b.dtype != 0 || out.dtype != 0 {
        let allocate = |v: &View| View::packed(client().empty(v.len.checked_mul(4).expect("size overflow").max(4)), v.shape.clone(), 0);
        let av = (a.dtype != 0).then(|| allocate(a));
        let bv = (b.dtype != 0).then(|| allocate(b));
        let ov = (out.dtype != 0).then(|| allocate(out));
        if op != 5 {
            if let Some(v) = &av { convert(a, v); }
            if let Some(v) = &bv { convert(b, v); }
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
            0..=5 | 8..=18 => kernels::pointwise::launch::<CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128),
                a.arg(), b.arg(), out.arg(), scalar, op),
            6 => kernels::reduce::launch::<CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), out.arg()),
            7 => kernels::matmul::launch::<CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), b.arg(), out.arg()),
            30 => kernels::bmm::launch::<CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), b.arg(), out.arg()),
            31..=34 => kernels::softmax::launch::<CudaRuntime>(
                &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), b.arg(), out.arg(),
                scalar as usize, op == 32 || op == 34, op >= 33),
            _ => panic!("unsupported RUDA operation {op}"),
        }
    }
    sync(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

fn convert(a: &View, out: &View) {
    assert_eq!(a.shape, out.shape);
    if out.len == 0 { return; }
    let client = client();
    let count = u32::try_from(out.len.div_ceil(128)).expect("launch grid overflow");
    macro_rules! run {
        ($i:ty, $o:ty) => { kernels::convert::launch::<$i, $o, CudaRuntime>(
            &client, RudaCount::Static(count, 1, 1), RudaDim::new_1d(128), a.arg(), out.arg()) };
    }
    match (a.dtype, out.dtype) {
        (0, 0) => run!(f32, f32), (0, 1) => run!(f32, f16), (0, 2) => run!(f32, bf16),
        (1, 0) => run!(f16, f32), (1, 1) => run!(f16, f16), (1, 2) => run!(f16, bf16),
        (2, 0) => run!(bf16, f32), (2, 1) => run!(bf16, f16), (2, 2) => run!(bf16, bf16),
        _ => panic!("unsupported RUDA conversion"),
    }
    sync(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_execute(op: u32, a: *const Descriptor, b: *const Descriptor, out: *const Descriptor, scalar: f32) -> i32 {
    checked(|| {
        let (a, b, out) = unsafe { (View::read(&*a), View::read(&*b), View::read(&*out)) };
        match op {
            0..=4 | 8..=18 => { assert_eq!(a.shape, out.shape); assert_eq!(b.shape, out.shape); }
            5 => (),
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
    match index { 0 => LAUNCHES.load(Ordering::Relaxed), 1 => UPLOAD.load(Ordering::Relaxed), 2 => DOWNLOAD.load(Ordering::Relaxed), _ => 0 }
}
