use ruda_core::future::block_on;
use half::{bf16, f16};
use ruda::runtime::server::Handle;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::dsl::prelude::*;
use std::{cell::RefCell, ffi::CString, panic::AssertUnwindSafe, sync::atomic::{AtomicU64, Ordering}};

mod kernels;
mod primitives;

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
pub extern "C" fn ruda_torch_abi_version() -> u32 { 3 }

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
        sync(&client);
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
        sync(&client);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        return;
    }
    assert!(a.dtype <= 2 && b.dtype <= 2 && out.dtype <= 2,
        "this RUDA arithmetic kernel requires floating tensors");
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
        sync(&client);
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
        sync(&client);
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if a.dtype != 0 || b.dtype != 0 || out.dtype != 0 {
        let allocate = |v: &View| View::packed(client().empty(v.len.checked_mul(4).expect("size overflow").max(4)), v.shape.clone(), 0);
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
            0..=5 | 8..=29 | 35..=57 | 62..=64 | 69..=72 => kernels::pointwise::launch::<CudaRuntime>(
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
    sync(&client);
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
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
