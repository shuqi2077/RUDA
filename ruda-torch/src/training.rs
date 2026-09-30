//! Additive native training API 4; base tensor ABI 10.
//! This is an eager, first-order API, not training graph capture.
use super::{bf16, f16, checked, client, finish_dispatch, Descriptor, View,
    CudaRuntime, LAUNCHES, Ordering};
use super::training_kernels as k;
use ruda_optim::fused_adamw::storage as optim;
use ruda_kernel::dsl::prelude::*;

fn packed(v: &View) -> bool {
    let mut stride = 1usize;
    for (&dim, &actual) in v.shape.iter().zip(&v.strides).rev() {
        if dim > 1 && actual != stride { return false; }
        stride = stride.checked_mul(dim).expect("training shape overflow");
    }
    true
}
fn same(a: &View, b: &View) {
    assert_eq!(a.shape, b.shape, "training shape mismatch");
    assert_eq!(a.dtype, b.dtype, "training dtype mismatch");
}
fn count(work: usize) -> RudaCount {
    RudaCount::Static(u32::try_from(work.div_ceil(128)).expect("training grid overflow"), 1, 1)
}
fn flag(x: f32) -> bool { assert!(x == 0.0 || x == 1.0); x == 1.0 }
fn norm<F: Float + RudaElement, W: Float + RudaElement>(
    op: u32, v: &[View], s: &[f32],
) -> usize {
    let x = &v[0]; let w = &v[1];
    assert!(!x.shape.is_empty());
    let width = *x.shape.last().unwrap();
    assert!(width > 0, "normalization width must be positive");
    let rows = x.len / width;
    let weighted = flag(s[if op == 0 { 1 } else { 0 }]);
    if weighted {
        assert_eq!(w.shape, vec![width]);
        assert!(w.dtype == x.dtype || w.dtype == 0);
    }
    let c = client(); let mut launches = 0;
    if op == 0 {
        same(x, &v[2]); assert_eq!((v[3].dtype, v[3].len), (0, rows));
        assert!(s[0].is_finite() && s[0] > 0.0);
        if rows > 0 {
            unsafe { k::rms_forward::launch::<F, W, CudaRuntime>(
                &c, count(rows.checked_mul(32).expect("norm grid overflow")), RudaDim::new_1d(128),
                x.arg(), w.arg(), v[2].arg(), v[3].arg(), s[0], weighted); }
            launches += 1;
        }
    } else {
        same(x, &v[2]); assert_eq!((v[3].dtype, v[3].len), (0, rows));
        if flag(s[1]) {
            same(x, &v[4]);
            if rows > 0 {
                unsafe { k::rms_dx::launch::<F, W, CudaRuntime>(
                    &c, count(rows.checked_mul(32).expect("norm grid overflow")), RudaDim::new_1d(128),
                    x.arg(), w.arg(), v[2].arg(), v[3].arg(), v[4].arg(), weighted); }
                launches += 1;
            }
        }
        if flag(s[2]) {
            assert!(weighted); same(w, &v[5]);
            let parts = rows.div_ceil(32).clamp(1, 128);
            let len = parts.checked_mul(width).expect("norm partial overflow");
            let partial = View::packed(c.empty(len.checked_mul(4).expect("norm allocation overflow")), vec![parts, width], 0);
            unsafe {
                k::rms_dw_partial::launch::<F, CudaRuntime>(&c, count(len), RudaDim::new_1d(128),
                    x.arg(), v[2].arg(), v[3].arg(), partial.arg());
                k::rms_dw_merge::launch::<W, CudaRuntime>(&c, count(width), RudaDim::new_1d(128),
                    partial.arg(), v[5].arg());
            }
            launches += 2;
        }
    }
    if launches > 0 { finish_dispatch(&c); }
    launches
}


fn layer<F: Float + RudaElement, W: Float + RudaElement, B: Float + RudaElement>(
    op:u32, v:&[View], s:&[f32],
) -> usize {
    let x=&v[0]; let w=&v[1]; let b=&v[2];
    assert!(!x.shape.is_empty()); let width=*x.shape.last().unwrap(); assert!(width>0);
    let rows=x.len/width;
    let weighted=flag(s[if op==9 {1}else{0}]);
    let biased=flag(s[if op==9 {2}else{1}]);
    if weighted { assert_eq!(w.shape,vec![width]); assert!(w.dtype==x.dtype || w.dtype==0); }
    if biased { assert_eq!(b.shape,vec![width]); assert!(b.dtype==x.dtype || b.dtype==0); }
    let c=client(); let mut launches=0usize;
    if op==9 {
        same(x,&v[3]); assert_eq!((v[4].dtype,v[4].len),(0,rows)); assert_eq!((v[5].dtype,v[5].len),(0,rows));
        assert!(s[0].is_finite() && s[0]>0.0);
        if rows>0 {
            unsafe { k::layer_forward::launch::<F,W,B,CudaRuntime>(
                &c,count(rows.checked_mul(32).expect("LayerNorm grid overflow")),RudaDim::new_1d(128),
                x.arg(),w.arg(),b.arg(),v[3].arg(),v[4].arg(),v[5].arg(),s[0],weighted,biased); }
            launches=1;
        }
    } else {
        same(x,&v[3]); assert_eq!((v[4].dtype,v[4].len),(0,rows)); assert_eq!((v[5].dtype,v[5].len),(0,rows));
        let need_x=flag(s[2]); let need_w=flag(s[3]); let need_b=flag(s[4]);
        assert!(!need_w || weighted,"LayerNorm weight gradient requested without weight");
        assert!(!need_b || biased,"LayerNorm bias gradient requested without bias");
        if need_x && rows>0 {
            same(x,&v[6]);
            unsafe { k::layer_dx::launch::<F,W,CudaRuntime>(
                &c,count(rows.checked_mul(32).expect("LayerNorm dx grid overflow")),RudaDim::new_1d(128),
                x.arg(),w.arg(),v[3].arg(),v[4].arg(),v[5].arg(),v[6].arg(),weighted); }
            launches+=1;
        }
        if (need_w || need_b) && width>0 {
            if need_w { assert_eq!(v[7].shape,w.shape); assert_eq!(v[7].dtype,w.dtype); }
            if need_b { assert_eq!(v[8].shape,b.shape); assert_eq!(v[8].dtype,b.dtype); }
            let parts=rows.div_ceil(32).clamp(1,128);
            let plane=parts.checked_mul(width).expect("LayerNorm partial overflow");
            let partial=View::packed(c.empty(plane.checked_mul(8).expect("LayerNorm partial allocation overflow")),vec![2,parts,width],0);
            unsafe {
                k::layer_affine_partial::launch::<F,CudaRuntime>(
                    &c,count(plane),RudaDim::new_1d(128),x.arg(),v[3].arg(),v[4].arg(),v[5].arg(),partial.arg(),need_w,need_b);
                k::layer_affine_merge::launch::<W,B,CudaRuntime>(
                    &c,count(width),RudaDim::new_1d(128),partial.arg(),v[7].arg(),v[8].arg(),need_w,need_b);
            }
            launches+=2;
        }
    }
    if launches>0 { finish_dispatch(&c); }
    launches
}

fn dispatch_layer(op:u32,v:&[View],s:&[f32])->usize {
    macro_rules! wb { ($f:ty,$w:ty) => {{ match v[2].dtype {
        0=>layer::<$f,$w,f32>(op,v,s),1=>layer::<$f,$w,f16>(op,v,s),2=>layer::<$f,$w,bf16>(op,v,s),_=>unreachable!() } }}; }
    macro_rules! f { ($f:ty) => {{ match v[1].dtype {
        0=>wb!($f,f32),1=>wb!($f,f16),2=>wb!($f,bf16),_=>unreachable!() } }}; }
    match v[0].dtype {0=>f!(f32),1=>f!(f16),2=>f!(bf16),_=>unreachable!()}
}

fn gate<F: Float + RudaElement>(op: u32, v: &[View], s: &[f32]) -> usize {
    same(&v[0], &v[1]); same(&v[0], &v[2]);
    if v[0].len == 0 { return 0; }
    let c = client();
    if op == 2 {
        unsafe { k::silu_mul_forward::launch::<F, CudaRuntime>(&c, count(v[0].len), RudaDim::new_1d(128),
            v[0].arg(), v[1].arg(), v[2].arg()); }
    } else {
        let dx = flag(s[0]); let du = flag(s[1]);
        if !dx && !du { return 0; }
        if dx { same(&v[0], &v[3]); }
        if du { same(&v[0], &v[4]); }
        unsafe { k::silu_mul_backward::launch::<F, CudaRuntime>(&c, count(v[0].len), RudaDim::new_1d(128),
            v[0].arg(), v[1].arg(), v[2].arg(), v[3].arg(), v[4].arg(), dx, du); }
    }
    finish_dispatch(&c); 1
}

fn unscale(v: &[View], inv: f32) -> usize {
    assert!(inv.is_finite() && inv > 0.0);
    let n = v.len() - 2; let workspace = &v[n]; let found = &v[n+1];
    assert_eq!(workspace.dtype, 0); assert_eq!((found.dtype, found.len), (0, 1));
    let required: usize = v[..n].iter().map(|g| g.len.div_ceil(32).clamp(1, 1024)).sum();
    assert_eq!(workspace.len, required);
    let c = client(); let mut offset = 0usize;
    for grad in &v[..n] {
        assert!(grad.dtype <= 2);
        let warps = grad.len.div_ceil(32).clamp(1, 1024);
        let flags = View::packed(workspace.handle.clone().offset_start((offset*4) as u64), vec![warps], 0);
        macro_rules! run { ($f:ty) => { unsafe {
            optim::unscale_check::launch::<$f, CudaRuntime>(&c, count(warps*32), RudaDim::new_1d(128),
                grad.arg(), flags.arg(), inv);
        } }; }
        match grad.dtype { 0 => run!(f32), 1 => run!(f16), 2 => run!(bf16), _ => unreachable!() }
        offset += warps;
    }
    unsafe { optim::merge_flags::launch::<CudaRuntime>(&c, RudaCount::Static(1,1,1), RudaDim::new_1d(32),
        workspace.arg(), found.arg()); }
    finish_dispatch(&c); n+1
}

fn adamw<F: Float + RudaElement>(v: &[View], s: &[f32]) -> usize {
    same(&v[0], &v[1]);
    for a in &v[2..] { assert_eq!(a.dtype, 0); assert_eq!(a.shape, v[0].shape); }
    assert!(s.iter().all(|x| x.is_finite()));
    assert!(s[0] >= 0.0 && (0.0..1.0).contains(&s[1]) && (0.0..1.0).contains(&s[2])
        && s[3] > 0.0 && s[4] >= 0.0 && s[5] > 0.0 && s[5] <= 1.0 && s[6] > 0.0 && s[6] <= 1.0);
    if v[0].len == 0 { return 0; }
    let c = client();
    unsafe { optim::adamw::launch::<F, CudaRuntime>(&c, count(v[0].len), RudaDim::new_1d(128),
        v[0].arg(), v[1].arg(), v[2].arg(), v[3].arg(), v[4].arg(),
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], v[0].dtype != 0); }
    finish_dispatch(&c); 1
}


// New operations are additive. Opcodes 0..5 retain API-1 semantics.
fn analyze(v: &[View], inv: f32, with_norm: bool) -> usize {
    assert!(inv.is_finite() && inv > 0.0);
    let n = v.len() - 2;
    let workspace = &v[n]; let report = &v[n + 1];
    assert_eq!(workspace.dtype, 0);
    assert_eq!((report.dtype, report.len), (0, 3));
    let required: usize = v[..n].iter().map(|g| g.len.div_ceil(32).clamp(1, 1024) * 3).sum();
    assert_eq!(workspace.len, required);
    let c = client(); let mut offset = 0usize;
    for grad in &v[..n] {
        let warps = grad.len.div_ceil(32).clamp(1, 1024);
        let stats = View::packed(workspace.handle.clone().offset_start((offset * 4) as u64), vec![warps * 3], 0);
        macro_rules! run { ($f:ty) => { unsafe {
            optim::analyze_gradient::launch::<$f, CudaRuntime>(&c, count(warps * 32), RudaDim::new_1d(128),
                grad.arg(), stats.arg(), inv, with_norm);
        } }; }
        match grad.dtype { 0 => run!(f32), 1 => run!(f16), 2 => run!(bf16), _ => unreachable!() }
        offset += warps * 3;
    }
    unsafe { optim::merge_gradient_stats::launch::<CudaRuntime>(
        &c, RudaCount::Static(1, 1, 1), RudaDim::new_1d(32), workspace.arg(), report.arg()); }
    finish_dispatch(&c); n + 1
}

// API 4: reuse caller-owned scratch for bounded parallel statistics reduction.
fn analyze_hierarchical(v: &[View], inv: f32, with_norm: bool) -> usize {
    use ruda_optim::fused_adamw::stats_plan::{StatsPlan, STATS_FAN_IN};
    assert!(inv.is_finite() && inv > 0.0);
    let n = v.len() - 3;
    let workspace = &v[n]; let scratch = &v[n + 1]; let report = &v[n + 2];
    let rows: usize = v[..n].iter().map(|g| g.len.div_ceil(32).clamp(1, 1024)).sum();
    let plan = StatsPlan::new(rows).expect("invalid gradient statistics layout");
    assert_eq!((workspace.dtype, workspace.len), (0, rows * 3));
    assert_eq!((scratch.dtype, scratch.len), (0, plan.scratch_elements));
    assert_eq!((report.dtype, report.len), (0, 3));
    // All layout checks are above the first write, including scratch capacity.
    let c = client(); let mut offset = 0usize;
    for grad in &v[..n] {
        let warps = grad.len.div_ceil(32).clamp(1, 1024);
        let stats = View::packed(workspace.handle.clone().offset_start((offset * 4) as u64), vec![warps * 3], 0);
        macro_rules! run { ($f:ty) => { unsafe {
            optim::analyze_gradient::launch::<$f, CudaRuntime>(&c, count(warps * 32), RudaDim::new_1d(128),
                grad.arg(), stats.arg(), inv, with_norm);
        } }; }
        match grad.dtype { 0 => run!(f32), 1 => run!(f16), 2 => run!(bf16), _ => unreachable!() }
        offset += warps * 3;
    }
    let mut current = View::packed(workspace.handle.clone(), vec![rows * 3], 0);
    for stage in &plan.stages {
        let next = View::packed(scratch.handle.clone().offset_start((stage.output_offset * 4) as u64),
                                vec![stage.output_rows * 3], 0);
        unsafe { optim::merge_gradient_stats_chunks::launch::<CudaRuntime>(
            &c, count(stage.output_rows * 32), RudaDim::new_1d(128),
            current.arg(), next.arg(), STATS_FAN_IN); }
        // Ordered submissions retain their handles through the runtime. The
        // full scratch descriptor is owned by this call until finish_dispatch.
        current = next;
    }
    unsafe { optim::merge_gradient_stats::launch::<CudaRuntime>(
        &c, RudaCount::Static(1, 1, 1), RudaDim::new_1d(32), current.arg(), report.arg()); }
    finish_dispatch(&c); n + plan.stages.len() + 1
}

fn batch_adamw(v: &[View], s: &[f32]) -> usize {
    let n = v.len() / 5;
    assert!(s[0].is_finite() && s[0] > 0.0);
    assert!(s[1].is_finite() && (0.0..=1.0).contains(&s[1]));
    // Validate every descriptor and hyperparameter before the first write launch.
    for i in 0..n {
        let a = &v[i * 5..i * 5 + 5]; let h = &s[2 + i * 7..2 + (i + 1) * 7];
        same(&a[0], &a[1]);
        for state in &a[2..] { assert_eq!(state.dtype, 0); assert_eq!(state.shape, a[0].shape); }
        assert!(h.iter().all(|x| x.is_finite()));
        assert!(h[0] >= 0.0 && (0.0..1.0).contains(&h[1]) && (0.0..1.0).contains(&h[2])
            && h[3] > 0.0 && h[4] >= 0.0 && h[5] > 0.0 && h[5] <= 1.0 && h[6] > 0.0 && h[6] <= 1.0);
    }
    let c = client(); let mut launches = 0usize;
    for i in 0..n {
        let a = &v[i * 5..i * 5 + 5]; let h = &s[2 + i * 7..2 + (i + 1) * 7];
        if a[0].len == 0 { continue; }
        macro_rules! run { ($f:ty) => { unsafe {
            optim::adamw_scaled::launch::<$f, CudaRuntime>(&c, count(a[0].len), RudaDim::new_1d(128),
                a[0].arg(), a[1].arg(), a[2].arg(), a[3].arg(), a[4].arg(),
                h[0], h[1], h[2], h[3], h[4], h[5], h[6], s[0], s[1], a[0].dtype != 0);
        } }; }
        match a[0].dtype { 0 => run!(f32), 1 => run!(f16), 2 => run!(bf16), _ => unreachable!() }
        launches += 1;
    }
    // Same ordered runtime stream; references in v live through all submissions.
    // Unlike API 1 this does not synchronize after every parameter update.
    if launches > 0 { finish_dispatch(&c); }
    launches
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_training_api_version() -> u32 { 4 }

/// Descriptors and scalar arrays are owned/validated by the in-process C++ bridge.
/// API 1 operations: RMS fwd/bwd, SiLU-mul fwd/bwd, unscale/check, AdamW.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_training(
    op: u32, descriptors: *const Descriptor, n: usize, scalars: *const f32, ns: usize,
) -> i32 {
    checked(|| {
        let expected = match op { 0 => (4,2), 1 => (6,3), 2 => (3,0), 3 => (5,2),
            4 => { assert!((3..=4098).contains(&n)); (n,1) }, 5 => (5,7),
            6 => { assert!((3..=4098).contains(&n)); (n,2) },
            8 => { assert!((4..=4099).contains(&n)); (n,2) },
            7 => { assert!((5..=20480).contains(&n) && n % 5 == 0); (n,2 + (n / 5) * 7) },
            9 => (6,3), 10 => (9,5),
            _ => panic!("unsupported training opcode") };
        assert_eq!((n,ns), expected); assert!(!descriptors.is_null());
        let descriptors = unsafe { std::slice::from_raw_parts(descriptors,n) };
        let v: Vec<View> = descriptors.iter().map(|d| unsafe { View::read(d) }).collect();
        assert!(v.iter().all(|v| v.dtype <= 2 && packed(v)), "training expects contiguous floating storage");
        let s = if ns == 0 { &[] } else {
            assert!(!scalars.is_null()); unsafe { std::slice::from_raw_parts(scalars,ns) }
        };
        let launches = if op <= 1 {
            match (v[0].dtype, v[1].dtype) {
                (0,0) => norm::<f32,f32>(op,&v,s),
                (1,0) => norm::<f16,f32>(op,&v,s), (1,1) => norm::<f16,f16>(op,&v,s),
                (2,0) => norm::<bf16,f32>(op,&v,s), (2,2) => norm::<bf16,bf16>(op,&v,s),
                _ => panic!("unsupported training normalization storage"),
            }
        } else if op <= 3 {
            match v[0].dtype { 0 => gate::<f32>(op,&v,s), 1 => gate::<f16>(op,&v,s), 2 => gate::<bf16>(op,&v,s), _=>unreachable!() }
        } else if op == 9 || op == 10 { dispatch_layer(op,&v,s) }
        else if op == 4 { unscale(&v,s[0]) }
        else if op == 6 { analyze(&v, s[0], flag(s[1])) }
        else if op == 7 { batch_adamw(&v, s) }
        else if op == 8 { analyze_hierarchical(&v, s[0], flag(s[1])) }
        else { match v[0].dtype { 0=>adamw::<f32>(&v,s),1=>adamw::<f16>(&v,s),2=>adamw::<bf16>(&v,s),_=>unreachable!() } };
        LAUNCHES.fetch_add(launches as u64,Ordering::Relaxed);
    })
}
