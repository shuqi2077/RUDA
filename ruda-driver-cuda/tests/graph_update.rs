//! Real NVIDIA/direct-PTX acceptance. Missing prerequisites are errors, not skips.
use ruda_driver_cuda::{CudaDevice, CudaRuntime, graph::CudaGraph};
use ruda_kernel::dsl::prelude::*;
use ruda::runtime::server::Handle;

#[ruda(launch)]
fn affine(input: &Array<f32>, output: &mut Array<f32>, scale: f32, bias: f32) {
    if ABSOLUTE_POS < output.len() { output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * scale + bias; }
}
#[ruda(launch)]
fn increment(output: &mut Array<f32>, amount: f32) {
    if ABSOLUTE_POS < output.len() { output[ABSOLUTE_POS] += amount; }
}
#[ruda(launch)]
fn mixed(output: &mut Array<f32>, a: f32, count: u32, b: f32) {
    if ABSOLUTE_POS < output.len() { output[ABSOLUTE_POS] = a * f32::cast_from(count) + b; }
}
fn client() -> ComputeClient<CudaRuntime> {
    assert_eq!(std::env::var("RUDA_CUDA_COMPILER").as_deref(), Ok("ptx"));
    let c = CudaRuntime::client(&CudaDevice::default()).fixed_execution_queue();
    println!("RUDA_GRAPH_UPDATE_RUNTIME,CudaRuntime,direct-ptx"); c
}
fn grid(n: usize) -> RudaCount { RudaCount::Static(n.div_ceil(64) as u32,1,1) }
fn node(c: &ComputeClient<CudaRuntime>, x: &Handle, y: &Handle, n: usize, a: f32, b: f32)
    -> PreparedKernel<CudaRuntime>
{
    unsafe { affine::prepare(c,grid(n),RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(x.clone(),n),ArrayArg::from_raw_parts(y.clone(),n),a,b) }
}
fn inc(c: &ComputeClient<CudaRuntime>, y: &Handle, n: usize, a: f32) -> PreparedKernel<CudaRuntime> {
    unsafe { increment::prepare(c,grid(n),RudaDim::new_1d(64),ArrayArg::from_raw_parts(y.clone(),n),a) }
}
fn read(c: &ComputeClient<CudaRuntime>, y: &Handle) -> Vec<f32> {
    f32::from_bytes(&c.read_one(y.clone()).expect("actual GPU read")).to_vec()
}

#[test]
fn graph_update_scalars_without_execution() {
    let c=client();
    for n in [1,31,32,33,63,64,65,257,1025] {
        let x=c.create_from_slice(f32::as_bytes(&vec![2.0;n]));
        let y=c.create_from_slice(f32::as_bytes(&vec![777.0;n+8]));
        let mut g=unsafe{CudaGraph::build(&c,vec![node(&c,&x,&y,n,1.0,1.0)])}.unwrap();
        unsafe{g.update_node(0,node(&c,&x,&y,n,3.0,4.0))}.unwrap();
        assert_eq!(read(&c,&y),vec![777.0;n+8]);
        g.replay().unwrap();
        let values=read(&c,&y);assert_eq!(&values[..n],vec![10.0;n]);
        assert_eq!(&values[n..],vec![777.0;8]);
    }
}
#[test]
fn graph_update_inflight_launches_keep_old_scalars() {
    let c=client();let n=4097;let y=c.create_from_slice(f32::as_bytes(&vec![0.0;n]));
    let mut g=unsafe{CudaGraph::build(&c,vec![inc(&c,&y,n,1.0)])}.unwrap();
    // No waits between launches: submitted instances must not all see the final value.
    for value in 1..=128 {unsafe{g.update_and_replay(0,inc(&c,&y,n,value as f32))}.unwrap();}
    g.synchronize().unwrap();assert_eq!(read(&c,&y),vec![8256.0;n]);
}
#[test]
fn graph_update_noop_still_replays_once() {
    let c=client();let y=c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g=unsafe{CudaGraph::build(&c,vec![inc(&c,&y,1,2.0)])}.unwrap();
    for _ in 0..11 {unsafe{g.update_and_replay(0,inc(&c,&y,1,2.0))}.unwrap();}
    assert_eq!(read(&c,&y),vec![22.0]);
}
#[test]
fn graph_update_mixed_scalar_types_preserve_packing() {
    let c=client();let y=c.empty(65*4);
    let make=|a,b,count| unsafe{mixed::prepare(&c,grid(65),RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(y.clone(),65),a,count,b)};
    let mut g=unsafe{CudaGraph::build(&c,vec![make(1.0,2.0,3)])}.unwrap();
    for (a,b,count) in [(2.0,5.0,7u32),(-3.0,1.0,11),(0.5,4.0,8)] {
        unsafe{g.update_and_replay(0,make(a,b,count))}.unwrap();
        assert_eq!(read(&c,&y),vec![a*count as f32+b;65]);
    }
}
#[test]
fn graph_update_rejects_pointer_rebinding_then_remains_usable() {
    let c=client();let x=c.create_from_slice(f32::as_bytes(&[2.0]));let other=c.create_from_slice(f32::as_bytes(&[8.0]));let y=c.empty(4);
    let mut g=unsafe{CudaGraph::build(&c,vec![node(&c,&x,&y,1,3.0,1.0)])}.unwrap();
    assert!(unsafe{g.update_and_replay(0,node(&c,&other,&y,1,9.0,0.0))}.is_err());
    g.replay().unwrap();assert_eq!(read(&c,&y),vec![7.0]);
}
#[test]
fn graph_update_rejects_shape_metadata_then_remains_usable() {
    let c=client();let x=c.create_from_slice(f32::as_bytes(&vec![2.0;64]));let y=c.empty(64*4);
    let mut g=unsafe{CudaGraph::build(&c,vec![node(&c,&x,&y,64,3.0,1.0)])}.unwrap();
    // Both grids are 1x1x1; logical length alone must still be rejected.
    assert!(unsafe{g.update_node(0,node(&c,&x,&y,63,9.0,0.0))}.is_err());
    g.replay().unwrap();assert_eq!(read(&c,&y),vec![7.0;64]);
}
#[test]
fn graph_update_rejects_grid_and_kernel_changes() {
    let c=client();let x=c.create_from_slice(f32::as_bytes(&[2.0]));let y=c.empty(4);
    let mut g=unsafe{CudaGraph::build(&c,vec![node(&c,&x,&y,1,3.0,1.0)])}.unwrap();
    let changed=unsafe{affine::prepare(&c,RudaCount::Static(2,1,1),RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(x.clone(),1),ArrayArg::from_raw_parts(y.clone(),1),3.0,1.0)};
    assert!(unsafe{g.update_node(0,changed)}.is_err());
    assert!(unsafe{g.update_node(0,inc(&c,&y,1,9.0))}.is_err());
    g.replay().unwrap();assert_eq!(read(&c,&y),vec![7.0]);
}
#[test]
fn graph_update_rejects_wrong_queue_and_node_index() {
    let c=client();let x=c.create_from_slice(f32::as_bytes(&[2.0]));let y=c.empty(4);
    let mut g=unsafe{CudaGraph::build(&c,vec![node(&c,&x,&y,1,3.0,1.0)])}.unwrap();
    let mut other=c.clone();unsafe{other.set_stream(ruda_core::stream_id::StreamId{value:c.execution_stream().value+10000});}
    assert!(unsafe{g.update_node(0,node(&other,&x,&y,1,2.0,0.0))}.is_err());
    assert!(unsafe{g.update_node(usize::MAX,node(&c,&x,&y,1,2.0,0.0))}.is_err());
    g.replay().unwrap();assert_eq!(read(&c,&y),vec![7.0]);
}
#[test]
fn graph_update_only_selected_node_changes() {
    let c=client();let x=c.create_from_slice(f32::as_bytes(&[2.0]));let tmp=c.empty(4);let y=c.empty(4);
    let mut g=unsafe{CudaGraph::build(&c,vec![node(&c,&x,&tmp,1,3.0,1.0),node(&c,&tmp,&y,1,2.0,1.0)])}.unwrap();
    unsafe{g.update_and_replay(1,node(&c,&tmp,&y,1,4.0,2.0))}.unwrap();
    assert_eq!(read(&c,&y),vec![30.0]);
}
#[test]
fn graph_update_query_try_close_and_closed_errors() {
    let c=client();let y=c.create_from_slice(f32::as_bytes(&vec![0.0;1025]));
    let mut g=unsafe{CudaGraph::build(&c,vec![inc(&c,&y,1025,1.0)])}.unwrap();
    unsafe{g.update_and_replay(0,inc(&c,&y,1025,2.0))}.unwrap();
    // GPU speed is not assumed; either nonblocking outcome is valid.
    let _ready=g.query().unwrap();
    if !g.try_close().unwrap() {g.synchronize().unwrap();assert!(g.query().unwrap());assert!(g.try_close().unwrap());}
    assert!(g.try_close().unwrap());
    assert!(g.query().is_err());assert!(g.replay().is_err());
    assert!(unsafe{g.update_node(0,inc(&c,&y,1025,8.0))}.is_err());
    assert_eq!(read(&c,&y),vec![2.0;1025]);
}
#[test]
fn graph_update_repeated_buffers_are_retained() {
    let c=client();let n=257;let y=c.create_from_slice(f32::as_bytes(&vec![0.0;n]));
    let mut g=unsafe{CudaGraph::build(&c,(0..32).map(|_|inc(&c,&y,n,1.0)).collect())}.unwrap();
    for i in 0..32 {unsafe{g.update_node(i,inc(&c,&y,n,2.0))}.unwrap();}
    // Every node shares a view; deduplication must not drop its allocation.
    g.replay().unwrap();assert_eq!(read(&c,&y),vec![64.0;n]);
}
#[test]
#[ignore="explicit same-device timing, not mandatory correctness acceptance"]
fn graph_update_benchmark() {
    use std::time::Instant;
    let c=client();
    for n in [1usize,256,4096] {
        let y=c.create_from_slice(f32::as_bytes(&vec![0.0;n]));let repeats=128;
        let mut g=unsafe{CudaGraph::build(&c,vec![inc(&c,&y,n,1.0)])}.unwrap();
        g.synchronize().unwrap();
        let start=Instant::now();
        for i in 1..=repeats {unsafe{g.update_node(0,inc(&c,&y,n,i as f32))}.unwrap();g.replay().unwrap();}
        g.synchronize().unwrap();let separate=start.elapsed().as_secs_f64();
        let start=Instant::now();
        for i in 1..=repeats {unsafe{g.update_and_replay(0,inc(&c,&y,n,i as f32))}.unwrap();}
        g.synchronize().unwrap();let combined=start.elapsed().as_secs_f64();
        assert_eq!(read(&c,&y),vec![(repeats*(repeats+1)) as f32;n]);
        println!("RUDA_GRAPH_UPDATE_TIMING,n={n},repeats={repeats},separate_s={separate},combined_s={combined}");
    }
}
