//! Fixed-address runtime scalar updates; requires a real NVIDIA/direct-PTX runtime.
use ruda_driver_cuda::{CudaDevice, CudaRuntime, graph::CudaGraph};
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
fn add(output: &mut Array<f32>, amount: f32) {
    if ABSOLUTE_POS < output.len() { output[ABSOLUTE_POS] += amount; }
}
fn main() {
    assert_eq!(std::env::var("RUDA_CUDA_COMPILER").as_deref(),Ok("ptx"),"set RUDA_CUDA_COMPILER=ptx");
    let client=CudaRuntime::client(&CudaDevice::default()).fixed_execution_queue();
    let n=257;
    let output=client.create_from_slice(f32::as_bytes(&vec![0.0;n]));
    let prepare=|amount| unsafe {add::prepare(&client,
        RudaCount::Static(n.div_ceil(64) as u32,1,1),RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(output.clone(),n),amount)};
    let mut graph=unsafe{CudaGraph::build(&client,vec![prepare(1.0)])}.expect("build graph");
    for value in 1..=32 {
        // The scalar is valid for this kernel; buffer size/grid/metadata never change.
        unsafe{graph.update_and_replay(0,prepare(value as f32))}.expect("scalar update and replay");
    }
    println!("whole fixed queue ready: {}",graph.query().expect("query"));
    graph.synchronize().expect("synchronize");
    let result=client.read_one(output).expect("GPU read");
    assert_eq!(f32::from_bytes(&result),vec![528.0;n]);
    assert!(graph.try_close().expect("close completed graph"));
    println!("32 queued updates completed with checked GPU output");
}
