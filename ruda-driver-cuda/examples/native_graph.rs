//! Run with the direct-ptx feature and RUDA_CUDA_COMPILER=ptx.
use ruda_driver_cuda::{CudaDevice, CudaRuntime, graph::CudaGraph};
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
fn affine(x: &Array<f32>, y: &mut Array<f32>, scale: f32, bias: f32) {
    if ABSOLUTE_POS < y.len() { y[ABSOLUTE_POS] = x[ABSOLUTE_POS] * scale + bias; }
}
fn main() {
    assert_eq!(std::env::var("RUDA_CUDA_COMPILER").as_deref(), Ok("ptx"),
        "set RUDA_CUDA_COMPILER=ptx and RUDA_PTX_VERSION before initializing RUDA");
    let client = CudaRuntime::client(&CudaDevice::default()).fixed_execution_queue();
    let n = 257usize;
    let x = client.create_from_slice(f32::as_bytes(&vec![2.0; n]));
    let scratch = client.empty(n * 4);
    let output = client.empty(n * 4);
    let grid = || RudaCount::Static(n.div_ceil(64) as u32, 1, 1);
    let first = unsafe { affine::prepare(&client, grid(), RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(x.clone(), n), ArrayArg::from_raw_parts(scratch.clone(), n), 3.0, 1.0) };
    let second = unsafe { affine::prepare(&client, grid(), RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(scratch.clone(), n), ArrayArg::from_raw_parts(output.clone(), n), 2.0, 0.0) };
    // SAFETY: all arguments are initialized/adequately sized native allocations
    // on this device/queue. First produces scratch before second consumes it.
    let mut graph = unsafe { CudaGraph::build(&client, vec![first, second]) }.expect("build graph");
    drop(x); drop(scratch); // the graph owns allocation/view references
    for _ in 0..100 { graph.replay().expect("native graph replay"); }
    graph.synchronize().expect("GPU completion");
    let bytes = client.read_one(output).expect("GPU output");
    assert!(f32::from_bytes(&bytes).iter().all(|&value| value == 14.0));
    println!("native graph: {} kernel nodes, 100 successful replays", graph.node_count());
    graph.close().expect("release graph after completion");
}
