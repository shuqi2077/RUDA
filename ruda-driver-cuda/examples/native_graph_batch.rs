//! cargo run -p ruda-driver-cuda --no-default-features --features std,direct-ptx --example native-graph-batch
use ruda_driver_cuda::{CudaDevice, CudaRuntime, graph::CudaGraph};
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
fn add(output: &mut Array<f32>, amount: f32) {
    if ABSOLUTE_POS < output.len() { output[ABSOLUTE_POS] += amount; }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(std::env::var("RUDA_CUDA_COMPILER").as_deref(), Ok("ptx"));
    let c = CudaRuntime::client(&CudaDevice::default()).fixed_execution_queue();
    let n = 257usize;
    let output = c.create_from_slice(f32::as_bytes(&vec![0.0; n]));
    let make = |amount: f32| unsafe { add::prepare(&c, RudaCount::Static(n.div_ceil(64) as u32, 1, 1),
        RudaDim::new_1d(64), ArrayArg::from_raw_parts(output.clone(), n), amount) };
    // Tracked build is optional. Ordinary build avoids the event-record overhead.
    let mut graph = unsafe { CudaGraph::build_tracked(&c, vec![make(1.0), make(2.0)]) }?;
    for position in 1..=32 {
        unsafe { graph.update_nodes_and_replay(vec![(0, make(position as f32)), (1, make(2.0))]) }?;
    }
    graph.wait_completion()?;
    assert!(graph.query_completion()?);
    let values = c.read_one(output)?;
    assert_eq!(f32::from_bytes(&values), vec![592.0; n]);
    graph.close()?;
    println!("32 batched updates verified on the real device; no speed claim.");
    Ok(())
}
