//! Two independent branches and an inferred join from actual buffer visibility. Not a model benchmark.
use ruda_driver_cuda::{CudaDevice, CudaRuntime, graph::CudaGraph};
use ruda_kernel::dsl::prelude::*;
#[ruda(launch)]
fn scale(input:&Array<f32>,output:&mut Array<f32>,factor:f32) {
    if ABSOLUTE_POS<output.len() {output[ABSOLUTE_POS]=input[ABSOLUTE_POS]*factor;}
}
#[ruda(launch)]
fn add(a:&Array<f32>,b:&Array<f32>,output:&mut Array<f32>) {
    if ABSOLUTE_POS<output.len() {output[ABSOLUTE_POS]=a[ABSOLUTE_POS]+b[ABSOLUTE_POS];}
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    assert_eq!(std::env::var("RUDA_CUDA_COMPILER").as_deref(),Ok("ptx"));
    let c=CudaRuntime::client(&CudaDevice::default()).fixed_execution_queue();
    let n=257usize; let x=c.create_from_slice(f32::as_bytes(&vec![2.0;n]));
    let a=c.empty(n*4); let b=c.empty(n*4); let output=c.empty(n*4);
    let grid=||RudaCount::Static(n.div_ceil(64) as u32,1,1);
    let nodes=unsafe {vec![
        scale::prepare(&c,grid(),RudaDim::new_1d(64),ArrayArg::from_raw_parts(x.clone(),n),ArrayArg::from_raw_parts(a.clone(),n),3.0),
        scale::prepare(&c,grid(),RudaDim::new_1d(64),ArrayArg::from_raw_parts(x.clone(),n),ArrayArg::from_raw_parts(b.clone(),n),4.0),
        add::prepare(&c,grid(),RudaDim::new_1d(64),ArrayArg::from_raw_parts(a.clone(),n),ArrayArg::from_raw_parts(b.clone(),n),ArrayArg::from_raw_parts(output.clone(),n)),
    ]};
    let mut graph=unsafe {CudaGraph::build_inferred_tracked(&c,nodes)}?;
    std::fs::write("ruda-graph.dot",graph.to_dot())?;
    for _ in 0..32 {graph.replay()?;}
    graph.wait_completion()?;
    assert_eq!(f32::from_bytes(&c.read_one(output)?),vec![14.0;n]);
    graph.close()?; println!("Native dependency graph output verified; no concurrency/speed claim."); Ok(())
}
