//! Mandatory native-device tests. CPU/source substitutes are not accepted.
use ruda_driver_cuda::{CudaDevice, CudaRuntime, graph::CudaGraph};
use ruda_kernel::dsl::prelude::*;
use ruda::runtime::server::Handle;

#[ruda(launch)]
fn affine(input: &Array<f32>, output: &mut Array<f32>, scale: f32, bias: f32) {
    if ABSOLUTE_POS < output.len() { output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * scale + bias; }
}
#[ruda(launch)]
fn sum(a: &Array<f32>, b: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS < output.len() { output[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS]; }
}
#[ruda(launch)]
fn increment(output: &mut Array<f32>, amount: f32) {
    if ABSOLUTE_POS < output.len() { output[ABSOLUTE_POS] += amount; }
}
fn client() -> ComputeClient<CudaRuntime> {
    assert_eq!(std::env::var("RUDA_CUDA_COMPILER").as_deref(), Ok("ptx"));
    let c=CudaRuntime::client(&CudaDevice::default()).fixed_execution_queue();
    println!("RUDA_GRAPH_INFER_RUNTIME,CudaRuntime,direct-ptx"); c
}
fn grid(n:usize)->RudaCount { RudaCount::Static(n.div_ceil(64) as u32,1,1) }
fn node(c:&ComputeClient<CudaRuntime>,x:&Handle,y:&Handle,n:usize,a:f32,b:f32)->PreparedKernel<CudaRuntime> {
    unsafe { affine::prepare(c,grid(n),RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(x.clone(),n),ArrayArg::from_raw_parts(y.clone(),n),a,b) }
}
fn join(c:&ComputeClient<CudaRuntime>,a:&Handle,b:&Handle,y:&Handle,n:usize)->PreparedKernel<CudaRuntime> {
    unsafe { sum::prepare(c,grid(n),RudaDim::new_1d(64),ArrayArg::from_raw_parts(a.clone(),n),
        ArrayArg::from_raw_parts(b.clone(),n),ArrayArg::from_raw_parts(y.clone(),n)) }
}
fn inc(c:&ComputeClient<CudaRuntime>,y:&Handle,n:usize,a:f32)->PreparedKernel<CudaRuntime> {
    unsafe { increment::prepare(c,grid(n),RudaDim::new_1d(64),ArrayArg::from_raw_parts(y.clone(),n),a) }
}
fn read(c:&ComputeClient<CudaRuntime>,h:&Handle)->Vec<f32> {
    f32::from_bytes(&c.read_one(h.clone()).expect("actual device read")).to_vec()
}



#[test]
fn graph_infer_fork_join_correct() {
    let c=client();
    for n in [1,31,32,33,257,1025] {
        let x=c.create_from_slice(f32::as_bytes(&vec![2.0;n]));
        let a=c.empty(n*4);let b=c.empty(n*4);let y=c.create_from_slice(f32::as_bytes(&vec![777.0;n+8]));
        let mut g=unsafe{CudaGraph::build_inferred(&c,vec![node(&c,&x,&a,n,2.0,1.0),
            node(&c,&x,&b,n,3.0,2.0),join(&c,&a,&b,&y,n)])}.unwrap();
        assert_eq!(g.dependencies(),&[vec![],vec![],vec![0,1]]);
        g.replay().unwrap();let got=read(&c,&y);
        assert_eq!(&got[..n],vec![13.0;n]);assert_eq!(&got[n..],vec![777.0;8]);
    }
}
#[test]
fn graph_infer_read_before_overwrite() {
    let c=client();let x=c.create_from_slice(f32::as_bytes(&[2.0]));let y=c.empty(4);
    let mut g=unsafe{CudaGraph::build_inferred(&c,vec![node(&c,&x,&y,1,1.0,0.0),inc(&c,&x,1,5.0)])}.unwrap();
    assert_eq!(g.dependencies(),&[vec![],vec![0]]);g.replay().unwrap();
    assert_eq!(read(&c,&y),vec![2.0]);assert_eq!(read(&c,&x),vec![7.0]);
}
#[test]
fn graph_infer_writes_reduce_to_chain() {
    let c=client();let y=c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g=unsafe{CudaGraph::build_inferred(&c,(0..16).map(|_|inc(&c,&y,1,1.0)).collect())}.unwrap();
    assert_eq!(g.edge_count(),15);g.replay().unwrap();assert_eq!(read(&c,&y),vec![16.0]);
}
#[test]
fn graph_infer_partial_overlap_preserves_order() {
    let c=client();let full=c.create_from_slice(f32::as_bytes(&vec![0.0;128]));
    let a=full.clone().offset_end(32*4);let b=full.clone().offset_start(32*4);
    let mut g=unsafe{CudaGraph::build_inferred(&c,vec![inc(&c,&a,96,1.0),inc(&c,&b,96,2.0)])}.unwrap();
    assert_eq!(g.dependencies(),&[vec![],vec![0]]);g.replay().unwrap();let got=read(&c,&full);
    assert_eq!(&got[..32],vec![1.0;32]);assert_eq!(&got[32..96],vec![3.0;64]);assert_eq!(&got[96..],vec![2.0;32]);
}
#[test]
fn graph_infer_disjoint_views_stay_independent() {
    let c=client();let full=c.create_from_slice(f32::as_bytes(&vec![0.0;128]));
    let a=full.clone().offset_end(64*4);let b=full.clone().offset_start(64*4);
    let mut g=unsafe{CudaGraph::build_inferred(&c,vec![inc(&c,&a,64,1.0),inc(&c,&b,64,2.0)])}.unwrap();
    assert_eq!(g.edge_count(),0);g.replay().unwrap();let got=read(&c,&full);
    assert_eq!(&got[..64],vec![1.0;64]);assert_eq!(&got[64..],vec![2.0;64]);
}
#[test]
fn graph_infer_tracked_completion() {
    let c=client();let y=c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g=unsafe{CudaGraph::build_inferred_tracked(&c,vec![inc(&c,&y,1,3.0)])}.unwrap();
    g.replay().unwrap();g.wait_completion().unwrap();assert!(g.query_completion().unwrap());
    assert_eq!(read(&c,&y),vec![3.0]);g.close().unwrap();assert!(g.replay().is_err());
}
#[test]
fn graph_infer_scalar_update_preserves_dependencies() {
    let c=client();let n=257;let x=c.create_from_slice(f32::as_bytes(&vec![2.0;n]));
    let a=c.empty(n*4);let b=c.empty(n*4);let y=c.empty(n*4);
    let mut g=unsafe{CudaGraph::build_inferred(&c,vec![node(&c,&x,&a,n,1.0,0.0),
        node(&c,&x,&b,n,1.0,0.0),join(&c,&a,&b,&y,n)])}.unwrap();let before=g.to_dot();
    unsafe{g.update_nodes_and_replay(vec![(0,node(&c,&x,&a,n,2.0,1.0)),(1,node(&c,&x,&b,n,3.0,2.0))])}.unwrap();
    assert_eq!(g.to_dot(),before);assert_eq!(read(&c,&y),vec![13.0;n]);
}
#[test]
fn graph_infer_build_does_not_execute_kernels() {
    let c=client();let y=c.create_from_slice(f32::as_bytes(&[123.0]));
    let _g=unsafe{CudaGraph::build_inferred(&c,vec![inc(&c,&y,1,1.0)])}.unwrap();
    assert_eq!(read(&c,&y),vec![123.0]);
}
#[test]
fn graph_infer_wrong_queue_is_refused() {
    let c=client();let y=c.create_from_slice(f32::as_bytes(&[0.0]));let mut other=c.clone();
    unsafe{other.set_stream(ruda_core::stream_id::StreamId{value:c.execution_stream().value+10000});}
    assert!(unsafe{CudaGraph::build_inferred(&c,vec![inc(&other,&y,1,1.0)])}.is_err());
}
#[test]
fn graph_infer_empty_is_refused() {
    let c=client();assert!(unsafe{CudaGraph::build_inferred(&c,vec![])}.is_err());
}
#[test]
fn graph_infer_retains_inputs_after_caller_drop() {
    let c=client();let n=257;let x=c.create_from_slice(f32::as_bytes(&vec![2.0;n]));let y=c.empty(n*4);
    let mut g=unsafe{CudaGraph::build_inferred(&c,vec![node(&c,&x,&y,n,3.0,1.0)])}.unwrap();drop(x);
    for _ in 0..32{g.replay().unwrap();}assert_eq!(read(&c,&y),vec![7.0;n]);
}
#[test]
fn graph_infer_does_not_weaken_explicit_dag_rejection() {
    let c=client();let y=c.create_from_slice(f32::as_bytes(&[0.0]));
    assert!(unsafe{CudaGraph::build_dag(&c,vec![inc(&c,&y,1,1.0),inc(&c,&y,1,2.0)],vec![vec![],vec![]])}.is_err());
    let mut g=unsafe{CudaGraph::build_inferred(&c,vec![inc(&c,&y,1,1.0),inc(&c,&y,1,2.0)])}.unwrap();
    g.replay().unwrap();assert_eq!(read(&c,&y),vec![3.0]);
}

#[test]
#[ignore = "explicit same-device benchmark, never part of correctness acceptance"]
fn graph_infer_benchmark() {
    use std::time::Instant;
    let c=client();let repeats=128;
    for (n,branches) in [(257usize,2usize),(4096,8),(65536,16)] {
        let x=c.create_from_slice(f32::as_bytes(&vec![2.0;n]));
        let outputs:Vec<Vec<_>>=(0..3).map(|_|(0..branches).map(|_|c.empty(n*4)).collect()).collect();
        let prepare=|i:usize|outputs[i].iter().map(|y|node(&c,&x,y,n,3.0,1.0)).collect();
        let mut serial=unsafe{CudaGraph::build(&c,prepare(0))}.unwrap();
        let mut explicit=unsafe{CudaGraph::build_dag(&c,prepare(1),vec![vec![];branches])}.unwrap();
        let mut inferred=unsafe{CudaGraph::build_inferred(&c,prepare(2))}.unwrap();
        assert_eq!(explicit.dependencies(),inferred.dependencies());
        for _ in 0..8{serial.replay().unwrap();explicit.replay().unwrap();inferred.replay().unwrap();}
        inferred.synchronize().unwrap();let mut elapsed=[0.0f64;3];
        for round in 0..6 { for offset in 0..3 {
            let which=(round+offset)%3;let g=match which{0=>&mut serial,1=>&mut explicit,_=>&mut inferred};
            let start=Instant::now();for _ in 0..repeats{g.replay().unwrap();}g.synchronize().unwrap();
            elapsed[which]+=start.elapsed().as_secs_f64();
        }}
        for y in outputs.iter().flatten(){assert_eq!(read(&c,y),vec![7.0;n]);}
        println!("RUDA_GRAPH_INFER_TIMING,elements={n},branches={branches},rounds=6,repeats={repeats},serial_s={},explicit_s={},inferred_s={}",elapsed[0],elapsed[1],elapsed[2]);
    }
}
