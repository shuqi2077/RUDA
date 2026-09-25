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
    println!("RUDA_GRAPH_DAG_RUNTIME,CudaRuntime,direct-ptx"); c
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
fn fork()->Vec<Vec<usize>> { vec![vec![],vec![],vec![0,1]] }

#[test]
fn graph_dag_shared_input_branches_join_correctly() {
    let c=client();
    for n in [1,31,32,33,65,257,1025] {
        let x=c.create_from_slice(f32::as_bytes(&vec![2.0;n])); let a=c.empty(n*4); let b=c.empty(n*4);
        let y=c.create_from_slice(f32::as_bytes(&vec![777.0;n+8]));
        let mut g=unsafe { CudaGraph::build_dag(&c,vec![node(&c,&x,&a,n,2.0,1.0),
            node(&c,&x,&b,n,3.0,2.0),join(&c,&a,&b,&y,n)],fork()) }.unwrap();
        assert_eq!(g.edge_count(),2); g.replay().unwrap();
        let got=read(&c,&y); assert_eq!(&got[..n],vec![13.0;n]); assert_eq!(&got[n..],vec![777.0;8]);
    }
}
#[test]
fn graph_dag_missing_join_edge_refused_before_execution() {
    let c=client(); let x=c.create_from_slice(f32::as_bytes(&[2.0])); let a=c.empty(4); let b=c.empty(4);
    let y=c.create_from_slice(f32::as_bytes(&[777.0]));
    let result=unsafe { CudaGraph::build_dag(&c,vec![node(&c,&x,&a,1,2.0,1.0),
        node(&c,&x,&b,1,3.0,2.0),join(&c,&a,&b,&y,1)],vec![vec![],vec![],vec![0]]) };
    assert!(result.is_err()); assert_eq!(read(&c,&y),vec![777.0]);
}
#[test]
fn graph_dag_unordered_writes_refused() {
    let c=client(); let y=c.create_from_slice(f32::as_bytes(&[0.0]));
    assert!(unsafe { CudaGraph::build_dag(&c,vec![inc(&c,&y,1,1.0),inc(&c,&y,1,2.0)],vec![vec![],vec![]]) }.is_err());
    assert_eq!(read(&c,&y),vec![0.0]);
}
#[test]
fn graph_dag_unordered_read_write_refused() {
    let c=client(); let x=c.create_from_slice(f32::as_bytes(&[2.0])); let y=c.empty(4);
    assert!(unsafe { CudaGraph::build_dag(&c,vec![node(&c,&x,&y,1,1.0,0.0),inc(&c,&x,1,2.0)],vec![vec![],vec![]]) }.is_err());
    assert_eq!(read(&c,&x),vec![2.0]);
}
#[test]
fn graph_dag_disjoint_views_can_be_independent() {
    let c=client(); let full=c.create_from_slice(f32::as_bytes(&vec![0.0;128]));
    let a=full.clone().offset_end(64*4); let b=full.clone().offset_start(64*4);
    let mut g=unsafe { CudaGraph::build_dag(&c,vec![inc(&c,&a,64,1.0),inc(&c,&b,64,2.0)],vec![vec![],vec![]]) }.unwrap();
    g.replay().unwrap(); let got=read(&c,&full);
    assert_eq!(&got[..64],vec![1.0;64]); assert_eq!(&got[64..],vec![2.0;64]);
}
#[test]
fn graph_dag_overlapping_views_refused() {
    let c=client(); let full=c.create_from_slice(f32::as_bytes(&vec![0.0;128]));
    let a=full.clone().offset_end(32*4); let b=full.clone().offset_start(32*4);
    assert!(unsafe { CudaGraph::build_dag(&c,vec![inc(&c,&a,96,1.0),inc(&c,&b,96,2.0)],vec![vec![],vec![]]) }.is_err());
}
#[test]
fn graph_dag_transitive_dependency_allows_shared_writes() {
    let c=client(); let y=c.create_from_slice(f32::as_bytes(&[0.0])); let other=c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g=unsafe { CudaGraph::build_dag(&c,vec![inc(&c,&y,1,1.0),inc(&c,&other,1,5.0),inc(&c,&y,1,2.0)],
        vec![vec![],vec![0],vec![1]]) }.unwrap();
    g.replay().unwrap(); assert_eq!(read(&c,&y),vec![3.0]);
}
#[test]
fn graph_dag_invalid_topologies_are_refused() {
    let c=client(); let y=c.create_from_slice(f32::as_bytes(&[0.0]));
    for deps in [vec![],vec![vec![0]],vec![vec![usize::MAX]],vec![vec![],vec![]]] {
        assert!(unsafe { CudaGraph::build_dag(&c,vec![inc(&c,&y,1,1.0)],deps) }.is_err());
    }
    assert_eq!(read(&c,&y),vec![0.0]);
}
#[test]
fn graph_dag_batch_update_keeps_topology() {
    let c=client(); let n=257; let x=c.create_from_slice(f32::as_bytes(&vec![2.0;n]));
    let a=c.empty(n*4); let b=c.empty(n*4); let y=c.empty(n*4);
    let mut g=unsafe { CudaGraph::build_dag(&c,vec![node(&c,&x,&a,n,1.0,0.0),
        node(&c,&x,&b,n,1.0,0.0),join(&c,&a,&b,&y,n)],fork()) }.unwrap();
    let dot=g.to_dot();
    for i in 1..=16 {
        unsafe { g.update_nodes_and_replay(vec![(1,node(&c,&x,&b,n,i as f32,1.0)),
            (0,node(&c,&x,&a,n,2.0,3.0))]) }.unwrap();
    }
    assert_eq!(read(&c,&y),vec![40.0;n]); assert_eq!(g.to_dot(),dot);
}
#[test]
fn graph_dag_legacy_build_still_orders_writes() {
    let c=client(); let y=c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g=unsafe { CudaGraph::build(&c,vec![inc(&c,&y,1,1.0),inc(&c,&y,1,2.0)]) }.unwrap();
    assert_eq!(g.dependencies(),&[vec![],vec![0]]); g.replay().unwrap(); assert_eq!(read(&c,&y),vec![3.0]);
}
#[test]
fn graph_dag_tracked_completion_and_close() {
    let c=client(); let a=c.create_from_slice(f32::as_bytes(&[0.0])); let b=c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g=unsafe { CudaGraph::build_dag_tracked(&c,vec![inc(&c,&a,1,1.0),inc(&c,&b,1,2.0)],vec![vec![],vec![]]) }.unwrap();
    g.replay().unwrap(); g.wait_completion().unwrap(); assert!(g.query_completion().unwrap());
    assert_eq!(read(&c,&a),vec![1.0]); assert_eq!(read(&c,&b),vec![2.0]);
    g.close().unwrap(); g.close().unwrap(); assert!(g.replay().is_err());
}
#[test]
fn graph_dag_dot_is_only_topology() {
    let c=client(); let a=c.create_from_slice(f32::as_bytes(&[0.0])); let b=c.create_from_slice(f32::as_bytes(&[0.0]));
    let g=unsafe { CudaGraph::build_dag(&c,vec![inc(&c,&a,1,1.0),inc(&c,&b,1,2.0)],vec![vec![],vec![]]) }.unwrap();
    let dot=g.to_dot(); assert_eq!(dot,"digraph ruda {\n  n0 [label=\"kernel 0\"];\n  n1 [label=\"kernel 1\"];\n}\n");
}
#[test]
fn graph_dag_wrong_queue_refused() {
    let c=client(); let y=c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut other=c.clone(); unsafe { other.set_stream(ruda_core::stream_id::StreamId{value:c.execution_stream().value+10000}); }
    assert!(unsafe { CudaGraph::build_dag(&c,vec![inc(&other,&y,1,1.0)],vec![vec![]]) }.is_err());
}
#[test]
fn graph_dag_bad_update_does_not_invalidate_existing_topology() {
    let c=client(); let y=c.create_from_slice(f32::as_bytes(&[0.0])); let other=c.empty(4);
    let mut g=unsafe { CudaGraph::build_dag(&c,vec![inc(&c,&y,1,1.0)],vec![vec![]]) }.unwrap();
    assert!(unsafe { g.update_node(0,inc(&c,&other,1,99.0)) }.is_err());
    g.replay().unwrap(); assert_eq!(read(&c,&y),vec![1.0]);
}

#[test]
#[ignore = "explicit device benchmark; not part of mandatory correctness acceptance"]
fn graph_dag_benchmark() {
    use std::time::Instant;
    let c=client(); let repeats=128;
    for (n,branches) in [(257usize,2usize),(4096,3),(65536,8)] {
        let x=c.create_from_slice(f32::as_bytes(&vec![2.0;n]));
        let a:Vec<_>=(0..branches).map(|_|c.empty(n*4)).collect();
        let b:Vec<_>=(0..branches).map(|_|c.empty(n*4)).collect();
        let mut serial=unsafe { CudaGraph::build(&c,a.iter().map(|y|node(&c,&x,y,n,3.0,1.0)).collect()) }.unwrap();
        let mut dag=unsafe { CudaGraph::build_dag(&c,b.iter().map(|y|node(&c,&x,y,n,3.0,1.0)).collect(),vec![vec![];branches]) }.unwrap();
        for _ in 0..8 { serial.replay().unwrap(); dag.replay().unwrap(); }
        dag.synchronize().unwrap(); let mut elapsed=[0.0f64;2];
        for round in 0..4 {
            for which in if round%2==0 {[0usize,1]} else {[1usize,0]} {
                let g=if which==0 {&mut serial} else {&mut dag};
                let start=Instant::now(); for _ in 0..repeats {g.replay().unwrap();} g.synchronize().unwrap();
                elapsed[which]+=start.elapsed().as_secs_f64();
            }
        }
        for y in a.iter().chain(&b) {assert_eq!(read(&c,y),vec![7.0;n]);}
        println!("RUDA_GRAPH_DAG_TIMING,elements={n},branches={branches},rounds=4,repeats={repeats},serial_s={},dag_s={}",elapsed[0],elapsed[1]);
    }
}
