//! Mandatory real-device cases. Missing NVIDIA/direct-PTX support is an error.
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
fn client() -> ComputeClient<CudaRuntime> {
    assert_eq!(std::env::var("RUDA_CUDA_COMPILER").as_deref(), Ok("ptx"));
    let c = CudaRuntime::client(&CudaDevice::default()).fixed_execution_queue();
    println!("RUDA_GRAPH_BATCH_RUNTIME,CudaRuntime,direct-ptx"); c
}
fn grid(n: usize) -> RudaCount { RudaCount::Static(n.div_ceil(64) as u32, 1, 1) }
fn affine_node(c: &ComputeClient<CudaRuntime>, x: &Handle, y: &Handle, n: usize, a: f32, b: f32)
    -> PreparedKernel<CudaRuntime>
{
    unsafe { affine::prepare(c, grid(n), RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(x.clone(), n), ArrayArg::from_raw_parts(y.clone(), n), a, b) }
}
fn inc(c: &ComputeClient<CudaRuntime>, y: &Handle, n: usize, a: f32) -> PreparedKernel<CudaRuntime> {
    unsafe { increment::prepare(c, grid(n), RudaDim::new_1d(64), ArrayArg::from_raw_parts(y.clone(), n), a) }
}
fn read(c: &ComputeClient<CudaRuntime>, y: &Handle) -> Vec<f32> {
    f32::from_bytes(&c.read_one(y.clone()).expect("actual device read")).to_vec()
}

#[test]
fn graph_batch_updates_all_nodes_before_one_replay() {
    let c = client();
    for n in [1, 31, 32, 33, 65, 257, 1025] {
        let x = c.create_from_slice(f32::as_bytes(&vec![2.0; n]));
        let tmp = c.empty(n * 4);
        let y = c.create_from_slice(f32::as_bytes(&vec![777.0; n + 8]));
        let mut g = unsafe { CudaGraph::build(&c, vec![
            affine_node(&c, &x, &tmp, n, 1.0, 0.0), affine_node(&c, &tmp, &y, n, 1.0, 0.0)]) }.unwrap();
        unsafe { g.update_nodes_and_replay(vec![
            (0, affine_node(&c, &x, &tmp, n, 3.0, 1.0)),
            (1, affine_node(&c, &tmp, &y, n, 4.0, 2.0))]) }.unwrap();
        let result = read(&c, &y);
        assert_eq!(&result[..n], vec![30.0; n]); assert_eq!(&result[n..], vec![777.0; 8]);
    }
}
#[test]
fn graph_batch_update_without_replay_does_not_execute() {
    let c = client(); let y = c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g = unsafe { CudaGraph::build(&c, vec![inc(&c, &y, 1, 1.0), inc(&c, &y, 1, 2.0)]) }.unwrap();
    unsafe { g.update_nodes(vec![(0, inc(&c, &y, 1, 3.0)), (1, inc(&c, &y, 1, 4.0))]) }.unwrap();
    assert_eq!(read(&c, &y), vec![0.0]); g.replay().unwrap(); assert_eq!(read(&c, &y), vec![7.0]);
}
#[test]
fn graph_batch_late_invalid_shape_changes_nothing() {
    let c = client(); let n = 64; let x = c.create_from_slice(f32::as_bytes(&vec![2.0; n]));
    let tmp = c.empty(n * 4); let y = c.empty(n * 4);
    let mut g = unsafe { CudaGraph::build(&c, vec![affine_node(&c, &x, &tmp, n, 3.0, 1.0),
        affine_node(&c, &tmp, &y, n, 2.0, 1.0)]) }.unwrap();
    // Node 0 is valid and different; node 1 has wrong shape with the SAME grid.
    assert!(unsafe { g.update_nodes_and_replay(vec![(0, affine_node(&c, &x, &tmp, n, 99.0, 1.0)),
        (1, affine_node(&c, &tmp, &y, n - 1, 9.0, 0.0))]) }.is_err());
    g.replay().unwrap(); assert_eq!(read(&c, &y), vec![15.0; n]);
}
#[test]
fn graph_batch_late_pointer_rebind_changes_nothing() {
    let c = client(); let y = c.create_from_slice(f32::as_bytes(&[0.0])); let other = c.empty(4);
    let mut g = unsafe { CudaGraph::build(&c, vec![inc(&c, &y, 1, 1.0), inc(&c, &y, 1, 2.0)]) }.unwrap();
    assert!(unsafe { g.update_nodes(vec![(0, inc(&c, &y, 1, 9.0)), (1, inc(&c, &other, 1, 9.0))]) }.is_err());
    g.replay().unwrap(); assert_eq!(read(&c, &y), vec![3.0]);
}
#[test]
fn graph_batch_duplicate_and_invalid_index_are_rejected() {
    let c = client(); let y = c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g = unsafe { CudaGraph::build(&c, vec![inc(&c, &y, 1, 1.0), inc(&c, &y, 1, 2.0)]) }.unwrap();
    for invalid in [0, 2, usize::MAX] {
        assert!(unsafe { g.update_nodes_and_replay(vec![(0, inc(&c, &y, 1, 10.0)), (invalid, inc(&c, &y, 1, 20.0))]) }.is_err());
    }
    assert_eq!(read(&c, &y), vec![0.0]); g.replay().unwrap(); assert_eq!(read(&c, &y), vec![3.0]);
}
#[test]
fn graph_batch_empty_is_not_implicit_replay() {
    let c = client(); let y = c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g = unsafe { CudaGraph::build(&c, vec![inc(&c, &y, 1, 1.0)]) }.unwrap();
    assert!(unsafe { g.update_nodes(vec![]) }.is_err());
    assert!(unsafe { g.update_nodes_and_replay(vec![]) }.is_err());
    assert_eq!(read(&c, &y), vec![0.0]);
}
#[test]
fn graph_batch_unsorted_partial_updates_preserve_node_order() {
    let c = client(); let x = c.create_from_slice(f32::as_bytes(&[2.0])); let a = c.empty(4); let b = c.empty(4); let y = c.empty(4);
    let mut g = unsafe { CudaGraph::build(&c, vec![affine_node(&c, &x, &a, 1, 1.0, 0.0),
        affine_node(&c, &a, &b, 1, 2.0, 1.0), affine_node(&c, &b, &y, 1, 1.0, 0.0)]) }.unwrap();
    unsafe { g.update_nodes_and_replay(vec![(2, affine_node(&c, &b, &y, 1, 4.0, 2.0)),
        (0, affine_node(&c, &x, &a, 1, 3.0, 1.0))]) }.unwrap();
    assert_eq!(read(&c, &y), vec![62.0]);
}
#[test]
fn graph_batch_unchanged_values_replay_exactly_once() {
    let c = client(); let y = c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g = unsafe { CudaGraph::build(&c, vec![inc(&c, &y, 1, 1.0), inc(&c, &y, 1, 2.0)]) }.unwrap();
    for _ in 0..19 { unsafe { g.update_nodes_and_replay(vec![(0, inc(&c, &y, 1, 1.0)), (1, inc(&c, &y, 1, 2.0))]) }.unwrap(); }
    assert_eq!(read(&c, &y), vec![57.0]);
}
#[test]
fn graph_batch_inflight_versions_are_ordered() {
    let c = client(); let n = 1025; let y = c.create_from_slice(f32::as_bytes(&vec![0.0; n]));
    let mut g = unsafe { CudaGraph::build(&c, vec![inc(&c, &y, n, 1.0), inc(&c, &y, n, 2.0)]) }.unwrap();
    for i in 1..=128 { unsafe { g.update_nodes_and_replay(vec![
        (0, inc(&c, &y, n, i as f32)), (1, inc(&c, &y, n, (2 * i) as f32))]) }.unwrap(); }
    assert_eq!(read(&c, &y), vec![24768.0; n]);
}
#[test]
fn graph_batch_tracking_is_opt_in_without_breaking_legacy_graph() {
    let c = client(); let y = c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g = unsafe { CudaGraph::build(&c, vec![inc(&c, &y, 1, 1.0)]) }.unwrap();
    assert!(g.query_completion().is_err()); assert!(g.wait_completion().is_err());
    g.replay().unwrap(); g.synchronize().unwrap(); assert!(g.query().unwrap()); assert_eq!(read(&c, &y), vec![1.0]);
}
#[test]
fn graph_batch_tracked_completion_covers_latest_launch() {
    let c = client(); let n = 257; let y = c.create_from_slice(f32::as_bytes(&vec![0.0; n]));
    let mut g = unsafe { CudaGraph::build_tracked(&c, vec![inc(&c, &y, n, 1.0), inc(&c, &y, n, 1.0)]) }.unwrap();
    g.wait_completion().unwrap(); assert!(g.query_completion().unwrap());
    for i in 1..=16 { unsafe { g.update_nodes_and_replay(vec![(0, inc(&c, &y, n, i as f32)), (1, inc(&c, &y, n, i as f32))]) }.unwrap(); }
    // No assumption that an immediate query is false on a fast GPU.
    let _ = g.query_completion().unwrap(); g.wait_completion().unwrap(); assert!(g.query_completion().unwrap());
    assert_eq!(read(&c, &y), vec![272.0; n]);
}
#[test]
fn graph_batch_tracked_update_only_does_not_execute_or_release() {
    let c = client(); let y = c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g = unsafe { CudaGraph::build_tracked(&c, vec![inc(&c, &y, 1, 1.0)]) }.unwrap();
    unsafe { g.update_nodes(vec![(0, inc(&c, &y, 1, 8.0))]) }.unwrap();
    g.wait_completion().unwrap(); assert!(g.query_completion().unwrap()); assert_eq!(read(&c, &y), vec![0.0]);
    g.replay().unwrap(); g.wait_completion().unwrap(); assert_eq!(read(&c, &y), vec![8.0]);
}
#[test]
fn graph_batch_tracked_close_is_idempotent_and_queries_fail_after_close() {
    let c = client(); let y = c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut g = unsafe { CudaGraph::build_tracked(&c, vec![inc(&c, &y, 1, 2.0)]) }.unwrap();
    g.replay().unwrap(); g.close().unwrap(); g.close().unwrap();
    assert!(g.query_completion().is_err()); assert!(g.wait_completion().is_err());
    assert!(unsafe { g.update_nodes(vec![(0, inc(&c, &y, 1, 8.0))]) }.is_err());
    assert_eq!(read(&c, &y), vec![2.0]);
}
#[test]
fn graph_batch_wrong_queue_in_late_node_leaves_graph_unchanged() {
    let c = client(); let y = c.create_from_slice(f32::as_bytes(&[0.0]));
    let mut other = c.clone(); unsafe { other.set_stream(ruda_core::stream_id::StreamId { value: c.execution_stream().value + 10000 }); }
    let mut g = unsafe { CudaGraph::build(&c, vec![inc(&c, &y, 1, 1.0), inc(&c, &y, 1, 2.0)]) }.unwrap();
    assert!(unsafe { g.update_nodes(vec![(0, inc(&c, &y, 1, 8.0)), (1, inc(&other, &y, 1, 9.0))]) }.is_err());
    g.replay().unwrap(); assert_eq!(read(&c, &y), vec![3.0]);
}

#[test]
#[ignore = "explicit same-device benchmark; not mandatory correctness acceptance"]
fn graph_batch_benchmark() {
    use std::time::Instant;
    let c = client(); let n = 257; let repeats = 64;
    for count in [2usize, 8, 32] {
        // Independent identical workloads, warmup outside both measurements.
        let a = c.create_from_slice(f32::as_bytes(&vec![0.0; n]));
        let b = c.create_from_slice(f32::as_bytes(&vec![0.0; n]));
        let mut separate = unsafe { CudaGraph::build(&c, (0..count).map(|_| inc(&c, &a, n, 1.0)).collect()) }.unwrap();
        let mut batched = unsafe { CudaGraph::build(&c, (0..count).map(|_| inc(&c, &b, n, 1.0)).collect()) }.unwrap();
        separate.replay().unwrap(); batched.replay().unwrap(); batched.synchronize().unwrap();
        let start = Instant::now();
        for value in 1..=repeats {
            for index in 0..count { unsafe { separate.update_node(index, inc(&c, &a, n, value as f32)) }.unwrap(); }
            separate.replay().unwrap();
        }
        separate.synchronize().unwrap(); let separate_s = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for value in 1..=repeats { unsafe { batched.update_nodes_and_replay(
            (0..count).map(|index| (index, inc(&c, &b, n, value as f32))).collect()) }.unwrap(); }
        batched.synchronize().unwrap(); let batched_s = start.elapsed().as_secs_f64();
        let expected = count as f32 * (1 + repeats * (repeats + 1) / 2) as f32;
        assert_eq!(read(&c, &a), vec![expected; n]); assert_eq!(read(&c, &b), vec![expected; n]);
        println!("RUDA_GRAPH_BATCH_TIMING,nodes={count},elements={n},repeats={repeats},separate_s={separate_s},batched_s={batched_s}");
    }
}
