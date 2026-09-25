//! Real NVIDIA-driver/PTX tests. No skip or CPU fallback when prerequisites are absent.
use ruda_driver_cuda::{CudaDevice, CudaRuntime, graph::CudaGraph};
use ruda_kernel::dsl::prelude::*;
use ruda::runtime::server::Handle;

#[ruda(launch)]
fn affine(input: &Array<f32>, output: &mut Array<f32>, scale: f32, bias: f32) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * scale + bias;
    }
}
#[ruda(launch)]
fn increment(output: &mut Array<f32>, amount: f32) {
    if ABSOLUTE_POS < output.len() { output[ABSOLUTE_POS] += amount; }
}
fn client() -> ComputeClient<CudaRuntime> {
    assert_eq!(std::env::var("RUDA_CUDA_COMPILER").as_deref(), Ok("ptx"), "graph acceptance requires direct PTX");
    let client = CudaRuntime::client(&CudaDevice::default()).fixed_execution_queue();
    println!("RUDA_GRAPH_RUNTIME,CudaRuntime,direct-ptx");
    client
}
fn count(n: usize) -> RudaCount { RudaCount::Static(n.div_ceil(64) as u32, 1, 1) }
fn node(client: &ComputeClient<CudaRuntime>, input: &Handle, output: &Handle,
        n: usize, scale: f32, bias: f32) -> PreparedKernel<CudaRuntime> {
    unsafe { affine::prepare(client, count(n), RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(input.clone(), n), ArrayArg::from_raw_parts(output.clone(), n), scale, bias) }
}
fn read(client: &ComputeClient<CudaRuntime>, handle: &Handle) -> Vec<f32> {
    let data = client.read_one(handle.clone()).expect("real GPU read");
    f32::from_bytes(&data).to_vec()
}

#[test]
fn graph_replay_two_nodes_and_tail_guards() {
    let client = client();
    for n in [1, 31, 32, 33, 63, 64, 65, 257, 1025] {
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.125).collect();
        let x = client.create_from_slice(f32::as_bytes(&input));
        let tmp = client.empty(n * 4);
        let y = client.create_from_slice(f32::as_bytes(&vec![777.0; n + 16]));
        let mut graph = unsafe { CudaGraph::build(&client, vec![
            node(&client, &x, &tmp, n, 2.0, 1.0), node(&client, &tmp, &y, n, 4.0, -2.0),
        ]) }.expect("graph build");
        assert_eq!(graph.node_count(), 2);
        assert!(read(&client, &y).iter().all(|&v| v == 777.0), "construction must not execute kernels");
        for _ in 0..3 {
            graph.replay().unwrap();
            let result = read(&client, &y);
            for i in 0..n { assert_eq!(result[i], (input[i] * 2.0 + 1.0) * 4.0 - 2.0); }
            assert!(result[n..].iter().all(|&v| v == 777.0));
        }
        graph.close().unwrap();
    }
}

#[test]
fn graph_replay_retains_input_and_intermediate_handles() {
    let client = client(); let n = 257;
    let x = client.create_from_slice(f32::as_bytes(&vec![2.0; n]));
    let tmp = client.empty(n * 4); let y = client.empty(n * 4);
    let mut graph = unsafe { CudaGraph::build(&client, vec![
        node(&client, &x, &tmp, n, 3.0, 1.0), node(&client, &tmp, &y, n, 2.0, 0.0)]) }.unwrap();
    drop(x); drop(tmp);
    // Pressure on allocator pools must not recycle graph-only buffers.
    for _ in 0..128 { let _ = client.create_from_slice(f32::as_bytes(&vec![-100.0; n])); }
    graph.replay().unwrap(); graph.synchronize().unwrap();
    assert_eq!(read(&client, &y), vec![14.0; n]);
    graph.close().unwrap();
}

#[test]
fn graph_replay_arguments_survive_later_host_scratch_reuse() {
    let client = client(); let n = 65;
    let x = client.create_from_slice(f32::as_bytes(&vec![2.0; n])); let y = client.empty(n * 4);
    let other = client.empty(n * 4);
    let mut graph = unsafe { CudaGraph::build(&client, vec![node(&client, &x, &y, n, 5.0, 7.0)]) }.unwrap();
    for i in 0..64 {
        unsafe { affine::launch(&client, count(n), RudaDim::new_1d(64),
            ArrayArg::from_raw_parts(x.clone(), n), ArrayArg::from_raw_parts(other.clone(), n), i as f32, -5.0); }
    }
    graph.replay().unwrap();
    assert_eq!(read(&client, &y), vec![17.0; n]);
}

#[test]
fn graph_replay_in_place_updates_and_queue_ordering() {
    let client = client(); let n = 257;
    let value = client.create_from_slice(f32::as_bytes(&vec![0.0; n]));
    let prepare = || unsafe { increment::prepare(&client, count(n), RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(value.clone(), n), 1.0) };
    let mut graph = unsafe { CudaGraph::build(&client, vec![prepare(), prepare()]) }.unwrap();
    for _ in 0..20 {
        graph.replay().unwrap();
        unsafe { increment::launch(&client, count(n), RudaDim::new_1d(64),
            ArrayArg::from_raw_parts(value.clone(), n), 3.0); }
    }
    graph.synchronize().unwrap();
    assert_eq!(read(&client, &value), vec![100.0; n]);
}

#[test]
fn graph_replay_close_is_idempotent_and_replay_after_close_fails() {
    let client = client(); let value = client.create_from_slice(f32::as_bytes(&[0.0]));
    let prepared = unsafe { increment::prepare(&client, count(1), RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(value.clone(), 1), 2.0) };
    let mut graph = unsafe { CudaGraph::build(&client, vec![prepared]) }.unwrap();
    graph.replay().unwrap(); graph.close().unwrap(); graph.close().unwrap();
    assert!(graph.replay().is_err());
    assert_eq!(read(&client, &value), vec![2.0]);
}

#[test]
fn graph_replay_empty_and_zero_grids_are_rejected() {
    let client = client();
    assert!(unsafe { CudaGraph::build(&client, vec![]) }.is_err());
    let x = client.create_from_slice(f32::as_bytes(&[1.0]));
    let y = client.create_from_slice(f32::as_bytes(&[777.0]));
    let prepared = unsafe { affine::prepare(&client, RudaCount::Static(0,1,1), RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(x, 1), ArrayArg::from_raw_parts(y.clone(), 1), 1.0, 0.0) };
    assert!(unsafe { CudaGraph::build(&client, vec![prepared]) }.is_err());
    assert_eq!(read(&client, &y), vec![777.0]);
}

#[test]
fn graph_replay_dynamic_grid_is_not_read_back() {
    let client = client(); let x = client.create_from_slice(f32::as_bytes(&[1.0]));
    let y = client.empty(4); let grid = client.create_from_slice(u32::as_bytes(&[1,1,1]));
    let prepared = unsafe { affine::prepare(&client, RudaCount::Dynamic(grid.binding()), RudaDim::new_1d(64),
        ArrayArg::from_raw_parts(x, 1), ArrayArg::from_raw_parts(y, 1), 1.0, 0.0) };
    let error = unsafe { CudaGraph::build(&client, vec![prepared]) }.err().expect("dynamic grid must be rejected");
    assert!(format!("{error:?}").contains("dynamic"));
}

#[test]
fn graph_replay_wrong_queue_rejected_before_graph_build() {
    let client = client(); let mut other = client.clone();
    unsafe { other.set_stream(ruda_core::stream_id::StreamId { value: client.execution_stream().value + 10000 }); }
    let x = client.create_from_slice(f32::as_bytes(&[1.0])); let y = client.empty(4);
    let prepared = node(&other, &x, &y, 1, 1.0, 0.0);
    assert!(unsafe { CudaGraph::build(&client, vec![prepared]) }.is_err());
}

#[test]
#[ignore = "explicit same-device benchmark; never counted as mandatory acceptance"]
fn graph_replay_benchmark() {
    use std::time::Instant;
    let client = client();
    for (n, nodes) in [(256usize, 2usize), (256, 16), (4096, 16)] {
        let value = client.create_from_slice(f32::as_bytes(&vec![0.0; n]));
        let prepare = || unsafe { increment::prepare(&client, count(n), RudaDim::new_1d(64),
            ArrayArg::from_raw_parts(value.clone(), n), 1.0) };
        let started = Instant::now();
        let mut graph = unsafe { CudaGraph::build(&client, (0..nodes).map(|_| prepare()).collect()) }.unwrap();
        graph.synchronize().unwrap();
        let build_seconds = started.elapsed().as_secs_f64();
        for _ in 0..10 { graph.replay().unwrap(); }
        graph.synchronize().unwrap();
        let repetitions = 1000;
        let start = Instant::now();
        for _ in 0..repetitions {
            for _ in 0..nodes {
                unsafe { increment::launch(&client, count(n), RudaDim::new_1d(64),
                    ArrayArg::from_raw_parts(value.clone(), n), 1.0); }
            }
        }
        graph.synchronize().unwrap(); let ordinary = start.elapsed().as_secs_f64();
        let start = Instant::now();
        for _ in 0..repetitions { graph.replay().unwrap(); }
        graph.synchronize().unwrap(); let replay = start.elapsed().as_secs_f64();
        let expected = ((10 + 2 * repetitions) * nodes) as f32;
        assert_eq!(read(&client, &value), vec![expected; n]);
        println!("RUDA_GRAPH_TIMING,n={n},nodes={nodes},repeats={repetitions},build_s={build_seconds},ordinary_s={ordinary},graph_s={replay}");
    }
}
