//! Real loopback WebSocket regression, not an in-memory communication mock.
//! Run with: cargo test -p ruCCL --features orchestrator rooted_network
use super::base::Node;
use crate::{PeerId, GlobalRegisterParams, ReduceOperation, ReduceStrategy, BroadcastStrategy};
use crate::global::orchestrator::base::GlobalOrchestrator;
use ruda_communication::{Address, websocket::{WebSocket,WsServer}};
use ruda_tensor::{TensorData, ops::FloatTensorOps};
use ruda_tensor_host::Host;
use std::{net::TcpListener, time::Duration};

type H = <Host as ruda_tensor::BackendTypes>::FloatTensorPrimitive;
fn tensor(value: f32) -> H { Host::float_from_data(TensorData::new(vec![value,value*2.0],[2]), &Default::default()) }
async fn values(tensor: H) -> Vec<f32> { Host::float_into_data(tensor).await.unwrap().to_vec::<f32>().unwrap() }

#[tokio::test(flavor="multi_thread",worker_threads=4)]
async fn rooted_network_repeated_reduce_broadcast_and_mismatched_round() {
    // Hold port reservations until all four unique ports have been selected.
    let listeners = (0..4).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect::<Vec<_>>();
    let ports = listeners.iter().map(|l|l.local_addr().unwrap().port()).collect::<Vec<_>>();
    drop(listeners);
    let address = |port| format!("ws://127.0.0.1:{port}").parse::<Address>().unwrap();
    let global = address(ports[0]);
    let server = tokio::spawn(GlobalOrchestrator::start(std::future::pending(),WsServer::new(ports[0])));
    let mut nodes: Vec<Node<Host,WebSocket>> = ports[1..].iter()
        .map(|&port| Node::new(&global,WsServer::new(port))).collect();
    let params = |index: usize| GlobalRegisterParams { global_address:global.clone(),
        node_address:address(ports[index+1]),data_service_port:ports[index+1],num_nodes:3 };
    let result = tokio::time::timeout(Duration::from_secs(30),async {
        let (first,tail) = nodes.split_at_mut(1);
        let (second,third) = tail.split_at_mut(1);
        let (a,b,c) = tokio::join!(
            first[0].register(vec![PeerId(10),PeerId(11)],params(0)),
            second[0].register(vec![PeerId(20)],params(1)),
            third[0].register(vec![PeerId(30)],params(2)));
        a.unwrap();b.unwrap();c.unwrap();
        // Node 0 already has local sum(1,2)=3. Four peers, NOT three nodes.
        for round in 0..12 {
            let strategy = if round%2 == 0 { ReduceStrategy::Centralized } else { ReduceStrategy::Tree(1) };
            let op = if round%3 == 0 { ReduceOperation::Mean } else { ReduceOperation::Sum };
            let (a,b,c) = tokio::join!(
                nodes[0].reduce(tensor(3.0),strategy,PeerId(20),op),
                nodes[1].reduce(tensor(3.0),strategy,PeerId(20),op),
                nodes[2].reduce(tensor(4.0),strategy,PeerId(20),op));
            assert!(a.unwrap().is_none());assert!(c.unwrap().is_none());
            let expected = if op == ReduceOperation::Mean {2.5} else {10.0};
            assert_eq!(values(b.unwrap().unwrap()).await,vec![expected,expected*2.0]);
            let strategy = if round%2 == 0 {BroadcastStrategy::Tree(2)} else {BroadcastStrategy::Centralized};
            let device = Default::default();
            let (a,b,c) = tokio::join!(
                nodes[0].broadcast(None,strategy,&device),
                nodes[1].broadcast(Some(tensor(round as f32)),strategy,&device),
                nodes[2].broadcast(None,strategy,&device));
            for value in [a,b,c] { assert_eq!(values(value.unwrap()).await,vec![round as f32,round as f32*2.0]); }
        }
        // All callers get a coordinated error rather than entering incompatible transfers.
        let device = Default::default();
        let (a,b,c) = tokio::join!(
            nodes[0].reduce(tensor(1.0),ReduceStrategy::Centralized,PeerId(20),ReduceOperation::Sum),
            nodes[1].broadcast(Some(tensor(2.0)),BroadcastStrategy::Centralized,&device),
            nodes[2].broadcast(None,BroadcastStrategy::Centralized,&device));
        assert!(a.is_err() && b.is_err() && c.is_err());
        let (first,tail) = nodes.split_at_mut(1);
        let (second,third) = tail.split_at_mut(1);
        tokio::join!(first[0].finish(),second[0].finish(),third[0].finish());
    }).await;
    drop(nodes);
    server.abort();
    result.expect("rooted network round timed out");
}
