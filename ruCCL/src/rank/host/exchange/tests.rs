use super::*;
use crate::rank::host::reduction::{HostReductionElement, reduce_host_payload};
use crate::rank::{TcpRankSession, TcpRendezvousServer, UniqueId};
use std::thread;

mod pairwise;

#[test]
fn native_tcp_host_exchange_and_reduction_need_no_gx_context() {
    let unique_id = UniqueId::from_bytes([0x6d; 16]);
    let server = TcpRendezvousServer::bind("127.0.0.1:0", unique_id, 2).unwrap();
    let address = server.local_addr().unwrap();
    let coordinator = thread::spawn(move || server.run());
    let ranks = (0..2_u32)
        .map(|rank| {
            thread::spawn(move || {
                let session =
                    TcpRankSession::connect(address, unique_id, rank, 2, Duration::from_secs(5))
                        .unwrap();
                let exchange = HostStagedExchange::new(&session);
                assert_eq!(exchange.rank(), rank);
                assert_eq!(exchange.world_size(), 2);
                assert!(matches!(
                    exchange.validate_root(2),
                    Err(RankError::RankOutOfRange {
                        rank: 2,
                        world_size: 2
                    })
                ));
                assert!(
                    exchange
                        .all_reduce_host_staged(
                            ElementType::I64,
                            2,
                            Vec::new(),
                            ReductionOperation::Sum
                        )
                        .is_err()
                );
                assert!(matches!(
                    exchange.reduce_scatter_host_staged(
                        ElementType::U8,
                        3,
                        vec![0; 3],
                        ReductionOperation::Sum
                    ),
                    Err(RankError::InvalidLength(_)),
                ));
                exchange.set_timeout(Duration::from_secs(5)).unwrap();
                exchange.heartbeat(Duration::from_secs(5)).unwrap();

                let input = i64::encode(&[i64::from(rank) + 1, i64::from(rank) + 2]);
                let (payload, stats) = exchange
                    .all_reduce_host_staged(ElementType::I64, 2, input, ReductionOperation::Sum)
                    .unwrap();
                assert_eq!(payload, i64::encode(&[1, 2, 2, 3]));
                assert_eq!(
                    stats,
                    CollectiveStats {
                        algorithm: CollectiveAlgorithm::Direct,
                        transport: session.transport(),
                        steps: 1,
                        transferred_bytes: 48,
                        reduction_kernel_launches: 0,
                    }
                );
                let reduced =
                    reduce_host_payload(ElementType::I64, &payload, 2, 2, ReductionOperation::Sum)
                        .unwrap();
                assert_eq!(reduced, i64::encode(&[3, 5]));

                let input = f64::encode(&[f64::from(rank) + 0.5]);
                let (payload, stats) = exchange
                    .reduce_host_staged(ElementType::F64, 1, input, 0, ReductionOperation::Maximum)
                    .unwrap();
                assert_eq!(stats.transferred_bytes, if rank == 0 { 24 } else { 8 });
                if rank == 0 {
                    let payload = payload.unwrap();
                    assert_eq!(payload, f64::encode(&[0.5, 1.5]));
                    assert_eq!(
                        reduce_host_payload(
                            ElementType::F64,
                            &payload,
                            1,
                            2,
                            ReductionOperation::Maximum
                        )
                        .unwrap(),
                        f64::encode(&[1.5])
                    );
                } else {
                    assert!(payload.is_none());
                }

                let input = if rank == 0 {
                    vec![0, 1, 1, 0]
                } else {
                    vec![1, 1, 0, 1]
                };
                let (payload, stats) = exchange
                    .reduce_scatter_host_staged(
                        ElementType::Bool,
                        4,
                        input,
                        ReductionOperation::BitOr,
                    )
                    .unwrap();
                assert_eq!(
                    payload,
                    if rank == 0 {
                        vec![0, 1, 1, 1]
                    } else {
                        vec![1, 0, 0, 1]
                    }
                );
                assert_eq!(stats.transferred_bytes, 8);
                assert_eq!(
                    reduce_host_payload(
                        ElementType::Bool,
                        &payload,
                        2,
                        2,
                        ReductionOperation::BitOr
                    )
                    .unwrap(),
                    vec![1, 1]
                );
                let (payload, _) = exchange
                    .all_reduce_host_staged(
                        ElementType::I64,
                        0,
                        Vec::new(),
                        ReductionOperation::Sum,
                    )
                    .unwrap();
                assert!(payload.is_empty());
                assert!(
                    reduce_host_payload(ElementType::I64, &payload, 0, 2, ReductionOperation::Sum)
                        .unwrap()
                        .is_empty()
                );
                let stats = exchange.barrier().unwrap();
                assert_eq!(stats.steps, 1);
                assert_eq!(stats.transferred_bytes, 0);
            })
        })
        .collect::<Vec<_>>();
    for rank in ranks {
        rank.join().unwrap();
    }
    coordinator.join().unwrap().unwrap();
}
