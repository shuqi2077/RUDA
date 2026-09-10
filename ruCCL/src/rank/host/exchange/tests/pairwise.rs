use super::*;

#[test]
fn native_host_pairwise_exchange_preserves_alignment_and_rank_payload() {
    let unique_id = UniqueId::from_bytes([0x7c; 16]);
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
                for (element_type, payload, receive_bytes) in [
                    (ElementType::None, vec![], 0),
                    (ElementType::U32, vec![0; 3], 4),
                    (ElementType::U32, vec![0; 4], 3),
                ] {
                    assert!(matches!(
                        exchange.exchange_host_payload(
                            &[0],
                            element_type,
                            payload,
                            receive_bytes,
                            1 - rank,
                            1 - rank,
                            7,
                            0x7c
                        ),
                        Err(RankError::InvalidLength(
                            "pairwise host payload does not align to its element type"
                        ))
                    ));
                }
                let payload = vec![rank as u8, 0, 255];
                assert_eq!(
                    exchange
                        .exchange_host_payload(
                            &[0],
                            ElementType::U8,
                            payload,
                            3,
                            1 - rank,
                            1 - rank,
                            7,
                            0x7d
                        )
                        .unwrap(),
                    [1 - rank as u8, 0, 255]
                );
                assert!(
                    exchange
                        .exchange_host_payload(
                            &[0],
                            ElementType::U32,
                            vec![],
                            0,
                            1 - rank,
                            1 - rank,
                            3,
                            0x7e
                        )
                        .unwrap()
                        .is_empty()
                );
                exchange.barrier().unwrap();
            })
        })
        .collect::<Vec<_>>();
    for rank in ranks {
        rank.join().unwrap();
    }
    coordinator.join().unwrap().unwrap();
}
