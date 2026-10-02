use std::{collections::{HashMap, HashSet}, marker::PhantomData,
    sync::{Arc, Mutex, atomic::{AtomicU64, Ordering}}};
use ruda_communication::{CommunicationChannel, Message, Protocol, ProtocolClient};
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, RwLock};
use crate::{NodeId, node::base::NodeState};

/// Generation-tagged barrier: preserves early next-round messages, deduplicates
/// peers, removes completed state, and registers waiters before checking state.
pub(crate) struct SyncService<P: Protocol> {
    node_state: Arc<RwLock<Option<NodeState>>>,
    rounds: Mutex<HashMap<u64, HashSet<NodeId>>>,
    generation: AtomicU64,
    completed: AtomicU64,
    sync_notif: Notify,
    _p: PhantomData<P>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SyncRequest { node: NodeId, generation: u64 }

impl<P: Protocol> SyncService<P> {
    pub fn new(node_state: Arc<RwLock<Option<NodeState>>>) -> Self {
        Self { node_state, rounds: Mutex::new(HashMap::new()), generation: AtomicU64::new(0),
            completed: AtomicU64::new(0), sync_notif: Notify::new(), _p: PhantomData }
    }
    fn add_syncing_peer(&self, peer: NodeId, generation: u64) {
        let mut rounds = self.rounds.lock().unwrap();
        if generation < self.completed.load(Ordering::SeqCst) { return; }
        rounds.entry(generation).or_default().insert(peer);
        drop(rounds);
        self.sync_notif.notify_waiters();
    }
    /// Calls on a node must be serialized, as in the local collective server.
    pub async fn sync(&self) {
        let guard = self.node_state.read().await;
        let state = guard.as_ref().expect("sync before registration");
        let generation = self.generation.fetch_add(1, Ordering::SeqCst);
        self.add_syncing_peer(state.node_id, generation);
        for (id, address) in &state.nodes {
            if *id == state.node_id { continue; }
            let mut channel = P::Client::connect(address.clone(), "sync").await
                .expect("could not connect to peer for sync");
            let data = rmp_serde::to_vec(&SyncRequest { node: state.node_id, generation }).unwrap();
            channel.send(Message::new(data.into())).await.expect("peer closed sync connection");
        }
        loop {
            let notified = self.sync_notif.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let mut rounds = self.rounds.lock().unwrap();
                if rounds.get(&generation).is_some_and(|peers| {
                    peers.len() == state.nodes.len() && state.nodes.keys().all(|n| peers.contains(n))
                }) {
                    rounds.remove(&generation);
                    self.completed.store(generation + 1, Ordering::SeqCst);
                    return;
                }
            }
            notified.await;
        }
    }
    pub async fn handle_sync_connection<C: CommunicationChannel>(&self, mut channel: C) {
        let Ok(Some(message)) = channel.recv().await else { return; };
        let Ok(request) = rmp_serde::from_slice::<SyncRequest>(&message.data) else { return; };
        let guard = self.node_state.read().await;
        let Some(state) = guard.as_ref() else { return; };
        if !state.nodes.contains_key(&request.node) { return; }
        self.add_syncing_peer(request.node, request.generation);
    }
}
