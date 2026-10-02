use crate::{
    PeerId,
    global::{
        NodeId,
        shared::{
            CollectiveMessageResponse, CollectiveSpec, GlobalCollectiveError, RemoteRequest, RemoteResponse,
            RequestId, SessionId,
        },
    },
};
use ruda_communication::Address;
use std::collections::HashMap;
use tokio::sync::mpsc::{Receiver, Sender};

pub(crate) struct Session {
    response_sender: Sender<CollectiveMessageResponse>,
    response_receiver: Option<Receiver<CollectiveMessageResponse>>,
}

impl Session {
    fn new() -> Self {
        let (response_sender, recv) = tokio::sync::mpsc::channel::<CollectiveMessageResponse>(64);
        Self {
            response_sender,
            response_receiver: Some(recv),
        }
    }

    async fn respond(&mut self, response: CollectiveMessageResponse) {
        if let Err(error) = self.response_sender.try_send(response) {
            log::warn!("collective response queue closed/full: {error}");
        }
    }
}

pub(crate) struct GlobalCollectiveState {
    /// The ids passed to each register so far, and their addresses
    registered_nodes: HashMap<SessionId, NodeId>,
    /// Address for each node
    node_addresses: HashMap<NodeId, Address>,
    /// Peer on each node
    node_peers: HashMap<NodeId, Vec<PeerId>>,

    /// How many total nodes for the current register operation, as defined by the first caller
    cur_num_nodes: Option<u32>,
    /// How many peers have registered total
    num_global_peers: u32,

    register_requests: Vec<(SessionId, RequestId, NodeId)>,

    sessions: HashMap<SessionId, Session>,
    pending_collectives: Vec<(SessionId, RequestId, CollectiveSpec)>,
    next_transfer_id: u64,
    next_node_id: u32,
    failed: Option<GlobalCollectiveError>,
}

impl GlobalCollectiveState {
    pub fn new() -> Self {
        Self {
            registered_nodes: HashMap::new(),
            node_addresses: HashMap::new(),
            node_peers: HashMap::new(),
            cur_num_nodes: None,
            num_global_peers: 0,
            register_requests: Vec::new(),
            sessions: HashMap::new(),
            pending_collectives: Vec::new(),
            next_transfer_id: 2, // All-reduce algorithms reserve 0 and 1.
            next_node_id: 0,
            failed: None,
        }
    }

    pub(crate) fn init_session(&mut self, id: SessionId) {
        if self.sessions.contains_key(&id) {
            return;
        }
        self.sessions.insert(id, Session::new());
    }

    /// Create the session with given id if necessary, and get the response receiver
    pub(crate) fn get_session_responder(
        &mut self,
        id: SessionId,
    ) -> Result<Receiver<CollectiveMessageResponse>, GlobalCollectiveError> {
        self.init_session(id);
        let session = self.sessions.get_mut(&id).unwrap();
        let response_recv = session.response_receiver.take();

        response_recv.ok_or(GlobalCollectiveError::InvalidMessage)
    }

    pub(crate) async fn respond(
        &mut self,
        session_id: SessionId,
        response: CollectiveMessageResponse,
    ) {
        if let Some(session) = self.sessions.get_mut(&session_id) { session.respond(response).await; }
    }

    /// A lost socket poisons the group; late Begin messages cannot form a new
    /// operation using stale peers. Fresh registration is allowed only after
    /// every old participant has disconnected/unregistered.
    pub(crate) async fn disconnect(&mut self, session_id: SessionId) {
        if let Some(node) = self.registered_nodes.remove(&session_id) {
            self.node_addresses.remove(&node);
            self.node_peers.remove(&node);
            self.num_global_peers = self.node_peers.values().map(|p| p.len() as u32).sum();
            let error = GlobalCollectiveError::PeerLost(node);
            self.failed = Some(error.clone());
            for (session, request_id, _) in core::mem::take(&mut self.pending_collectives) {
                self.respond(session, CollectiveMessageResponse {request_id, content: RemoteResponse::Error(error.clone())}).await;
            }
            for (session, request_id, _) in core::mem::take(&mut self.register_requests) {
                self.respond(session, CollectiveMessageResponse {request_id, content: RemoteResponse::Error(error.clone())}).await;
            }
        }
        self.sessions.remove(&session_id);
        if self.registered_nodes.is_empty() {
            self.cur_num_nodes = None; self.next_node_id = 0; self.failed = None;
        }
    }

    /// Process an incoming node's request
    pub(crate) async fn process_request(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        request: RemoteRequest,
    ) {
        if let Some(error) = self.failed.clone() {
            if !matches!(request, RemoteRequest::Finish) {
                self.respond(session_id, CollectiveMessageResponse { request_id,
                    content: RemoteResponse::Error(error) }).await;
                return;
            }
        }
        if let Err(err) = match request {
            RemoteRequest::Register {
                node_addr,
                num_nodes,
                peers,
            } => {
                self.register(session_id, request_id, node_addr, num_nodes, peers)
                    .await
            }
            RemoteRequest::Begin(spec) => self.begin(session_id, request_id, spec).await,
            RemoteRequest::Finish => self.finish(session_id, request_id).await,
        } {
            // Error occurred, send it as response
            let content = RemoteResponse::Error(err);
            self.respond(
                session_id,
                CollectiveMessageResponse {
                    request_id,
                    content,
                },
            )
            .await;
        }
    }

    /// Un-register a node. Any pending requests will be cancelled, returning error responses.
    async fn finish(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
    ) -> Result<(), GlobalCollectiveError> {
        let node_id = self
            .registered_nodes
            .remove(&session_id)
            .ok_or(GlobalCollectiveError::NotRegisteredOnFinish)?;
        self.failed = Some(GlobalCollectiveError::PeerLost(node_id));
        self.node_addresses.remove(&node_id);
        self.node_peers.remove(&node_id);
        self.num_global_peers = self.node_peers.values().map(|peers| peers.len() as u32).sum();
        for (session, request_id, _) in core::mem::take(&mut self.pending_collectives) {
            self.respond(session, CollectiveMessageResponse {
                request_id,
                content: RemoteResponse::Error(GlobalCollectiveError::PeerLost(node_id)),
            }).await;
        }
        if self.registered_nodes.is_empty() { self.cur_num_nodes = None; self.next_node_id = 0; self.failed = None; }

        for (session, req, _) in core::mem::take(&mut self.register_requests) {
            self.respond(session, CollectiveMessageResponse { request_id: req,
                content: RemoteResponse::Error(GlobalCollectiveError::PeerLost(node_id)) }).await;
        }

        self.respond(
            session_id,
            CollectiveMessageResponse {
                request_id,
                content: RemoteResponse::FinishAck,
            },
        )
        .await;

        Ok(())
    }

    async fn register(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        node_addr: Address,
        num_nodes: u32,
        peers: Vec<PeerId>,
    ) -> Result<(), GlobalCollectiveError> {
        // Validate before mutating the membership or global mean denominator.
        if num_nodes == 0 || peers.is_empty() {
            return Err(GlobalCollectiveError::RegisterParamsMismatch);
        }
        if self.registered_nodes.contains_key(&session_id)
            || self.node_addresses.values().any(|address| *address == node_addr)
        {
            return Err(GlobalCollectiveError::DoubleRegister);
        }
        // Replacing a node in a live membership requires a fresh group: the
        // existing nodes otherwise retain stale peer addresses in NodeState.
        if !self.registered_nodes.is_empty() && self.register_requests.is_empty() {
            return Err(GlobalCollectiveError::RegisterParamsMismatch);
        }
        let mut seen = std::collections::HashSet::new();
        for peer in &peers {
            if !seen.insert(*peer) || self.node_peers.values().any(|p| p.contains(peer)) {
                return Err(GlobalCollectiveError::DuplicatePeer(*peer));
            }
        }
        match &self.cur_num_nodes {
            Some(cur_num_nodes) => {
                if *cur_num_nodes != num_nodes {
                    return Err(GlobalCollectiveError::RegisterParamsMismatch);
                }
            }
            None => {
                self.cur_num_nodes = Some(num_nodes);
            }
        }

        self.num_global_peers += peers.len() as u32;

        let node_id: NodeId = self.next_node_id.into();
        self.next_node_id = self.next_node_id.checked_add(1).expect("node identifier exhausted");
        self.registered_nodes.insert(session_id, node_id);
        if self.node_addresses.values().any(|addr| node_addr == *addr) {
            return Err(GlobalCollectiveError::DoubleRegister);
        }
        self.node_addresses.insert(node_id, node_addr);
        self.node_peers.insert(node_id, peers);

        self.register_requests
            .push((session_id, request_id, node_id));

        if self.registered_nodes.len() == num_nodes as usize {
            let mut callbacks = vec![];
            core::mem::swap(&mut callbacks, &mut self.register_requests);

            for (session, request, node_id) in callbacks {
                let content = RemoteResponse::Register {
                    node_id,
                    nodes: self.node_addresses.clone(),
                    num_global_devices: self.num_global_peers,
                };
                let resp = CollectiveMessageResponse {
                    request_id: request,
                    content,
                };
                self.respond(session, resp).await;
            }
        }

        Ok(())
    }

    async fn begin(&mut self, session: SessionId, request: RequestId, spec: CollectiveSpec)
        -> Result<(), GlobalCollectiveError>
    {
        if !self.registered_nodes.contains_key(&session)
            || !self.register_requests.is_empty()
            || self.registered_nodes.len() != self.cur_num_nodes.unwrap_or(0) as usize
        {
            return Err(GlobalCollectiveError::CollectiveBeforeRegister);
        }
        if self.pending_collectives.iter().any(|(s, _, _)| *s == session) {
            return Err(GlobalCollectiveError::CollectiveParamsMismatch);
        }
        self.pending_collectives.push((session, request, spec));
        if self.pending_collectives.len() != self.registered_nodes.len() { return Ok(()); }
        let requests = core::mem::take(&mut self.pending_collectives);
        let specs = requests.iter().map(|(session, _, spec)| (self.registered_nodes[session], spec))
            .collect::<Vec<_>>();
        let content = match agree_collective(&specs, &self.node_peers) {
            Ok(root_node) => {
                let transfer_id = self.next_transfer_id;
                self.next_transfer_id = self.next_transfer_id.checked_add(1)
                    .expect("collective transfer identifier exhausted");
                RemoteResponse::Begin { root_node, transfer_id }
            }
            Err(error) => RemoteResponse::Error(error),
        };
        for (session, request_id, _) in requests {
            self.respond(session, CollectiveMessageResponse { request_id, content: content.clone() }).await;
        }
        Ok(())
    }

}


fn agree_collective(specs: &[(NodeId, &CollectiveSpec)], peers: &HashMap<NodeId, Vec<PeerId>>)
    -> Result<NodeId, GlobalCollectiveError>
{
    use crate::{AllReduceStrategy, BroadcastStrategy, ReduceStrategy};
    let first = specs.first().ok_or(GlobalCollectiveError::CollectiveParamsMismatch)?.1;
    match first {
        CollectiveSpec::AllReduce { strategy, .. } => {
            if specs.iter().any(|(_, spec)| *spec != first) {
                return Err(GlobalCollectiveError::CollectiveParamsMismatch);
            }
            if matches!(strategy, AllReduceStrategy::Tree(0)) {
                return Err(GlobalCollectiveError::InvalidTreeArity);
            }
            peers.keys().copied().min().ok_or(GlobalCollectiveError::CollectiveBeforeRegister)
        }
        CollectiveSpec::Reduce { root, strategy, .. } => {
            if specs.iter().any(|(_, spec)| *spec != first) {
                return Err(GlobalCollectiveError::CollectiveParamsMismatch);
            }
            if matches!(strategy, ReduceStrategy::Tree(0)) {
                return Err(GlobalCollectiveError::InvalidTreeArity);
            }
            peers.iter().find_map(|(node, p)| p.contains(root).then_some(*node))
                .ok_or(GlobalCollectiveError::UnknownRoot(*root))
        }
        CollectiveSpec::Broadcast { strategy, .. } => {
            if matches!(strategy, BroadcastStrategy::Tree(0)) {
                return Err(GlobalCollectiveError::InvalidTreeArity);
            }
            let mut source = None;
            for (node, spec) in specs {
                let CollectiveSpec::Broadcast { strategy: other, metadata } = spec else {
                    return Err(GlobalCollectiveError::CollectiveParamsMismatch);
                };
                if strategy != other { return Err(GlobalCollectiveError::CollectiveParamsMismatch); }
                if metadata.is_some() && source.replace(*node).is_some() {
                    return Err(GlobalCollectiveError::BroadcastMultipleTensors);
                }
            }
            source.ok_or(GlobalCollectiveError::BroadcastNoTensor)
        }
    }
}

#[cfg(test)]
mod rooted_agreement_tests {
    use super::*;
    use crate::{BroadcastStrategy, ReduceOperation, ReduceStrategy};
    use ruda_tensor::DType;

    #[test]
    fn nonzero_root_and_different_peer_counts() {
        let peers = HashMap::from([(NodeId::from(0), vec![PeerId(3), PeerId(5)]),
                                  (NodeId::from(1), vec![PeerId(9)])]);
        let spec = CollectiveSpec::Reduce { root: PeerId(9), op: ReduceOperation::Mean,
            strategy: ReduceStrategy::Tree(2), shape: vec![4], dtype: DType::F32 };
        assert_eq!(agree_collective(&[(0.into(), &spec), (1.into(), &spec)], &peers), Ok(1.into()));
        let mut wrong = spec.clone();
        if let CollectiveSpec::Reduce { dtype, .. } = &mut wrong { *dtype = DType::F64; }
        assert_eq!(agree_collective(&[(0.into(), &spec), (1.into(), &wrong)], &peers),
            Err(GlobalCollectiveError::CollectiveParamsMismatch));
    }

    #[test]
    fn broadcast_requires_one_source_and_matching_kind() {
        let peers = HashMap::from([(NodeId::from(0), vec![PeerId(0)]), (1.into(), vec![PeerId(1)])]);
        let none = CollectiveSpec::Broadcast { strategy: BroadcastStrategy::Centralized, metadata: None };
        let some = CollectiveSpec::Broadcast { strategy: BroadcastStrategy::Centralized,
            metadata: Some((vec![2], DType::F32)) };
        assert_eq!(agree_collective(&[(0.into(), &none), (1.into(), &none)], &peers),
            Err(GlobalCollectiveError::BroadcastNoTensor));
        assert_eq!(agree_collective(&[(0.into(), &some), (1.into(), &some)], &peers),
            Err(GlobalCollectiveError::BroadcastMultipleTensors));
        assert_eq!(agree_collective(&[(0.into(), &none), (1.into(), &some)], &peers), Ok(1.into()));
    }
}

#[cfg(test)]
mod failure_state_tests {
    use super::*;
    #[tokio::test]
    async fn disconnected_member_fails_pending_and_allows_only_fresh_epoch() {
        let mut state=GlobalCollectiveState::new();
        let a=SessionId::new();let b=SessionId::new();
        let mut ar=state.get_session_responder(a).unwrap();
        let mut br=state.get_session_responder(b).unwrap();
        state.register(a,RequestId::new(),"ws://a".parse().unwrap(),2,vec![PeerId(0)]).await.unwrap();
        state.register(b,RequestId::new(),"ws://b".parse().unwrap(),2,vec![PeerId(1)]).await.unwrap();
        ar.recv().await.unwrap();br.recv().await.unwrap();
        let spec=CollectiveSpec::AllReduce{op:crate::ReduceOperation::Sum,
            strategy:crate::AllReduceStrategy::Centralized,shape:vec![2],dtype:ruda_tensor::DType::F32};
        state.begin(a,RequestId::new(),spec.clone()).await.unwrap();
        state.disconnect(b).await;
        assert!(matches!(ar.recv().await.unwrap().content,RemoteResponse::Error(GlobalCollectiveError::PeerLost(_))));
        state.process_request(a,RequestId::new(),RemoteRequest::Begin(spec)).await;
        assert!(matches!(ar.recv().await.unwrap().content,RemoteResponse::Error(_)));
        state.disconnect(a).await;assert!(state.failed.is_none());assert!(state.cur_num_nodes.is_none());
        let fresh=SessionId::new();let mut response=state.get_session_responder(fresh).unwrap();
        state.register(fresh,RequestId::new(),"ws://fresh".parse().unwrap(),1,vec![PeerId(9)]).await.unwrap();
        assert!(matches!(response.recv().await.unwrap().content,RemoteResponse::Register{..}));
    }
    #[test]
    fn duplicate_response_socket_returns_error_not_panic() {
        let mut state=GlobalCollectiveState::new();let session=SessionId::new();
        let _receiver=state.get_session_responder(session).unwrap();
        assert!(state.get_session_responder(session).is_err());
    }
    #[tokio::test]
    async fn closed_response_queue_does_not_panic_under_state_lock() {
        let mut state=GlobalCollectiveState::new();let session=SessionId::new();
        drop(state.get_session_responder(session).unwrap());
        state.respond(session,CollectiveMessageResponse{request_id:RequestId::new(),content:RemoteResponse::FinishAck}).await;
        state.disconnect(session).await;
    }
}
