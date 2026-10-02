//! Single-owner control pump. Requests are removed after one response; no
//! unchecked callbacks, infinite reconnect loops, or post-dispatch retries.
use std::{collections::HashMap, marker::PhantomData};
use ruda_communication::{Address, CommunicationChannel, Message, ProtocolClient};
use tokio::{runtime::Runtime, sync::mpsc::{Receiver, Sender}, task::JoinHandle};
use tokio_util::sync::CancellationToken;
use crate::global::{policy::GlobalFailurePolicy, shared::{CollectiveMessage,
    CollectiveMessageResponse, GlobalCollectiveError, RemoteRequest, RemoteResponse, RequestId, SessionId}};

pub(crate) struct GlobalClientWorker<P: ProtocolClient> {
    handle: Option<JoinHandle<Result<(), GlobalCollectiveError>>>,
    cancel_token: CancellationToken,
    request_sender: Sender<ClientRequest>,
    policy: GlobalFailurePolicy,
    _phantom_data: PhantomData<P>,
}
struct ClientRequest { request: RemoteRequest, callback: Sender<RemoteResponse> }

impl<C: ProtocolClient> GlobalClientWorker<C> {
    pub(crate) fn new(runtime: &Runtime, cancel_token: CancellationToken, address: &Address) -> Self {
        let policy = GlobalFailurePolicy::from_environment().expect("invalid ruCCL failure policy");
        let (request_sender, request_recv) = tokio::sync::mpsc::channel(10);
        let handle = runtime.spawn(Self::start(cancel_token.clone(), address.clone(), request_recv, policy));
        Self { handle: Some(handle), cancel_token, request_sender, policy, _phantom_data: PhantomData }
    }
    pub(crate) fn policy(&self) -> GlobalFailurePolicy { self.policy }
    pub(crate) fn token(&self) -> CancellationToken { self.cancel_token.clone() }
    pub(crate) fn abort(&self) { self.cancel_token.cancel(); }

    async fn connect(address: Address, route: &str, session: SessionId,
        token: CancellationToken, policy: GlobalFailurePolicy) -> Result<C::Channel, GlobalCollectiveError>
    {
        for attempt in 0..policy.connect_attempts {
            let stream = tokio::select! {
                _ = token.cancelled() => return Err(GlobalCollectiveError::CommunicatorAborted),
                value = tokio::time::timeout(policy.connect_timeout, C::connect(address.clone(), route)) => value,
            };
            if let Ok(Some(mut stream)) = stream {
                let bytes = rmp_serde::to_vec(&CollectiveMessage::Init(session))
                    .map_err(|_| GlobalCollectiveError::InvalidMessage)?;
                // Once Init is submitted, an ambiguous send must abort rather
                // than retry on another socket with the same session identity.
                tokio::select! {
                    _ = token.cancelled() => return Err(GlobalCollectiveError::CommunicatorAborted),
                    value = tokio::time::timeout(policy.connect_timeout, stream.send(Message::new(bytes.into()))) => {
                        value.map_err(|_| GlobalCollectiveError::OperationTimeout)??;
                    }
                }
                return Ok(stream);
            }
            if attempt + 1 < policy.connect_attempts {
                tokio::select! {
                    _ = token.cancelled() => return Err(GlobalCollectiveError::CommunicatorAborted),
                    _ = tokio::time::sleep(policy.backoff(attempt)) => {}
                }
            }
        }
        Err(GlobalCollectiveError::OrchestratorUnreachable)
    }

    async fn start(token: CancellationToken, address: Address, mut requests: Receiver<ClientRequest>,
        policy: GlobalFailurePolicy) -> Result<(), GlobalCollectiveError>
    {
        let mut pending: HashMap<RequestId, Sender<RemoteResponse>> = HashMap::new();
        let result = async {
            let id = SessionId::new();
            let (mut send, mut recv) = tokio::try_join!(
                Self::connect(address.clone(), "request", id, token.clone(), policy),
                Self::connect(address, "response", id, token.clone(), policy))?;
            loop {
                tokio::select! {
                    _ = token.cancelled() => return Err(GlobalCollectiveError::CommunicatorAborted),
                    request = requests.recv() => {
                        let Some(request) = request else { return Ok(()); };
                        if request.callback.is_closed() { continue; }
                        let id = RequestId::new();
                        pending.insert(id, request.callback);
                        let bytes = rmp_serde::to_vec(&CollectiveMessage::Request(id, request.request))
                            .map_err(|_| GlobalCollectiveError::InvalidMessage)?;
                        tokio::time::timeout(policy.request_timeout, send.send(Message::new(bytes.into())))
                            .await.map_err(|_| GlobalCollectiveError::OperationTimeout)??;
                    }
                    packet = recv.recv() => {
                        let packet = packet?.ok_or(GlobalCollectiveError::CommunicatorAborted)?;
                        let response: CollectiveMessageResponse = rmp_serde::from_slice(&packet.data)
                            .map_err(|_| GlobalCollectiveError::InvalidMessage)?;
                        // Duplicate, stale or unsolicited replies are protocol errors,
                        // never a second completion or an unbounded map leak.
                        let callback = pending.remove(&response.request_id)
                            .ok_or(GlobalCollectiveError::WrongOrchestratorResponse)?;
                        let _ = callback.try_send(response.content);
                    }
                }
            }
        }.await;
        token.cancel(); // Also stops this node's data/sync listener.
        let error = result.as_ref().err().cloned().unwrap_or(GlobalCollectiveError::CommunicatorAborted);
        for (_, callback) in pending { let _ = callback.try_send(RemoteResponse::Error(error.clone())); }
        while let Ok(request) = requests.try_recv() {
            let _ = request.callback.try_send(RemoteResponse::Error(error.clone()));
        }
        result
    }

    pub(crate) async fn request(&self, request: RemoteRequest) -> RemoteResponse {
        if self.cancel_token.is_cancelled() {
            return RemoteResponse::Error(GlobalCollectiveError::CommunicatorAborted);
        }
        let wait = async {
            let (callback, mut recv) = tokio::sync::mpsc::channel(1);
            self.request_sender.send(ClientRequest { request, callback }).await
                .map_err(|_| GlobalCollectiveError::CommunicatorAborted)?;
            recv.recv().await.ok_or(GlobalCollectiveError::CommunicatorAborted)
        };
        // Cancellation of the CALLER also poisons the epoch. A dropped request
        // may already be executing remotely, so it cannot be ignored or replayed.
        let mut guard = CancelOnDrop { token: self.token(), completed: false };
        let result = tokio::select! {
            _ = self.cancel_token.cancelled() => Err(GlobalCollectiveError::CommunicatorAborted),
            value = tokio::time::timeout(self.policy.request_timeout, wait) =>
                value.map_err(|_| GlobalCollectiveError::OperationTimeout).and_then(|x| x),
        };
        guard.completed = result.is_ok();
        result.unwrap_or_else(RemoteResponse::Error)
    }

    pub(crate) async fn close_connection(&mut self) -> Result<(), GlobalCollectiveError> {
        let Some(mut handle) = self.handle.take() else { return Ok(()); };
        let response = if self.cancel_token.is_cancelled() { RemoteResponse::FinishAck }
            else { self.request(RemoteRequest::Finish).await };
        self.abort();
        if tokio::time::timeout(self.policy.request_timeout, &mut handle).await.is_err() { handle.abort(); }
        if response == RemoteResponse::FinishAck { Ok(()) }
        else { Err(GlobalCollectiveError::WrongOrchestratorResponse) }
    }
}
impl<C: ProtocolClient> Drop for GlobalClientWorker<C> {
    fn drop(&mut self) { self.cancel_token.cancel(); if let Some(handle) = &self.handle { handle.abort(); } }
}

pub(super) struct CancelOnDrop { pub token: CancellationToken, pub completed: bool }
impl Drop for CancelOnDrop {
    fn drop(&mut self) { if !self.completed { self.token.cancel(); } }
}

#[cfg(test)]
mod bounded_tests {
    use super::*;
    use ruda_core::future::DynFut;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[derive(Debug)] struct MockError;
    impl ruda_communication::CommunicationError for MockError {}
    struct Channel;
    impl CommunicationChannel for Channel {
        type Error = MockError;
        async fn send(&mut self, _: Message) -> Result<(), MockError> { Err(MockError) }
        async fn recv(&mut self) -> Result<Option<Message>, MockError> { std::future::pending().await }
        async fn close(&mut self) -> Result<(), MockError> { Ok(()) }
    }
    static REFUSED: AtomicUsize = AtomicUsize::new(0);
    struct Refused;
    impl ProtocolClient for Refused {
        type Channel = Channel; type Error = MockError;
        fn connect(_: Address, _: &str) -> DynFut<Option<Channel>> {
            REFUSED.fetch_add(1, Ordering::SeqCst); Box::pin(async {None})
        }
    }
    static HANDSHAKE: AtomicUsize = AtomicUsize::new(0);
    struct AmbiguousHandshake;
    impl ProtocolClient for AmbiguousHandshake {
        type Channel = Channel; type Error = MockError;
        fn connect(_: Address, _: &str) -> DynFut<Option<Channel>> {
            HANDSHAKE.fetch_add(1, Ordering::SeqCst); Box::pin(async {Some(Channel)})
        }
    }
    fn policy() -> GlobalFailurePolicy {
        GlobalFailurePolicy { connect_attempts:3, connect_timeout:std::time::Duration::from_millis(20),
            retry_backoff:std::time::Duration::from_millis(1), ..Default::default() }
    }
    #[tokio::test]
    async fn refused_startup_has_finite_attempt_count() {
        let result=GlobalClientWorker::<Refused>::connect("unused".parse().unwrap(),"request",SessionId::new(),
            CancellationToken::new(),policy()).await;
        assert!(matches!(result,Err(GlobalCollectiveError::OrchestratorUnreachable)));
        assert_eq!(REFUSED.load(Ordering::SeqCst),3);
    }
    #[tokio::test]
    async fn ambiguous_handshake_is_never_retried() {
        let result=GlobalClientWorker::<AmbiguousHandshake>::connect("unused".parse().unwrap(),"request",SessionId::new(),
            CancellationToken::new(),policy()).await;
        assert!(result.is_err());assert_eq!(HANDSHAKE.load(Ordering::SeqCst),1);
    }
    #[test]
    fn dropped_pending_call_poisoned_but_completed_call_not_poisoned() {
        let token=CancellationToken::new();
        drop(CancelOnDrop{token:token.clone(),completed:true});assert!(!token.is_cancelled());
        drop(CancelOnDrop{token:token.clone(),completed:false});assert!(token.is_cancelled());
    }
}
