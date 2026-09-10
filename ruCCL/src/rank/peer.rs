use super::network::{NetworkError, read_frame, write_frame};
use super::protocol::{
    ANY_RANK, ElementType, FLAG_P2P_CHANNEL, Frame, FrameHeader, Opcode, ProtocolError, UniqueId,
};
use std::collections::{HashMap, VecDeque};
use std::io;
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const FAILURE_POLL_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AckKey {
    peer: u32,
    rail: usize,
    sequence: u64,
}

type PendingPeerAcks = Arc<Mutex<HashMap<AckKey, Sender<Result<(), String>>>>>;

#[derive(Debug)]
struct PeerSubmission {
    stream: TcpStream,
    next_sequence: u64,
}

#[derive(Debug)]
struct PeerConnection {
    peer: u32,
    rail: usize,
    submission: Mutex<PeerSubmission>,
}

#[derive(Debug)]
struct QueuedFrame {
    rail: usize,
    frame: Frame,
}

#[derive(Debug, Default)]
struct PeerInbox {
    frames: VecDeque<QueuedFrame>,
}

#[derive(Debug)]
pub(crate) struct DirectPeerMesh {
    unique_id: UniqueId,
    rank: u32,
    world_size: u32,
    rails: usize,
    connections: Vec<Vec<Option<Arc<PeerConnection>>>>,
    inbox: Arc<(Mutex<PeerInbox>, Condvar)>,
    pending_acks: PendingPeerAcks,
    failure: Arc<Mutex<Option<String>>>,
    timeout: Mutex<Duration>,
    readers: Mutex<Vec<JoinHandle<()>>>,
}

impl DirectPeerMesh {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn connect(
        listeners: Vec<TcpListener>,
        endpoints: &[Vec<SocketAddr>],
        unique_id: UniqueId,
        rank: u32,
        world_size: u32,
        rails: usize,
        timeout: Duration,
        failure: Arc<Mutex<Option<String>>>,
    ) -> Result<Self, NetworkError> {
        if endpoints.len() != world_size as usize {
            return Err(NetworkError::InvalidConfiguration(format!(
                "peer endpoint table contains {} ranks, expected {world_size}",
                endpoints.len()
            )));
        }
        if let Some((peer, endpoints)) = endpoints
            .iter()
            .enumerate()
            .find(|(_, endpoints)| endpoints.len() != rails)
        {
            return Err(NetworkError::InvalidConfiguration(format!(
                "peer endpoint table contains {} rail endpoints for rank {peer}, expected {rails}",
                endpoints.len()
            )));
        }
        if listeners.len() != 1 && listeners.len() != rails {
            return Err(NetworkError::InvalidConfiguration(format!(
                "direct peer mesh contains {} listeners, expected one shared listener or {rails} rail listeners",
                listeners.len()
            )));
        }
        let mut connections = (0..rails)
            .map(|_| vec![None; world_size as usize])
            .collect::<Vec<Vec<Option<Arc<PeerConnection>>>>>();

        for peer in 0..rank {
            for (rail, rail_connections) in connections.iter_mut().enumerate() {
                let stream = connect_peer(
                    endpoints[peer as usize][rail],
                    unique_id,
                    rank,
                    peer,
                    world_size,
                    rail,
                    timeout,
                )?;
                rail_connections[peer as usize] = Some(Arc::new(PeerConnection {
                    peer,
                    rail,
                    submission: Mutex::new(PeerSubmission {
                        stream,
                        next_sequence: 0,
                    }),
                }));
            }
        }

        for listener in &listeners {
            listener.set_nonblocking(true)?;
        }
        let expected_incoming = (world_size - rank - 1) as usize * rails;
        let deadline = Instant::now() + timeout;
        let mut accepted = 0_usize;
        while accepted < expected_incoming {
            let mut made_progress = false;
            for (listener_index, listener) in listeners.iter().enumerate() {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        made_progress = true;
                        stream.set_nonblocking(false)?;
                        stream.set_nodelay(true)?;
                        stream.set_read_timeout(Some(timeout))?;
                        stream.set_write_timeout(Some(timeout))?;
                        let join = read_frame(&mut stream)?.ok_or_else(|| {
                            NetworkError::InvalidConfiguration(
                                "direct peer closed before JOIN".into(),
                            )
                        })?;
                        let (peer, rail) =
                            validate_peer_join(&join, unique_id, rank, world_size, rails)?;
                        if listeners.len() == rails && rail != listener_index {
                            return Err(NetworkError::InvalidConfiguration(format!(
                                "direct peer rail {rail} joined dedicated listener {listener_index}"
                            )));
                        }
                        if peer <= rank {
                            return Err(NetworkError::InvalidConfiguration(format!(
                                "rank {rank} accepted direct peer {peer}; only higher ranks connect"
                            )));
                        }
                        if connections[rail][peer as usize].is_some() {
                            return Err(NetworkError::InvalidConfiguration(format!(
                                "direct peer rank {peer} rail {rail} joined twice"
                            )));
                        }
                        let ready = peer_ready_frame(unique_id, rank, peer, world_size, rail)?;
                        write_frame(&mut stream, &ready)?;
                        connections[rail][peer as usize] = Some(Arc::new(PeerConnection {
                            peer,
                            rail,
                            submission: Mutex::new(PeerSubmission {
                                stream,
                                next_sequence: 0,
                            }),
                        }));
                        accepted += 1;
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                    Err(error) => return Err(error.into()),
                }
            }
            if !made_progress {
                if Instant::now() >= deadline {
                    return Err(NetworkError::Timeout(format!(
                        "direct peer mesh setup; accepted {accepted} of {expected_incoming} connections"
                    )));
                }
                thread::park_timeout(Duration::from_millis(1));
            }
        }

        let inbox = Arc::new((Mutex::new(PeerInbox::default()), Condvar::new()));
        let pending_acks = Arc::new(Mutex::new(HashMap::new()));
        let mut readers = Vec::with_capacity((world_size.saturating_sub(1) as usize) * rails);
        for (rail, peers) in connections.iter().enumerate() {
            for (peer, connection) in peers.iter().enumerate() {
                let Some(connection) = connection else {
                    continue;
                };
                let mut read_stream = connection
                    .submission
                    .lock()
                    .map_err(|_| NetworkError::Poisoned)?
                    .stream
                    .try_clone()?;
                read_stream.set_read_timeout(None)?;
                let connection = Arc::clone(connection);
                let inbox = Arc::clone(&inbox);
                let pending_acks = Arc::clone(&pending_acks);
                let failure = Arc::clone(&failure);
                readers.push(
                    thread::Builder::new()
                        .name(format!("gx1-direct-peer-{rank}-{peer}-rail-{rail}"))
                        .spawn(move || {
                            run_peer_reader(
                                &mut read_stream,
                                unique_id,
                                rank,
                                world_size,
                                &connection,
                                &inbox,
                                &pending_acks,
                                &failure,
                            );
                        })
                        .map_err(|error| {
                            NetworkError::InvalidConfiguration(format!(
                                "cannot start direct peer reader for rank {peer} rail {rail}: {error}"
                            ))
                        })?,
                );
            }
        }

        Ok(Self {
            unique_id,
            rank,
            world_size,
            rails,
            connections,
            inbox,
            pending_acks,
            failure,
            timeout: Mutex::new(timeout),
            readers: Mutex::new(readers),
        })
    }

    pub(crate) const fn rails(&self) -> usize {
        self.rails
    }

    pub(crate) fn set_timeout(&self, timeout: Duration) -> Result<(), NetworkError> {
        if timeout.is_zero() {
            return Err(NetworkError::InvalidConfiguration(
                "direct peer timeout must be greater than zero".into(),
            ));
        }
        for peers in &self.connections {
            for connection in peers.iter().flatten() {
                connection
                    .submission
                    .lock()
                    .map_err(|_| NetworkError::Poisoned)?
                    .stream
                    .set_write_timeout(Some(timeout))?;
            }
        }
        *self.timeout.lock().map_err(|_| NetworkError::Poisoned)? = timeout;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn send_on_rail(
        &self,
        rail: usize,
        destination: u32,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
        payload: Vec<u8>,
    ) -> Result<(), NetworkError> {
        self.validate_rail(rail)?;
        self.validate_peer(destination)?;
        self.check_failure()?;
        if destination == self.rank {
            let mut header = FrameHeader::collective(
                self.unique_id,
                Opcode::Send,
                element_type,
                self.rank,
                ANY_RANK,
                self.world_size,
                0,
                element_count,
            );
            header.destination_rank = self.rank;
            header.tag = tag;
            let frame = Frame::new(header, payload)?;
            let (inbox, ready) = &*self.inbox;
            inbox
                .lock()
                .map_err(|_| NetworkError::Poisoned)?
                .frames
                .push_back(QueuedFrame { rail, frame });
            ready.notify_all();
            return Ok(());
        }

        let connection = self.connection(rail, destination)?;
        let timeout = *self.timeout.lock().map_err(|_| NetworkError::Poisoned)?;
        let (sender, receiver) = mpsc::channel();
        let key;
        {
            let mut submission = connection
                .submission
                .lock()
                .map_err(|_| NetworkError::Poisoned)?;
            let sequence = submission.next_sequence;
            submission.next_sequence = sequence
                .checked_add(1)
                .ok_or(ProtocolError::SequenceOverflow)?;
            key = AckKey {
                peer: destination,
                rail,
                sequence,
            };
            let mut header = FrameHeader::collective(
                self.unique_id,
                Opcode::Send,
                element_type,
                self.rank,
                ANY_RANK,
                self.world_size,
                sequence,
                element_count,
            );
            header.destination_rank = destination;
            header.tag = tag;
            let frame = Frame::new(header, payload)?;
            self.pending_acks
                .lock()
                .map_err(|_| NetworkError::Poisoned)?
                .insert(key, sender);
            if let Err(error) = write_frame(&mut submission.stream, &frame) {
                set_peer_failure(
                    &self.failure,
                    &self.inbox,
                    &self.pending_acks,
                    &format!("direct peer send to rank {destination} rail {rail} failed: {error}"),
                );
                return Err(error);
            }
        }
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(message) = self.failure_message()? {
                if let Ok(mut pending) = self.pending_acks.lock() {
                    pending.remove(&key);
                }
                return Err(NetworkError::RemoteAbort(message));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                if let Ok(mut pending) = self.pending_acks.lock() {
                    pending.remove(&key);
                }
                let message = format!(
                    "direct peer send to rank {destination} rail {rail} with tag {tag} timed out"
                );
                set_peer_failure(&self.failure, &self.inbox, &self.pending_acks, &message);
                return Err(NetworkError::Timeout(message));
            }
            match receiver.recv_timeout(remaining.min(FAILURE_POLL_INTERVAL)) {
                Ok(Ok(())) => return Ok(()),
                Ok(Err(message)) => return Err(NetworkError::RemoteAbort(message)),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(NetworkError::ChannelClosed);
                }
            }
        }
    }

    pub(crate) fn receive_on_rail(
        &self,
        rail: usize,
        source: Option<u32>,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
    ) -> Result<Frame, NetworkError> {
        self.validate_rail(rail)?;
        if let Some(source) = source {
            self.validate_peer(source)?;
        }
        let timeout = *self.timeout.lock().map_err(|_| NetworkError::Poisoned)?;
        let deadline = Instant::now() + timeout;
        let (inbox, ready) = &*self.inbox;
        let mut inbox = inbox.lock().map_err(|_| NetworkError::Poisoned)?;
        loop {
            if let Some(message) = self.failure_message()? {
                return Err(NetworkError::RemoteAbort(message));
            }
            if let Some(index) = inbox.frames.iter().position(|queued| {
                queued.rail == rail
                    && queued.frame.header.tag == tag
                    && source.is_none_or(|source| queued.frame.header.source_rank == source)
            }) {
                let queued = inbox
                    .frames
                    .remove(index)
                    .expect("matched peer frame exists");
                if queued.frame.header.element_type != element_type
                    || queued.frame.header.element_count != element_count
                {
                    return Err(NetworkError::InvalidConfiguration(format!(
                        "direct peer tag {tag} contract mismatch: send {:?}/{} receive {:?}/{}",
                        queued.frame.header.element_type,
                        queued.frame.header.element_count,
                        element_type,
                        element_count
                    )));
                }
                return Ok(queued.frame);
            }
            let now = Instant::now();
            if now >= deadline {
                drop(inbox);
                let message =
                    format!("direct peer receive on rail {rail} with tag {tag} timed out");
                set_peer_failure(&self.failure, &self.inbox, &self.pending_acks, &message);
                return Err(NetworkError::Timeout(message));
            }
            let wait = deadline
                .saturating_duration_since(now)
                .min(FAILURE_POLL_INTERVAL);
            let (next, _) = ready
                .wait_timeout(inbox, wait)
                .map_err(|_| NetworkError::Poisoned)?;
            inbox = next;
        }
    }

    pub(crate) fn abort(&self, message: &str) -> Result<(), NetworkError> {
        set_peer_failure(&self.failure, &self.inbox, &self.pending_acks, message);
        let mut first_error = None;
        for peers in &self.connections {
            for connection in peers.iter().flatten() {
                let payload = message.as_bytes().to_vec();
                let mut header = FrameHeader::collective(
                    self.unique_id,
                    Opcode::Abort,
                    ElementType::U8,
                    self.rank,
                    ANY_RANK,
                    self.world_size,
                    0,
                    payload.len() as u64,
                );
                header.destination_rank = connection.peer;
                let result = connection
                    .submission
                    .lock()
                    .map_err(|_| NetworkError::Poisoned)
                    .and_then(|mut submission| {
                        write_frame(&mut submission.stream, &Frame::new(header, payload)?)
                    });
                if let Err(error) = result
                    && first_error.is_none()
                {
                    first_error = Some(error);
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn connection(&self, rail: usize, peer: u32) -> Result<&Arc<PeerConnection>, NetworkError> {
        self.connections[rail][peer as usize]
            .as_ref()
            .ok_or_else(|| {
                NetworkError::InvalidConfiguration(format!(
                    "direct peer connection to rank {peer} rail {rail} is absent"
                ))
            })
    }

    fn validate_peer(&self, peer: u32) -> Result<(), NetworkError> {
        if peer < self.world_size {
            Ok(())
        } else {
            Err(ProtocolError::RankOutOfRange {
                name: "peer",
                rank: peer,
                world_size: self.world_size,
            }
            .into())
        }
    }

    fn validate_rail(&self, rail: usize) -> Result<(), NetworkError> {
        if rail < self.rails {
            Ok(())
        } else {
            Err(NetworkError::InvalidConfiguration(format!(
                "direct peer rail {rail} is outside 0..{}",
                self.rails
            )))
        }
    }

    fn failure_message(&self) -> Result<Option<String>, NetworkError> {
        Ok(self
            .failure
            .lock()
            .map_err(|_| NetworkError::Poisoned)?
            .clone())
    }

    fn check_failure(&self) -> Result<(), NetworkError> {
        match self.failure_message()? {
            Some(message) => Err(NetworkError::RemoteAbort(message)),
            None => Ok(()),
        }
    }
}

impl Drop for DirectPeerMesh {
    fn drop(&mut self) {
        for peers in &self.connections {
            for connection in peers.iter().flatten() {
                if let Ok(mut submission) = connection.submission.lock() {
                    let mut header = FrameHeader::collective(
                        self.unique_id,
                        Opcode::Leave,
                        ElementType::None,
                        self.rank,
                        ANY_RANK,
                        self.world_size,
                        0,
                        0,
                    );
                    header.destination_rank = connection.peer;
                    if let Ok(frame) = Frame::new(header, Vec::new()) {
                        let _ = write_frame(&mut submission.stream, &frame);
                    }
                    let _ = submission.stream.shutdown(Shutdown::Both);
                }
            }
        }
        if let Ok(readers) = self.readers.get_mut() {
            for reader in readers.drain(..) {
                let _ = reader.join();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn connect_peer(
    endpoint: SocketAddr,
    unique_id: UniqueId,
    rank: u32,
    peer: u32,
    world_size: u32,
    rail: usize,
    timeout: Duration,
) -> Result<TcpStream, NetworkError> {
    let mut stream = TcpStream::connect_timeout(&endpoint, timeout)?;
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    let mut header = FrameHeader::collective(
        unique_id,
        Opcode::Join,
        ElementType::None,
        rank,
        ANY_RANK,
        world_size,
        0,
        0,
    );
    header.destination_rank = peer;
    header.flags |= FLAG_P2P_CHANNEL;
    header.tag = rail as u64;
    write_frame(&mut stream, &Frame::new(header, Vec::new())?)?;
    let ready = read_frame(&mut stream)?.ok_or_else(|| {
        NetworkError::InvalidConfiguration("direct peer closed before READY".into())
    })?;
    if ready.header.opcode != Opcode::Ready
        || ready.header.unique_id != unique_id
        || ready.header.source_rank != peer
        || ready.header.destination_rank != rank
        || ready.header.world_size != world_size
        || ready.header.flags & FLAG_P2P_CHANNEL == 0
        || ready.header.tag != rail as u64
        || !ready.payload.is_empty()
    {
        return Err(NetworkError::InvalidConfiguration(format!(
            "invalid READY from direct peer rank {peer} rail {rail}"
        )));
    }
    Ok(stream)
}

fn validate_peer_join(
    frame: &Frame,
    unique_id: UniqueId,
    rank: u32,
    world_size: u32,
    rails: usize,
) -> Result<(u32, usize), NetworkError> {
    if frame.header.opcode != Opcode::Join
        || frame.header.element_type != ElementType::None
        || frame.header.unique_id != unique_id
        || frame.header.destination_rank != rank
        || frame.header.world_size != world_size
        || frame.header.flags & FLAG_P2P_CHANNEL == 0
        || !frame.payload.is_empty()
    {
        return Err(NetworkError::InvalidConfiguration(
            "invalid direct peer JOIN".into(),
        ));
    }
    let rail = usize::try_from(frame.header.tag)
        .map_err(|_| NetworkError::InvalidConfiguration("direct peer rail exceeds usize".into()))?;
    if rail >= rails {
        return Err(NetworkError::InvalidConfiguration(format!(
            "direct peer rail {rail} is outside 0..{rails}"
        )));
    }
    Ok((frame.header.source_rank, rail))
}

fn peer_ready_frame(
    unique_id: UniqueId,
    rank: u32,
    peer: u32,
    world_size: u32,
    rail: usize,
) -> Result<Frame, NetworkError> {
    let mut header = FrameHeader::collective(
        unique_id,
        Opcode::Ready,
        ElementType::None,
        rank,
        ANY_RANK,
        world_size,
        0,
        0,
    );
    header.destination_rank = peer;
    header.flags |= FLAG_P2P_CHANNEL;
    header.tag = rail as u64;
    Ok(Frame::new(header, Vec::new())?)
}

#[allow(clippy::too_many_arguments)]
fn run_peer_reader(
    stream: &mut TcpStream,
    unique_id: UniqueId,
    rank: u32,
    world_size: u32,
    connection: &Arc<PeerConnection>,
    inbox: &Arc<(Mutex<PeerInbox>, Condvar)>,
    pending_acks: &PendingPeerAcks,
    failure: &Arc<Mutex<Option<String>>>,
) {
    loop {
        let frame = match read_frame(stream) {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                set_peer_failure(
                    failure,
                    inbox,
                    pending_acks,
                    &format!(
                        "direct peer rank {} rail {} closed",
                        connection.peer, connection.rail
                    ),
                );
                return;
            }
            Err(error) => {
                set_peer_failure(
                    failure,
                    inbox,
                    pending_acks,
                    &format!(
                        "direct peer rank {} rail {} read failed: {error}",
                        connection.peer, connection.rail
                    ),
                );
                return;
            }
        };
        if frame.header.unique_id != unique_id
            || frame.header.world_size != world_size
            || frame.header.source_rank != connection.peer
            || frame.header.destination_rank != rank
            || frame.header.root_rank != ANY_RANK
        {
            set_peer_failure(
                failure,
                inbox,
                pending_acks,
                "invalid direct peer frame envelope",
            );
            return;
        }
        match frame.header.opcode {
            Opcode::Send if frame.header.element_type == ElementType::None => {
                if !frame.payload.is_empty() || frame.header.element_count != 0 {
                    set_peer_failure(
                        failure,
                        inbox,
                        pending_acks,
                        "invalid direct peer send acknowledgement",
                    );
                    return;
                }
                let key = AckKey {
                    peer: connection.peer,
                    rail: connection.rail,
                    sequence: frame.header.sequence,
                };
                if let Ok(mut pending) = pending_acks.lock()
                    && let Some(sender) = pending.remove(&key)
                {
                    let _ = sender.send(Ok(()));
                }
            }
            Opcode::Send => {
                let acknowledgement = match peer_send_acknowledgement(
                    unique_id,
                    rank,
                    connection.peer,
                    world_size,
                    &frame,
                ) {
                    Ok(frame) => frame,
                    Err(error) => {
                        set_peer_failure(failure, inbox, pending_acks, &error.to_string());
                        return;
                    }
                };
                let (queue, ready) = &**inbox;
                match queue.lock() {
                    Ok(mut queue) => queue.frames.push_back(QueuedFrame {
                        rail: connection.rail,
                        frame,
                    }),
                    Err(_) => return,
                }
                ready.notify_all();
                let write_result = connection
                    .submission
                    .lock()
                    .map_err(|_| NetworkError::Poisoned)
                    .and_then(|mut submission| {
                        write_frame(&mut submission.stream, &acknowledgement)
                    });
                if let Err(error) = write_result {
                    set_peer_failure(
                        failure,
                        inbox,
                        pending_acks,
                        &format!("direct peer acknowledgement failed: {error}"),
                    );
                    return;
                }
            }
            Opcode::Abort => {
                set_peer_failure(
                    failure,
                    inbox,
                    pending_acks,
                    &String::from_utf8_lossy(&frame.payload),
                );
                return;
            }
            Opcode::Leave => {
                set_peer_failure(
                    failure,
                    inbox,
                    pending_acks,
                    &format!("direct peer rank {} left", connection.peer),
                );
                return;
            }
            opcode => {
                set_peer_failure(
                    failure,
                    inbox,
                    pending_acks,
                    &format!("unexpected {opcode:?} on direct peer connection"),
                );
                return;
            }
        }
    }
}

fn peer_send_acknowledgement(
    unique_id: UniqueId,
    rank: u32,
    peer: u32,
    world_size: u32,
    send: &Frame,
) -> Result<Frame, NetworkError> {
    let mut header = FrameHeader::collective(
        unique_id,
        Opcode::Send,
        ElementType::None,
        rank,
        ANY_RANK,
        world_size,
        send.header.sequence,
        0,
    );
    header.destination_rank = peer;
    header.tag = send.header.tag;
    Ok(Frame::new(header, Vec::new())?)
}

fn set_peer_failure(
    failure: &Arc<Mutex<Option<String>>>,
    inbox: &Arc<(Mutex<PeerInbox>, Condvar)>,
    pending_acks: &PendingPeerAcks,
    message: &str,
) {
    if let Ok(mut failure) = failure.lock() {
        failure.get_or_insert_with(|| message.to_owned());
    }
    if let Ok(mut pending) = pending_acks.lock() {
        for (_, sender) in pending.drain() {
            let _ = sender.send(Err(message.to_owned()));
        }
    }
    inbox.1.notify_all();
}
