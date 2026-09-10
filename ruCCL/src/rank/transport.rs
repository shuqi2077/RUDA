//! Transport boundary used by the rank-local collective scheduler.
//!
//! Implementations own both the ordered collective control path and the tagged
//! point-to-point data rails.  `TcpRankSession` is the current implementation;
//! PCIe peer, RDMA and GX-Link implementations can provide the same contract
//! without changing collective algorithms or their GX reduction kernels.

use super::{
    ANY_RANK, CollectiveTransport, ElementType, ExchangeOptions, Frame, NetworkError, Opcode,
};
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt::Debug;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const TRANSPORT_TRACE_SCHEMA_VERSION: u16 = 1;
static NEXT_TRACE_FILE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportTraceOperation {
    Heartbeat,
    Send,
    Receive,
    Exchange,
    SetTimeout,
    Abort,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportTraceEvent {
    pub schema_version: u16,
    pub timestamp_unix_ns: u64,
    pub duration_ns: u64,
    pub rank: u32,
    pub world_size: u32,
    pub transport: CollectiveTransport,
    pub operation: TransportTraceOperation,
    pub opcode: Option<String>,
    pub element_type: Option<String>,
    pub peer_rank: Option<u32>,
    pub rail: Option<usize>,
    pub tag: u64,
    pub element_count: u64,
    pub input_bytes: usize,
    pub output_bytes: usize,
    pub success: bool,
    pub error: Option<String>,
}

pub trait TransportTraceSink: Debug + Send + Sync {
    fn record(&self, event: &TransportTraceEvent);
}

#[derive(Debug)]
pub struct JsonlTransportTraceSink {
    path: PathBuf,
    file: Mutex<File>,
    failed: AtomicBool,
}

impl JsonlTransportTraceSink {
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            file: Mutex::new(file),
            failed: AtomicBool::new(false),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl TransportTraceSink for JsonlTransportTraceSink {
    fn record(&self, event: &TransportTraceEvent) {
        if self.failed.load(Ordering::Relaxed) {
            return;
        }
        let encoded = match serde_json::to_vec(event) {
            Ok(mut encoded) => {
                encoded.push(b'\n');
                encoded
            }
            Err(error) => {
                if !self.failed.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "GX collective trace serialization failed for {}: {error}",
                        self.path.display()
                    );
                }
                return;
            }
        };
        let result = self
            .file
            .lock()
            .map_err(|_| io::Error::other("GX collective trace file lock poisoned"))
            .and_then(|mut file| {
                file.write_all(&encoded)?;
                file.flush()
            });
        if let Err(error) = result
            && !self.failed.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "GX collective trace write failed for {}: {error}",
                self.path.display()
            );
        }
    }
}

#[derive(Debug)]
pub struct TracingRankTransport<T> {
    inner: T,
    sink: Arc<dyn TransportTraceSink>,
}

impl<T> TracingRankTransport<T> {
    pub fn new(inner: T, sink: Arc<dyn TransportTraceSink>) -> Self {
        Self { inner, sink }
    }

    pub fn with_jsonl(inner: T, path: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self::new(
            inner,
            Arc::new(JsonlTransportTraceSink::create(path)?),
        ))
    }

    pub fn inner(&self) -> &T {
        &self.inner
    }
}

pub trait RankTransport: Debug + Send + Sync {
    fn rank(&self) -> u32;

    fn world_size(&self) -> u32;

    fn transport(&self) -> CollectiveTransport;

    fn p2p_rails(&self) -> usize;

    fn heartbeat(&self, timeout: Duration) -> Result<Duration, NetworkError>;

    #[allow(clippy::too_many_arguments)]
    fn send_on_rail(
        &self,
        rail: usize,
        destination: u32,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
        payload: Vec<u8>,
    ) -> Result<(), NetworkError>;

    fn receive_on_rail(
        &self,
        rail: usize,
        source: Option<u32>,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
    ) -> Result<Frame, NetworkError>;

    fn exchange_with_options(
        &self,
        opcode: Opcode,
        element_type: ElementType,
        options: ExchangeOptions,
        payload: Vec<u8>,
    ) -> Result<Frame, NetworkError>;

    fn set_timeout(&self, timeout: Duration) -> Result<(), NetworkError>;

    fn abort(&self, message: &str) -> Result<(), NetworkError>;

    fn send(
        &self,
        destination: u32,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
        payload: Vec<u8>,
    ) -> Result<(), NetworkError> {
        let rail = (tag % self.p2p_rails() as u64) as usize;
        self.send_on_rail(rail, destination, tag, element_type, element_count, payload)
    }

    fn receive(
        &self,
        source: Option<u32>,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
    ) -> Result<Frame, NetworkError> {
        let rail = (tag % self.p2p_rails() as u64) as usize;
        self.receive_on_rail(rail, source, tag, element_type, element_count)
    }

    fn exchange(
        &self,
        opcode: Opcode,
        element_type: ElementType,
        root_rank: u32,
        element_count: u64,
        payload: Vec<u8>,
    ) -> Result<Frame, NetworkError> {
        self.exchange_with_options(
            opcode,
            element_type,
            ExchangeOptions::new(root_rank, element_count),
            payload,
        )
    }

    fn barrier(&self) -> Result<(), NetworkError> {
        let response =
            self.exchange(Opcode::Barrier, ElementType::None, ANY_RANK, 0, Vec::new())?;
        if response.header.opcode != Opcode::Barrier || !response.payload.is_empty() {
            return Err(NetworkError::InvalidConfiguration(
                "invalid barrier response".into(),
            ));
        }
        Ok(())
    }
}

impl<T> TracingRankTransport<T>
where
    T: RankTransport,
{
    #[allow(clippy::too_many_arguments)]
    fn record<R>(
        &self,
        started: Instant,
        operation: TransportTraceOperation,
        opcode: Option<Opcode>,
        element_type: Option<ElementType>,
        peer_rank: Option<u32>,
        rail: Option<usize>,
        tag: u64,
        element_count: u64,
        input_bytes: usize,
        output_bytes: impl FnOnce(&R) -> usize,
        result: &Result<R, NetworkError>,
    ) {
        let event = TransportTraceEvent {
            schema_version: TRANSPORT_TRACE_SCHEMA_VERSION,
            timestamp_unix_ns: unix_timestamp_ns(),
            duration_ns: duration_ns(started.elapsed()),
            rank: self.inner.rank(),
            world_size: self.inner.world_size(),
            transport: self.inner.transport(),
            operation,
            opcode: opcode.map(opcode_name).map(str::to_owned),
            element_type: element_type.map(element_type_name).map(str::to_owned),
            peer_rank,
            rail,
            tag,
            element_count,
            input_bytes,
            output_bytes: result.as_ref().map_or(0, output_bytes),
            success: result.is_ok(),
            error: result.as_ref().err().map(ToString::to_string),
        };
        self.sink.record(&event);
    }
}

impl<T> RankTransport for TracingRankTransport<T>
where
    T: RankTransport,
{
    fn rank(&self) -> u32 {
        self.inner.rank()
    }

    fn world_size(&self) -> u32 {
        self.inner.world_size()
    }

    fn transport(&self) -> CollectiveTransport {
        self.inner.transport()
    }

    fn p2p_rails(&self) -> usize {
        self.inner.p2p_rails()
    }

    fn heartbeat(&self, timeout: Duration) -> Result<Duration, NetworkError> {
        let started = Instant::now();
        let result = self.inner.heartbeat(timeout);
        self.record(
            started,
            TransportTraceOperation::Heartbeat,
            Some(Opcode::Heartbeat),
            Some(ElementType::None),
            None,
            None,
            timeout.as_millis().try_into().unwrap_or(u64::MAX),
            0,
            0,
            |_| 0,
            &result,
        );
        result
    }

    fn send_on_rail(
        &self,
        rail: usize,
        destination: u32,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
        payload: Vec<u8>,
    ) -> Result<(), NetworkError> {
        let input_bytes = payload.len();
        let started = Instant::now();
        let result =
            self.inner
                .send_on_rail(rail, destination, tag, element_type, element_count, payload);
        self.record(
            started,
            TransportTraceOperation::Send,
            Some(Opcode::Send),
            Some(element_type),
            Some(destination),
            Some(rail),
            tag,
            element_count,
            input_bytes,
            |_| 0,
            &result,
        );
        result
    }

    fn receive_on_rail(
        &self,
        rail: usize,
        source: Option<u32>,
        tag: u64,
        element_type: ElementType,
        element_count: u64,
    ) -> Result<Frame, NetworkError> {
        let started = Instant::now();
        let result = self
            .inner
            .receive_on_rail(rail, source, tag, element_type, element_count);
        self.record(
            started,
            TransportTraceOperation::Receive,
            Some(Opcode::Receive),
            Some(element_type),
            source,
            Some(rail),
            tag,
            element_count,
            0,
            |frame| frame.payload.len(),
            &result,
        );
        result
    }

    fn exchange_with_options(
        &self,
        opcode: Opcode,
        element_type: ElementType,
        options: ExchangeOptions,
        payload: Vec<u8>,
    ) -> Result<Frame, NetworkError> {
        let input_bytes = payload.len();
        let started = Instant::now();
        let result = self
            .inner
            .exchange_with_options(opcode, element_type, options, payload);
        self.record(
            started,
            TransportTraceOperation::Exchange,
            Some(opcode),
            Some(element_type),
            (options.root_rank != ANY_RANK).then_some(options.root_rank),
            None,
            options.tag,
            options.element_count,
            input_bytes,
            |frame| frame.payload.len(),
            &result,
        );
        result
    }

    fn set_timeout(&self, timeout: Duration) -> Result<(), NetworkError> {
        let started = Instant::now();
        let result = self.inner.set_timeout(timeout);
        self.record(
            started,
            TransportTraceOperation::SetTimeout,
            Some(Opcode::SetTimeout),
            Some(ElementType::None),
            None,
            None,
            timeout.as_millis().try_into().unwrap_or(u64::MAX),
            0,
            0,
            |_| 0,
            &result,
        );
        result
    }

    fn abort(&self, message: &str) -> Result<(), NetworkError> {
        let started = Instant::now();
        let result = self.inner.abort(message);
        self.record(
            started,
            TransportTraceOperation::Abort,
            Some(Opcode::Abort),
            Some(ElementType::U8),
            None,
            None,
            0,
            message.len() as u64,
            message.len(),
            |_| 0,
            &result,
        );
        result
    }
}

pub fn rank_transport_from_environment<T>(
    transport: T,
) -> Result<Arc<dyn RankTransport>, NetworkError>
where
    T: RankTransport + 'static,
{
    let trace_directory = match env::var("GX1_COLLECTIVE_TRACE_DIR") {
        Ok(value) if value.trim().is_empty() => {
            return Err(NetworkError::InvalidConfiguration(
                "GX1_COLLECTIVE_TRACE_DIR must not be empty".into(),
            ));
        }
        Ok(value) => PathBuf::from(value),
        Err(env::VarError::NotPresent) => return Ok(Arc::new(transport)),
        Err(env::VarError::NotUnicode(_)) => {
            return Err(NetworkError::InvalidConfiguration(
                "GX1_COLLECTIVE_TRACE_DIR is not valid Unicode".into(),
            ));
        }
    };
    let trace_file_id = NEXT_TRACE_FILE_ID.fetch_add(1, Ordering::Relaxed);
    let path = trace_directory.join(format!(
        "gxcl-rank-{}-pid-{}-{trace_file_id}.jsonl",
        transport.rank(),
        std::process::id()
    ));
    let traced = TracingRankTransport::with_jsonl(transport, path)?;
    Ok(Arc::new(traced))
}

fn unix_timestamp_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, duration_ns)
}

fn duration_ns(duration: Duration) -> u64 {
    duration.as_nanos().try_into().unwrap_or(u64::MAX)
}

fn opcode_name(opcode: Opcode) -> &'static str {
    match opcode {
        Opcode::Join => "join",
        Opcode::Ready => "ready",
        Opcode::Broadcast => "broadcast",
        Opcode::AllGather => "all_gather",
        Opcode::Gather => "gather",
        Opcode::Scatter => "scatter",
        Opcode::Reduce => "reduce",
        Opcode::AllReduce => "all_reduce",
        Opcode::ReduceScatter => "reduce_scatter",
        Opcode::AllToAll => "all_to_all",
        Opcode::AllToAllV => "all_to_all_v",
        Opcode::Barrier => "barrier",
        Opcode::Send => "send",
        Opcode::Receive => "receive",
        Opcode::Abort => "abort",
        Opcode::Heartbeat => "heartbeat",
        Opcode::Leave => "leave",
        Opcode::SetTimeout => "set_timeout",
        Opcode::PeerEndpoint => "peer_endpoint",
    }
}

fn element_type_name(element_type: ElementType) -> &'static str {
    match element_type {
        ElementType::None => "none",
        ElementType::U8 => "u8",
        ElementType::U32 => "u32",
        ElementType::I32 => "i32",
        ElementType::F32 => "f32",
        ElementType::F16 => "f16",
        ElementType::BF16 => "bf16",
        ElementType::Bool => "bool",
        ElementType::I8 => "i8",
        ElementType::I16 => "i16",
        ElementType::I64 => "i64",
        ElementType::F64 => "f64",
        ElementType::U16 => "u16",
        ElementType::U64 => "u64",
        ElementType::Complex64 => "complex64",
        ElementType::Complex128 => "complex128",
        ElementType::F8E4M3Fn => "f8_e4m3fn",
        ElementType::F8E5M2 => "f8_e5m2",
        ElementType::F8E4M3Fnuz => "f8_e4m3fnuz",
        ElementType::F8E5M2Fnuz => "f8_e5m2fnuz",
        ElementType::F8E8M0Fnu => "f8_e8m0fnu",
        ElementType::F4E2M1FnX2 => "f4_e2m1fn_x2",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_trace_sink_writes_round_trippable_event() {
        let path = env::temp_dir().join(format!(
            "gx1-collective-trace-test-{}-{}.jsonl",
            std::process::id(),
            NEXT_TRACE_FILE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let event = TransportTraceEvent {
            schema_version: TRANSPORT_TRACE_SCHEMA_VERSION,
            timestamp_unix_ns: 123,
            duration_ns: 456,
            rank: 2,
            world_size: 8,
            transport: CollectiveTransport::GxLink,
            operation: TransportTraceOperation::Send,
            opcode: Some("send".into()),
            element_type: Some("bf16".into()),
            peer_rank: Some(7),
            rail: Some(3),
            tag: 99,
            element_count: 1024,
            input_bytes: 2048,
            output_bytes: 0,
            success: true,
            error: None,
        };
        let sink = JsonlTransportTraceSink::create(&path).unwrap();
        sink.record(&event);
        let encoded = fs::read_to_string(&path).unwrap();
        let decoded = serde_json::from_str::<TransportTraceEvent>(encoded.trim()).unwrap();
        assert_eq!(decoded, event);
        fs::remove_file(path).unwrap();
    }
}
