//! Versioned wire contract shared by host-staged, TCP, peer-memory, and RDMA
//! collective transports.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub const PROTOCOL_MAGIC: [u8; 4] = *b"GXCL";
pub const PROTOCOL_VERSION: u16 = 3;
pub const FRAME_HEADER_BYTES: usize = 80;
pub const MAX_FRAME_PAYLOAD_BYTES: usize = 1 << 30;
pub const ANY_RANK: u32 = u32::MAX;
pub const FLAG_PAYLOAD: u16 = 1;
pub const FLAG_COUNTS_PREFIX: u16 = 1 << 1;
pub const FLAG_P2P_CHANNEL: u16 = 1 << 2;

static NEXT_UNIQUE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UniqueId([u8; 16]);

impl UniqueId {
    pub fn new() -> Self {
        let counter = NEXT_UNIQUE_ID.fetch_add(1, Ordering::Relaxed);
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let process = u128::from(std::process::id());
        let mixed = time ^ (u128::from(counter) << 64) ^ (process << 32);
        Self(mixed.to_le_bytes())
    }

    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl Default for UniqueId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Opcode {
    Join = 1,
    Ready = 2,
    Broadcast = 3,
    AllGather = 4,
    Gather = 5,
    Scatter = 6,
    Reduce = 7,
    AllReduce = 8,
    ReduceScatter = 9,
    AllToAll = 10,
    AllToAllV = 11,
    Barrier = 12,
    Send = 13,
    Receive = 14,
    Abort = 15,
    Heartbeat = 16,
    Leave = 17,
    SetTimeout = 18,
    PeerEndpoint = 19,
}

impl TryFrom<u8> for Opcode {
    type Error = ProtocolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Join),
            2 => Ok(Self::Ready),
            3 => Ok(Self::Broadcast),
            4 => Ok(Self::AllGather),
            5 => Ok(Self::Gather),
            6 => Ok(Self::Scatter),
            7 => Ok(Self::Reduce),
            8 => Ok(Self::AllReduce),
            9 => Ok(Self::ReduceScatter),
            10 => Ok(Self::AllToAll),
            11 => Ok(Self::AllToAllV),
            12 => Ok(Self::Barrier),
            13 => Ok(Self::Send),
            14 => Ok(Self::Receive),
            15 => Ok(Self::Abort),
            16 => Ok(Self::Heartbeat),
            17 => Ok(Self::Leave),
            18 => Ok(Self::SetTimeout),
            19 => Ok(Self::PeerEndpoint),
            _ => Err(ProtocolError::UnknownOpcode(value)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ElementType {
    None = 0,
    U8 = 1,
    U32 = 2,
    I32 = 3,
    F32 = 4,
    F16 = 5,
    BF16 = 6,
    Bool = 7,
    I8 = 8,
    I16 = 9,
    I64 = 10,
    F64 = 11,
    U16 = 12,
    U64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    F8E4M3Fn = 16,
    F8E5M2 = 17,
    F8E4M3Fnuz = 18,
    F8E5M2Fnuz = 19,
    F8E8M0Fnu = 20,
    F4E2M1FnX2 = 21,
}

impl ElementType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::None => 0,
            Self::U8
            | Self::Bool
            | Self::I8
            | Self::F8E4M3Fn
            | Self::F8E5M2
            | Self::F8E4M3Fnuz
            | Self::F8E5M2Fnuz
            | Self::F8E8M0Fnu
            | Self::F4E2M1FnX2 => 1,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 | Self::U64 | Self::Complex64 => 8,
            Self::Complex128 => 16,
        }
    }

    pub const fn is_low_precision_storage(self) -> bool {
        matches!(
            self,
            Self::F8E4M3Fn
                | Self::F8E5M2
                | Self::F8E4M3Fnuz
                | Self::F8E5M2Fnuz
                | Self::F8E8M0Fnu
                | Self::F4E2M1FnX2
        )
    }
}

impl TryFrom<u8> for ElementType {
    type Error = ProtocolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::U8),
            2 => Ok(Self::U32),
            3 => Ok(Self::I32),
            4 => Ok(Self::F32),
            5 => Ok(Self::F16),
            6 => Ok(Self::BF16),
            7 => Ok(Self::Bool),
            8 => Ok(Self::I8),
            9 => Ok(Self::I16),
            10 => Ok(Self::I64),
            11 => Ok(Self::F64),
            12 => Ok(Self::U16),
            13 => Ok(Self::U64),
            14 => Ok(Self::Complex64),
            15 => Ok(Self::Complex128),
            16 => Ok(Self::F8E4M3Fn),
            17 => Ok(Self::F8E5M2),
            18 => Ok(Self::F8E4M3Fnuz),
            19 => Ok(Self::F8E5M2Fnuz),
            20 => Ok(Self::F8E8M0Fnu),
            21 => Ok(Self::F4E2M1FnX2),
            _ => Err(ProtocolError::UnknownElementType(value)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub opcode: Opcode,
    pub element_type: ElementType,
    pub flags: u16,
    pub source_rank: u32,
    pub destination_rank: u32,
    pub root_rank: u32,
    pub world_size: u32,
    pub sequence: u64,
    pub tag: u64,
    pub element_count: u64,
    pub payload_bytes: u64,
    pub unique_id: UniqueId,
    pub payload_checksum: u32,
}

impl FrameHeader {
    #[allow(clippy::too_many_arguments)]
    pub fn collective(
        unique_id: UniqueId,
        opcode: Opcode,
        element_type: ElementType,
        source_rank: u32,
        root_rank: u32,
        world_size: u32,
        sequence: u64,
        element_count: u64,
    ) -> Self {
        Self {
            opcode,
            element_type,
            flags: 0,
            source_rank,
            destination_rank: ANY_RANK,
            root_rank,
            world_size,
            sequence,
            tag: 0,
            element_count,
            payload_bytes: 0,
            unique_id,
            payload_checksum: checksum(&[]),
        }
    }

    pub fn with_payload(mut self, payload: &[u8]) -> Result<Self, ProtocolError> {
        if payload.len() > MAX_FRAME_PAYLOAD_BYTES {
            return Err(ProtocolError::PayloadTooLarge(payload.len()));
        }
        self.flags |= FLAG_PAYLOAD;
        self.payload_bytes = payload.len() as u64;
        self.payload_checksum = checksum(payload);
        // The checksum above was computed from this exact slice. Validate the
        // resulting wire contract without scanning the payload a second time.
        self.validate(None)?;
        Ok(self)
    }

    pub fn encode(&self) -> Result<[u8; FRAME_HEADER_BYTES], ProtocolError> {
        self.validate(None)?;
        let mut bytes = [0_u8; FRAME_HEADER_BYTES];
        bytes[0..4].copy_from_slice(&PROTOCOL_MAGIC);
        put_u16(&mut bytes, 4, PROTOCOL_VERSION);
        put_u16(&mut bytes, 6, FRAME_HEADER_BYTES as u16);
        bytes[8] = self.opcode as u8;
        bytes[9] = self.element_type as u8;
        put_u16(&mut bytes, 10, self.flags);
        put_u32(&mut bytes, 12, self.source_rank);
        put_u32(&mut bytes, 16, self.destination_rank);
        put_u32(&mut bytes, 20, self.root_rank);
        put_u32(&mut bytes, 24, self.world_size);
        put_u64(&mut bytes, 28, self.sequence);
        put_u64(&mut bytes, 36, self.tag);
        put_u64(&mut bytes, 44, self.element_count);
        put_u64(&mut bytes, 52, self.payload_bytes);
        bytes[60..76].copy_from_slice(self.unique_id.as_bytes());
        put_u32(&mut bytes, 76, self.payload_checksum);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ProtocolError> {
        if bytes.len() < FRAME_HEADER_BYTES {
            return Err(ProtocolError::TruncatedHeader(bytes.len()));
        }
        if bytes[0..4] != PROTOCOL_MAGIC {
            return Err(ProtocolError::BadMagic(bytes[0..4].try_into().unwrap()));
        }
        let version = get_u16(bytes, 4);
        if version != PROTOCOL_VERSION {
            return Err(ProtocolError::UnsupportedVersion(version));
        }
        let header_bytes = get_u16(bytes, 6) as usize;
        if header_bytes != FRAME_HEADER_BYTES {
            return Err(ProtocolError::UnsupportedHeaderLength(header_bytes));
        }
        let header = Self {
            opcode: Opcode::try_from(bytes[8])?,
            element_type: ElementType::try_from(bytes[9])?,
            flags: get_u16(bytes, 10),
            source_rank: get_u32(bytes, 12),
            destination_rank: get_u32(bytes, 16),
            root_rank: get_u32(bytes, 20),
            world_size: get_u32(bytes, 24),
            sequence: get_u64(bytes, 28),
            tag: get_u64(bytes, 36),
            element_count: get_u64(bytes, 44),
            payload_bytes: get_u64(bytes, 52),
            unique_id: UniqueId::from_bytes(bytes[60..76].try_into().unwrap()),
            payload_checksum: get_u32(bytes, 76),
        };
        header.validate(None)?;
        Ok(header)
    }

    pub fn validate(&self, payload: Option<&[u8]>) -> Result<(), ProtocolError> {
        let unknown_flags = self.flags & !(FLAG_PAYLOAD | FLAG_COUNTS_PREFIX | FLAG_P2P_CHANNEL);
        if unknown_flags != 0 {
            return Err(ProtocolError::UnknownFlags(unknown_flags));
        }
        let has_counts_prefix = self.flags & FLAG_COUNTS_PREFIX != 0;
        if has_counts_prefix && self.opcode != Opcode::AllToAllV {
            return Err(ProtocolError::UnexpectedCountsPrefix(self.opcode));
        }
        if self.flags & FLAG_P2P_CHANNEL != 0
            && !matches!(self.opcode, Opcode::Join | Opcode::Ready)
        {
            return Err(ProtocolError::UnexpectedP2pChannelFlag(self.opcode));
        }
        if self.world_size == 0 {
            return Err(ProtocolError::EmptyWorld);
        }
        validate_rank("source", self.source_rank, self.world_size, false)?;
        validate_rank("destination", self.destination_rank, self.world_size, true)?;
        validate_rank("root", self.root_rank, self.world_size, true)?;
        let payload_bytes = usize::try_from(self.payload_bytes)
            .map_err(|_| ProtocolError::PayloadTooLarge(usize::MAX))?;
        if payload_bytes > MAX_FRAME_PAYLOAD_BYTES {
            return Err(ProtocolError::PayloadTooLarge(payload_bytes));
        }
        let has_payload = self.flags & FLAG_PAYLOAD != 0;
        if has_payload != (self.payload_bytes != 0) {
            return Err(ProtocolError::PayloadFlagMismatch);
        }
        if self.element_type == ElementType::None {
            if self.element_count != 0 || self.payload_bytes != 0 {
                return Err(ProtocolError::ElementLengthMismatch);
            }
        } else if has_payload {
            let element_bytes = self
                .element_count
                .checked_mul(self.element_type.byte_width() as u64)
                .ok_or(ProtocolError::ElementLengthMismatch)?;
            let prefix_bytes = if has_counts_prefix {
                u64::from(self.world_size)
                    .checked_mul(8)
                    .ok_or(ProtocolError::ElementLengthMismatch)?
            } else {
                0
            };
            let expected = element_bytes
                .checked_add(prefix_bytes)
                .ok_or(ProtocolError::ElementLengthMismatch)?;
            if expected != self.payload_bytes {
                return Err(ProtocolError::ElementLengthMismatch);
            }
        }
        if let Some(payload) = payload {
            if payload.len() != payload_bytes {
                return Err(ProtocolError::PayloadLengthMismatch {
                    declared: payload_bytes,
                    actual: payload.len(),
                });
            }
            let actual = checksum(payload);
            if actual != self.payload_checksum {
                return Err(ProtocolError::ChecksumMismatch {
                    declared: self.payload_checksum,
                    actual,
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub header: FrameHeader,
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn new(header: FrameHeader, payload: Vec<u8>) -> Result<Self, ProtocolError> {
        let header = if payload.is_empty() {
            header.validate(Some(&payload))?;
            header
        } else {
            header.with_payload(&payload)?
        };
        Ok(Self { header, payload })
    }

    pub fn encode(&self) -> Result<Vec<u8>, ProtocolError> {
        self.header.validate(Some(&self.payload))?;
        let mut bytes = Vec::with_capacity(FRAME_HEADER_BYTES + self.payload.len());
        bytes.extend_from_slice(&self.header.encode()?);
        bytes.extend_from_slice(&self.payload);
        Ok(bytes)
    }

    pub(crate) fn encode_transport_header(
        &self,
    ) -> Result<[u8; FRAME_HEADER_BYTES], ProtocolError> {
        let declared = usize::try_from(self.header.payload_bytes)
            .map_err(|_| ProtocolError::PayloadTooLarge(usize::MAX))?;
        if self.payload.len() != declared {
            return Err(ProtocolError::PayloadLengthMismatch {
                declared,
                actual: self.payload.len(),
            });
        }
        self.header.encode()
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ProtocolError> {
        let header = FrameHeader::decode(bytes)?;
        let payload_bytes = usize::try_from(header.payload_bytes)
            .map_err(|_| ProtocolError::PayloadTooLarge(usize::MAX))?;
        let expected = FRAME_HEADER_BYTES
            .checked_add(payload_bytes)
            .ok_or(ProtocolError::PayloadTooLarge(payload_bytes))?;
        if bytes.len() != expected {
            return Err(ProtocolError::FrameLengthMismatch {
                declared: expected,
                actual: bytes.len(),
            });
        }
        let payload = bytes[FRAME_HEADER_BYTES..].to_vec();
        header.validate(Some(&payload))?;
        Ok(Self { header, payload })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationDescriptor {
    pub opcode: Opcode,
    pub element_type: ElementType,
    pub root_rank: u32,
    pub element_count: u64,
    pub layout_hash: u64,
}

impl OperationDescriptor {
    pub const fn new(
        opcode: Opcode,
        element_type: ElementType,
        root_rank: u32,
        element_count: u64,
    ) -> Self {
        Self {
            opcode,
            element_type,
            root_rank,
            element_count,
            layout_hash: 0,
        }
    }

    pub const fn with_layout_hash(mut self, layout_hash: u64) -> Self {
        self.layout_hash = layout_hash;
        self
    }
}

/// Coordinator-side sequence agreement. An operation becomes ready only when
/// every rank submitted exactly the same descriptor for the same sequence.
#[derive(Debug)]
pub struct CollectiveAgreement {
    world_size: usize,
    next_sequence: u64,
    pending: BTreeMap<u64, Vec<Option<OperationDescriptor>>>,
}

impl CollectiveAgreement {
    pub fn new(world_size: usize) -> Result<Self, ProtocolError> {
        if world_size == 0 {
            return Err(ProtocolError::EmptyWorld);
        }
        Ok(Self {
            world_size,
            next_sequence: 0,
            pending: BTreeMap::new(),
        })
    }

    pub const fn next_sequence(&self) -> u64 {
        self.next_sequence
    }

    pub fn submit(
        &mut self,
        rank: usize,
        sequence: u64,
        descriptor: OperationDescriptor,
    ) -> Result<bool, ProtocolError> {
        if rank >= self.world_size {
            return Err(ProtocolError::RankOutOfRange {
                name: "source",
                rank: rank as u32,
                world_size: self.world_size as u32,
            });
        }
        if sequence != self.next_sequence {
            return Err(ProtocolError::SequenceMismatch {
                expected: self.next_sequence,
                actual: sequence,
            });
        }
        let ranks = self
            .pending
            .entry(sequence)
            .or_insert_with(|| vec![None; self.world_size]);
        if ranks[rank].is_some() {
            return Err(ProtocolError::DuplicateSubmission { rank, sequence });
        }
        if let Some(expected) = ranks.iter().flatten().next()
            && expected != &descriptor
        {
            return Err(ProtocolError::CollectiveMismatch {
                sequence,
                expected: expected.clone(),
                actual: descriptor,
            });
        }
        ranks[rank] = Some(descriptor);
        let ready = ranks.iter().all(Option::is_some);
        if ready {
            self.pending.remove(&sequence);
            self.next_sequence = self
                .next_sequence
                .checked_add(1)
                .ok_or(ProtocolError::SequenceOverflow)?;
        }
        Ok(ready)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtocolError {
    TruncatedHeader(usize),
    BadMagic([u8; 4]),
    UnsupportedVersion(u16),
    UnsupportedHeaderLength(usize),
    UnknownOpcode(u8),
    UnknownElementType(u8),
    UnknownFlags(u16),
    UnexpectedCountsPrefix(Opcode),
    UnexpectedP2pChannelFlag(Opcode),
    EmptyWorld,
    RankOutOfRange {
        name: &'static str,
        rank: u32,
        world_size: u32,
    },
    PayloadTooLarge(usize),
    PayloadFlagMismatch,
    ElementLengthMismatch,
    PayloadLengthMismatch {
        declared: usize,
        actual: usize,
    },
    FrameLengthMismatch {
        declared: usize,
        actual: usize,
    },
    ChecksumMismatch {
        declared: u32,
        actual: u32,
    },
    SequenceMismatch {
        expected: u64,
        actual: u64,
    },
    DuplicateSubmission {
        rank: usize,
        sequence: u64,
    },
    CollectiveMismatch {
        sequence: u64,
        expected: OperationDescriptor,
        actual: OperationDescriptor,
    },
    SequenceOverflow,
}

impl Display for ProtocolError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TruncatedHeader(actual) => write!(
                formatter,
                "GX collective header needs {FRAME_HEADER_BYTES} bytes, got {actual}"
            ),
            Self::BadMagic(actual) => write!(formatter, "bad GX collective magic {actual:?}"),
            Self::UnsupportedVersion(version) => {
                write!(
                    formatter,
                    "unsupported GX collective protocol version {version}"
                )
            }
            Self::UnsupportedHeaderLength(length) => {
                write!(
                    formatter,
                    "unsupported GX collective header length {length}"
                )
            }
            Self::UnknownOpcode(opcode) => write!(formatter, "unknown collective opcode {opcode}"),
            Self::UnknownElementType(element_type) => {
                write!(formatter, "unknown collective element type {element_type}")
            }
            Self::UnknownFlags(flags) => {
                write!(formatter, "unknown collective frame flags {flags:#06x}")
            }
            Self::UnexpectedCountsPrefix(opcode) => {
                write!(formatter, "counts prefix is invalid for {opcode:?}")
            }
            Self::UnexpectedP2pChannelFlag(opcode) => {
                write!(
                    formatter,
                    "point-to-point channel flag is invalid for {opcode:?}"
                )
            }
            Self::EmptyWorld => write!(formatter, "collective protocol world cannot be empty"),
            Self::RankOutOfRange {
                name,
                rank,
                world_size,
            } => write!(
                formatter,
                "{name} rank {rank} is outside protocol world size {world_size}"
            ),
            Self::PayloadTooLarge(bytes) => {
                write!(
                    formatter,
                    "collective payload of {bytes} bytes is too large"
                )
            }
            Self::PayloadFlagMismatch => {
                write!(formatter, "collective payload flag and length disagree")
            }
            Self::ElementLengthMismatch => {
                write!(
                    formatter,
                    "collective element count and payload length disagree"
                )
            }
            Self::PayloadLengthMismatch { declared, actual } => write!(
                formatter,
                "collective payload declares {declared} bytes, got {actual}"
            ),
            Self::FrameLengthMismatch { declared, actual } => write!(
                formatter,
                "collective frame declares {declared} bytes, got {actual}"
            ),
            Self::ChecksumMismatch { declared, actual } => write!(
                formatter,
                "collective payload checksum {actual:#010x} does not match {declared:#010x}"
            ),
            Self::SequenceMismatch { expected, actual } => write!(
                formatter,
                "collective sequence {actual} does not match expected {expected}"
            ),
            Self::DuplicateSubmission { rank, sequence } => write!(
                formatter,
                "rank {rank} submitted collective sequence {sequence} twice"
            ),
            Self::CollectiveMismatch {
                sequence,
                expected,
                actual,
            } => write!(
                formatter,
                "collective sequence {sequence} mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::SequenceOverflow => write!(formatter, "collective sequence overflow"),
        }
    }
}

impl Error for ProtocolError {}

fn validate_rank(
    name: &'static str,
    rank: u32,
    world_size: u32,
    allow_any: bool,
) -> Result<(), ProtocolError> {
    if (allow_any && rank == ANY_RANK) || rank < world_size {
        Ok(())
    } else {
        Err(ProtocolError::RankOutOfRange {
            name,
            rank,
            world_size,
        })
    }
}

fn checksum(bytes: &[u8]) -> u32 {
    crc32fast::hash(bytes)
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn get_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn get_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip_is_little_endian_and_checksum_protected() {
        let unique_id = UniqueId::from_bytes([7; 16]);
        let mut header = FrameHeader::collective(
            unique_id,
            Opcode::Send,
            ElementType::U32,
            1,
            ANY_RANK,
            4,
            9,
            3,
        );
        header.destination_rank = 2;
        header.tag = 44;
        let frame = Frame::new(header, vec![1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]).unwrap();
        let encoded = frame.encode().unwrap();
        assert_eq!(&encoded[0..4], b"GXCL");
        assert_eq!(Frame::decode(&encoded).unwrap(), frame);

        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 0x80;
        assert!(matches!(
            Frame::decode(&corrupt),
            Err(ProtocolError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn protocol_v3_uses_crc32_and_transport_reuses_the_precomputed_checksum() {
        assert_eq!(PROTOCOL_VERSION, 3);
        assert_eq!(checksum(b"123456789"), 0xcbf4_3926);

        let header = FrameHeader::collective(
            UniqueId::from_bytes([3; 16]),
            Opcode::Send,
            ElementType::U8,
            0,
            ANY_RANK,
            2,
            1,
            4,
        );
        let mut frame = Frame::new(header, vec![1, 2, 3, 4]).unwrap();
        frame.payload[0] ^= 0xff;

        let encoded_header = frame.encode_transport_header().unwrap();
        let mut wire = encoded_header.to_vec();
        wire.extend_from_slice(&frame.payload);
        assert!(matches!(
            Frame::decode(&wire),
            Err(ProtocolError::ChecksumMismatch { .. })
        ));

        frame.payload.push(5);
        assert!(matches!(
            frame.encode_transport_header(),
            Err(ProtocolError::PayloadLengthMismatch {
                declared: 4,
                actual: 5
            })
        ));
    }

    #[test]
    fn extended_pytorch_element_types_round_trip_with_exact_widths() {
        for (code, element_type, width) in [
            (7, ElementType::Bool, 1),
            (8, ElementType::I8, 1),
            (9, ElementType::I16, 2),
            (10, ElementType::I64, 8),
            (11, ElementType::F64, 8),
            (12, ElementType::U16, 2),
            (13, ElementType::U64, 8),
            (14, ElementType::Complex64, 8),
            (15, ElementType::Complex128, 16),
            (16, ElementType::F8E4M3Fn, 1),
            (17, ElementType::F8E5M2, 1),
            (18, ElementType::F8E4M3Fnuz, 1),
            (19, ElementType::F8E5M2Fnuz, 1),
            (20, ElementType::F8E8M0Fnu, 1),
            (21, ElementType::F4E2M1FnX2, 1),
        ] {
            assert_eq!(ElementType::try_from(code).unwrap(), element_type);
            assert_eq!(element_type.byte_width(), width);
            assert_eq!(element_type.is_low_precision_storage(), code >= 16);
        }
    }

    #[test]
    fn decoder_rejects_version_rank_and_length_mismatches() {
        let header = FrameHeader::collective(
            UniqueId::from_bytes([1; 16]),
            Opcode::AllReduce,
            ElementType::F32,
            0,
            ANY_RANK,
            2,
            0,
            8,
        );
        let mut encoded = Frame::new(header, vec![0; 32]).unwrap().encode().unwrap();
        encoded[4..6].copy_from_slice(&99_u16.to_le_bytes());
        assert!(matches!(
            Frame::decode(&encoded),
            Err(ProtocolError::UnsupportedVersion(99))
        ));

        let mut bad_rank = FrameHeader::collective(
            UniqueId::from_bytes([1; 16]),
            Opcode::Barrier,
            ElementType::None,
            2,
            ANY_RANK,
            2,
            0,
            0,
        );
        assert!(matches!(
            bad_rank.encode(),
            Err(ProtocolError::RankOutOfRange { .. })
        ));
        bad_rank.source_rank = 0;
        bad_rank.flags = FLAG_PAYLOAD;
        assert!(matches!(
            bad_rank.encode(),
            Err(ProtocolError::PayloadFlagMismatch)
        ));
    }

    #[test]
    fn agreement_detects_order_and_contract_mismatch() {
        let mut agreement = CollectiveAgreement::new(3).unwrap();
        let operation =
            OperationDescriptor::new(Opcode::AllReduce, ElementType::BF16, ANY_RANK, 1024);
        assert!(!agreement.submit(2, 0, operation.clone()).unwrap());
        assert!(!agreement.submit(0, 0, operation.clone()).unwrap());
        assert!(matches!(
            agreement.submit(
                1,
                0,
                OperationDescriptor::new(Opcode::AllGather, ElementType::BF16, ANY_RANK, 1024)
            ),
            Err(ProtocolError::CollectiveMismatch { .. })
        ));
        assert!(agreement.submit(1, 0, operation.clone()).unwrap());
        assert_eq!(agreement.next_sequence(), 1);
        assert!(matches!(
            agreement.submit(0, 0, operation),
            Err(ProtocolError::SequenceMismatch { .. })
        ));
    }
}
