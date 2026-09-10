use super::*;
use crate::rank::{FrameHeader, Opcode, UniqueId};
use std::cell::Cell;

fn counts_frame(counts: &[usize], data: &[u8], element_type: ElementType) -> Frame {
    let mut header = FrameHeader::collective(
        UniqueId::from_bytes([7; 16]),
        Opcode::AllToAllV,
        element_type,
        0,
        0,
        counts.len() as u32,
        0,
        counts.iter().sum::<usize>() as u64,
    );
    header.flags |= FLAG_COUNTS_PREFIX;
    let mut payload = encode_counts(counts).unwrap();
    payload.extend_from_slice(data);
    Frame::new(header, payload).unwrap()
}

#[test]
fn counts_encoding_and_decode_keep_little_endian_layout_and_borrowed_data() {
    assert_eq!(
        encode_counts(&[1, 256]).unwrap(),
        vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    );
    let frame = counts_frame(&[2, 0, 1], &[1, 0, 2, 0, 3, 0], ElementType::U16);
    let (counts, data) = decode_counts_response(&frame, 3, 2).unwrap();
    assert_eq!(counts, vec![2, 0, 1]);
    assert_eq!(data, &[1, 0, 2, 0, 3, 0]);
    assert_eq!(data.as_ptr(), frame.payload[24..].as_ptr());
    let empty = counts_frame(&[0], &[], ElementType::U8);
    assert!(decode_counts_response(&empty, 1, 1).unwrap().1.is_empty());
}

#[test]
fn counts_validation_keeps_flag_prefix_sum_and_data_errors() {
    let frame = counts_frame(&[1, 1], &[1, 2], ElementType::U8);
    let mut missing_flag = frame.clone();
    missing_flag.header.flags &= !FLAG_COUNTS_PREFIX;
    assert!(
        decode_counts_response(&missing_flag, 2, 1)
            .unwrap_err()
            .to_string()
            .contains("missing counts")
    );
    let mut truncated = frame.clone();
    truncated.payload.truncate(15);
    assert!(
        decode_counts_response(&truncated, 2, 1)
            .unwrap_err()
            .to_string()
            .contains("prefix is truncated")
    );
    let mut wrong_sum = frame.clone();
    wrong_sum.header.element_count = 3;
    assert!(
        decode_counts_response(&wrong_sum, 2, 1)
            .unwrap_err()
            .to_string()
            .contains("counts sum to 2, header declares 3")
    );
    let mut wrong_data = frame;
    wrong_data.payload.pop();
    assert!(
        decode_counts_response(&wrong_data, 2, 1)
            .unwrap_err()
            .to_string()
            .contains("data has 1 bytes, expected 2")
    );
}

#[test]
fn counts_validation_keeps_checked_arithmetic() {
    let mut frame = counts_frame(&[0, 0], &[], ElementType::U8);
    frame.payload[..16].copy_from_slice(&encode_counts(&[usize::MAX, 1]).unwrap());
    assert!(matches!(
        decode_counts_response(&frame, 2, 1),
        Err(RankError::Overflow("TCP all-to-all-v receive count"))
    ));
    assert!(matches!(
        decode_counts_response(&frame, usize::MAX, 1),
        Err(RankError::Overflow("TCP all-to-all-v counts prefix"))
    ));
    let mut frame = counts_frame(&[0], &[], ElementType::U8);
    frame.payload[..8].copy_from_slice(&encode_counts(&[usize::MAX]).unwrap());
    frame.header.element_count = usize::MAX as u64;
    assert!(matches!(
        decode_counts_response(&frame, 1, 2),
        Err(RankError::Overflow("TCP all-to-all-v data"))
    ));
}

#[derive(Debug)]
enum DecodeError {
    Rank(RankError),
    Decoder(u8),
}

impl From<RankError> for DecodeError {
    fn from(error: RankError) -> Self {
        Self::Rank(error)
    }
}

#[test]
fn typed_decoder_runs_once_only_after_length_validation_and_keeps_its_error() {
    let calls = Cell::new(0);
    let decoder = |_: &[u8]| -> Result<Vec<u16>, DecodeError> {
        calls.set(calls.get() + 1);
        Err(DecodeError::Decoder(9))
    };
    assert!(matches!(
        decode_exact(&[], usize::MAX, 2, &decoder),
        Err(DecodeError::Rank(RankError::Overflow(
            "TCP decoded payload bytes"
        )))
    ));
    assert!(matches!(
        decode_exact(&[1], 1, 2, &decoder),
        Err(DecodeError::Rank(RankError::Network(_)))
    ));
    assert_eq!(calls.get(), 0);
    assert!(matches!(
        decode_exact(&[1, 0], 1, 2, &decoder),
        Err(DecodeError::Decoder(9))
    ));
    assert_eq!(calls.get(), 1);
    let values = decode_exact::<_, RankError>(&[0, 128, 255, 127], 2, 2, |bytes| {
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect::<Vec<_>>())
    })
    .unwrap();
    assert_eq!(values, vec![0x8000, 0x7fff]);
}

#[test]
fn host_reduction_payloads_keep_dtype_sizes_and_validation() {
    assert_eq!(
        validate_host_reduction_payload(ElementType::BF16, 3, &[0; 6]).unwrap(),
        6
    );
    assert_eq!(
        validate_host_reduction_payload(ElementType::Complex128, 2, &[0; 32]).unwrap(),
        32
    );
    assert_eq!(
        validate_host_reduction_payload(ElementType::F4E2M1FnX2, 3, &[0; 3]).unwrap(),
        3
    );
    assert_eq!(
        validate_host_reduction_payload(ElementType::F32, 0, &[]).unwrap(),
        0
    );
    assert!(matches!(
        validate_host_reduction_payload(ElementType::None, 0, &[]),
        Err(RankError::InvalidLength(_))
    ));
    assert!(matches!(
        validate_host_reduction_payload(ElementType::F32, usize::MAX, &[]),
        Err(RankError::Overflow(_))
    ));
    assert!(matches!(
        validate_host_reduction_payload(ElementType::F32, 1, &[0; 3]),
        Err(RankError::Network(_))
    ));
    validate_host_reduction_response("reduce", &[], 0).unwrap();
    assert!(
        validate_host_reduction_response("reduce", &[0], 0)
            .unwrap_err()
            .to_string()
            .contains("host reduce response has 1 bytes, expected 0")
    );
}
