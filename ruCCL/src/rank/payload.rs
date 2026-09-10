use super::error::RankError;
use super::{ElementType, FLAG_COUNTS_PREFIX, Frame, NetworkError};

#[cfg(test)]
mod tests;

pub fn validate_host_reduction_payload(
    element_type: ElementType,
    element_count: usize,
    payload: &[u8],
) -> Result<usize, RankError> {
    let element_bytes = element_type.byte_width();
    if element_bytes == 0 {
        return Err(RankError::InvalidLength(
            "host reduction requires a tensor element type",
        ));
    }
    let expected = element_count
        .checked_mul(element_bytes)
        .ok_or(RankError::Overflow("host reduction payload bytes"))?;
    if payload.len() != expected {
        return Err(NetworkError::InvalidConfiguration(format!(
            "host reduction payload has {} bytes, expected {expected}",
            payload.len()
        ))
        .into());
    }
    Ok(expected)
}

pub fn validate_host_reduction_response(
    operation: &str,
    payload: &[u8],
    expected: usize,
) -> Result<(), RankError> {
    if payload.len() != expected {
        return Err(NetworkError::InvalidConfiguration(format!(
            "host {operation} response has {} bytes, expected {expected}",
            payload.len()
        ))
        .into());
    }
    Ok(())
}

pub fn decode_exact<T, E: From<RankError>>(
    payload: &[u8],
    expected_elements: usize,
    element_bytes: usize,
    decode: impl FnOnce(&[u8]) -> Result<Vec<T>, E>,
) -> Result<Vec<T>, E> {
    let expected_bytes = expected_elements
        .checked_mul(element_bytes)
        .ok_or(RankError::Overflow("TCP decoded payload bytes"))?;
    if payload.len() != expected_bytes {
        return Err(
            RankError::Network(NetworkError::InvalidConfiguration(format!(
                "TCP response has {} bytes, expected {expected_bytes}",
                payload.len()
            )))
            .into(),
        );
    }
    decode(payload)
}

pub fn encode_counts(counts: &[usize]) -> Result<Vec<u8>, RankError> {
    let mut encoded = Vec::with_capacity(counts.len().saturating_mul(8));
    for count in counts {
        encoded.extend_from_slice(
            &u64::try_from(*count)
                .map_err(|_| RankError::Overflow("TCP all-to-all-v count"))?
                .to_le_bytes(),
        );
    }
    Ok(encoded)
}

pub fn decode_counts_response(
    response: &Frame,
    world_size: usize,
    element_bytes: usize,
) -> Result<(Vec<usize>, &[u8]), RankError> {
    if response.header.flags & FLAG_COUNTS_PREFIX == 0 {
        return Err(NetworkError::InvalidConfiguration(
            "TCP all-to-all-v response is missing counts".into(),
        )
        .into());
    }
    let prefix_bytes = world_size
        .checked_mul(8)
        .ok_or(RankError::Overflow("TCP all-to-all-v counts prefix"))?;
    if response.payload.len() < prefix_bytes {
        return Err(NetworkError::InvalidConfiguration(
            "TCP all-to-all-v counts prefix is truncated".into(),
        )
        .into());
    }
    let counts = response.payload[..prefix_bytes]
        .chunks_exact(8)
        .map(|bytes| {
            usize::try_from(u64::from_le_bytes(bytes.try_into().unwrap()))
                .map_err(|_| RankError::Overflow("TCP all-to-all-v receive count"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let total = counts
        .iter()
        .try_fold(0_usize, |sum, count| sum.checked_add(*count))
        .ok_or(RankError::Overflow("TCP all-to-all-v receive count"))?;
    if total as u64 != response.header.element_count {
        return Err(NetworkError::InvalidConfiguration(format!(
            "TCP all-to-all-v counts sum to {total}, header declares {}",
            response.header.element_count
        ))
        .into());
    }
    let data = &response.payload[prefix_bytes..];
    let expected = total
        .checked_mul(element_bytes)
        .ok_or(RankError::Overflow("TCP all-to-all-v data"))?;
    if data.len() != expected {
        return Err(NetworkError::InvalidConfiguration(format!(
            "TCP all-to-all-v data has {} bytes, expected {expected}",
            data.len()
        ))
        .into());
    }
    Ok((counts, data))
}
