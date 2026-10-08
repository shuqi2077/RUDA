use super::{Primitive, TensorDevice, TensorDeviceError};
use super::bounded_sharded::{native_reshape, native_element_type};
use crate::rank::{ElementType, communicator::RankCommunicator, device_collective::NativeChunkPlan};
use ruda_tensor::{Backend, Shape, collective::{VariableTensorCollective, VariableTensorExchange}};
use serde::{Deserialize, Serialize};

/// Actual successful variable-row collective work, not a timer-based estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowExchangeProgress {
    /// Completed collective rounds, including zero-payload participation when peers still exchange rows.
    pub completed_rounds: usize,
    /// Exact rounds required by the gathered original source/destination counts.
    pub total_rounds: usize,
    /// Actual source rows sent by this rank so far.
    pub sent_rows: usize,
    /// Actual original total source rows for this rank.
    pub total_send_rows: usize,
    /// Actual received rows written on this rank's owned device so far.
    pub received_rows: usize,
    /// Exact received rows from the original source-count matrix.
    pub total_receive_rows: usize,
}

impl<B: Backend> RankCommunicator<TensorDevice<B>> {
    /// Exchange variable floating row blocks using explicit bounded native tensor messages.
    /// Original trailing axes/dtype and source-rank receive order are retained.
    /// No expert ownership, row sorting or capacity truncation is inferred.
    pub fn all_to_all_v_float_chunked(
        &self, value: B::FloatTensorPrimitive, send_counts: &[usize], max_chunk_bytes: usize,
    ) -> Result<VariableTensorExchange<B::FloatTensorPrimitive>, TensorDeviceError> {
        self.all_to_all_v_float_chunked_with_progress(value, send_counts, max_chunk_bytes, |_| {})
    }

    /// Bounded floating row exchange reporting actual device-completed rows and collective rounds.
    pub fn all_to_all_v_float_chunked_with_progress<F: FnMut(RowExchangeProgress)>(
        &self, value: B::FloatTensorPrimitive, send_counts: &[usize], max_chunk_bytes: usize, mut progress: F,
    ) -> Result<VariableTensorExchange<B::FloatTensorPrimitive>, TensorDeviceError> {
        let (value, receive_counts) = self.native_rows_chunked(Primitive::Float(value), send_counts, max_chunk_bytes, &mut progress)?;
        match value { Primitive::Float(value) => Ok(VariableTensorExchange { value, receive_counts }),
            Primitive::Int(_) => unreachable!("original floating variable-row kind") }
    }

    /// Exchange original U8/U32/I32/I64 row blocks without widening packed storage to floating point.
    /// Complete original rows, including zero-row destinations, retain their source-rank ordering.
    pub fn all_to_all_v_int_chunked(
        &self, value: B::IntTensorPrimitive, send_counts: &[usize], max_chunk_bytes: usize,
    ) -> Result<VariableTensorExchange<B::IntTensorPrimitive>, TensorDeviceError> {
        self.all_to_all_v_int_chunked_with_progress(value, send_counts, max_chunk_bytes, |_| {})
    }

    /// Bounded integer row exchange with actual transmitted/received work counters.
    pub fn all_to_all_v_int_chunked_with_progress<F: FnMut(RowExchangeProgress)>(
        &self, value: B::IntTensorPrimitive, send_counts: &[usize], max_chunk_bytes: usize, mut progress: F,
    ) -> Result<VariableTensorExchange<B::IntTensorPrimitive>, TensorDeviceError> {
        let (value, receive_counts) = self.native_rows_chunked(Primitive::Int(value), send_counts, max_chunk_bytes, &mut progress)?;
        match value { Primitive::Int(value) => Ok(VariableTensorExchange { value, receive_counts }),
            Primitive::Float(_) => unreachable!("original integer variable-row kind") }
    }

    fn native_rows_chunked<F: FnMut(RowExchangeProgress)>(
        &self, value: Primitive<B>, send_counts: &[usize], max_chunk_bytes: usize, progress: &mut F,
    ) -> Result<(Primitive<B>, Vec<usize>), TensorDeviceError> {
        let (shape, dtype) = self.native_sharded_metadata(&value)?;
        let world = self.world_size() as usize;
        if world == 0 { return Err(TensorDeviceError::InvalidOperation("variable-row world must be positive")); }
        let rows = shape.first().copied();
        let count_sum = send_counts.iter().try_fold(0usize, |total, count| total.checked_add(*count));
        let valid = !shape.is_empty() && send_counts.len() == world && count_sum == rows;
        let status = [u64::from(valid), send_counts.len() as u64];
        let (statuses, _) = self.host().all_gather_host_staged(ElementType::U64, status.len(),
            status.into_iter().flat_map(u64::to_le_bytes).collect())?;
        if statuses.chunks_exact(16).any(|state| u64::from_le_bytes(state[..8].try_into().unwrap()) == 0) {
            return Err(TensorDeviceError::InvalidBuffer("at least one rank's row counts differ from its actual input/leading axis"));
        }
        let rows = rows.expect("collectively validated row axis");
        let element_type = native_element_type(dtype)?;
        let tail = Shape::new(shape[1..].to_vec());
        self.native_chunk_agreement(&tail, element_type, 0x300, max_chunk_bytes)?;
        let row_elements = tail.iter().try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .filter(|count| *count != 0).ok_or(TensorDeviceError::InvalidBuffer("variable-row trailing axes are zero or overflowed"))?;
        let counts_payload = send_counts.iter().flat_map(|count| (*count as u64).to_le_bytes()).collect::<Vec<_>>();
        let (matrix, _) = self.host().all_gather_host_staged(ElementType::U64, world, counts_payload)?;
        let matrix = matrix.chunks_exact(8).map(|bytes| usize::try_from(u64::from_le_bytes(bytes.try_into().unwrap()))
            .map_err(|_| TensorDeviceError::InvalidBuffer("variable-row wire count exceeds native index range")))
            .collect::<Result<Vec<_>, _>>()?;
        let receive_counts = (0..world).map(|source| matrix[source * world + self.rank() as usize]).collect::<Vec<_>>();
        let send_prefix = row_prefix(send_counts)?;
        let receive_prefix = row_prefix(&receive_counts)?;
        let received_rows = *receive_prefix.last().expect("rank prefix has its zero origin");
        let input_elements = rows.checked_mul(row_elements).ok_or(TensorDeviceError::InvalidBuffer("variable-row source size overflow"))?;
        let output_elements = received_rows.checked_mul(row_elements).ok_or(TensorDeviceError::InvalidBuffer("variable-row output size overflow"))?;
        let maximum_peer_rows = matrix.iter().copied().max().unwrap_or(0);
        let row_bytes = row_elements.checked_mul(element_type.byte_width()).ok_or(TensorDeviceError::InvalidBuffer("native row byte count overflow"))?;
        let peer_rows = if maximum_peer_rows == 0 { 1 } else { max_chunk_bytes / world / row_bytes };
        if peer_rows == 0 { return Err(TensorDeviceError::InvalidBuffer("variable-row budget must fit one native row per participant")); }
        NativeChunkPlan::new(input_elements, element_type.byte_width(), element_type.byte_width())?;
        NativeChunkPlan::new(output_elements, element_type.byte_width(), element_type.byte_width())?;
        let value = native_reshape::<B>(value, Shape::new([input_elements]));
        let mut output = self.native_empty_like(&value, output_elements, dtype);
        let mut state = RowExchangeProgress { completed_rounds: 0, total_rounds: maximum_peer_rows.div_ceil(peer_rows),
            sent_rows: 0, total_send_rows: rows, received_rows: 0, total_receive_rows: received_rows };
        progress(state);
        for round in 0..state.total_rounds {
            let offset = round * peer_rows;
            let counts = send_counts.iter().map(|count| count.saturating_sub(offset).min(peer_rows)).collect::<Vec<_>>();
            let pieces = counts.iter().enumerate().filter(|(_, count)| **count > 0).map(|(destination, count)| {
                let start = (send_prefix[destination] + offset) * row_elements;
                value.clone().slice(start..start + count * row_elements)
            }).collect::<Vec<_>>();
            let sent = counts.iter().sum::<usize>();
            let packet = if pieces.is_empty() { self.native_empty_like(&value, 0, dtype) } else {
                match &value {
                    Primitive::Float(_) => Primitive::Float(B::float_cat(pieces.into_iter().map(|piece| match piece {
                        Primitive::Float(value) => value, Primitive::Int(_) => unreachable!("original floating source rows"),
                    }).collect(), 0)),
                    Primitive::Int(_) => Primitive::Int(B::int_cat(pieces.into_iter().map(|piece| match piece {
                        Primitive::Int(value) => value, Primitive::Float(_) => unreachable!("original integer source rows"),
                    }).collect(), 0)),
                }
            };
            let mut packet_shape = shape.clone(); packet_shape[0] = sent;
            let packet = native_reshape::<B>(packet, packet_shape);
            let (received, chunk_counts) = match packet {
                Primitive::Float(value) => { let result = self.all_to_all_v_float(value, &counts)?;
                    (Primitive::Float(result.value), result.receive_counts) },
                Primitive::Int(value) => { let result = self.all_to_all_v_int(value, &counts)?;
                    (Primitive::Int(result.value), result.receive_counts) },
            };
            let expected = receive_counts.iter().map(|count| count.saturating_sub(offset).min(peer_rows)).collect::<Vec<_>>();
            if chunk_counts != expected { return Err(TensorDeviceError::InvalidBuffer("native row round receive counts differ from original source matrix")); }
            let chunk_prefix = row_prefix(&chunk_counts)?;
            let received = native_reshape::<B>(received, Shape::new([chunk_prefix[world] * row_elements]));
            for source in 0..world {
                let count = chunk_counts[source]; if count == 0 { continue; }
                let start = (receive_prefix[source] + offset) * row_elements;
                let piece = received.clone().slice(chunk_prefix[source] * row_elements..chunk_prefix[source + 1] * row_elements);
                output = output.assign(start..start + count * row_elements, piece);
            }
            B::sync(self.execution().device())?;
            state.completed_rounds += 1;
            state.sent_rows += sent;
            state.received_rows += chunk_prefix[world];
            progress(state);
        }
        let mut output_shape = shape; output_shape[0] = received_rows;
        Ok((native_reshape::<B>(output, output_shape), receive_counts))
    }
}

fn row_prefix(counts: &[usize]) -> Result<Vec<usize>, TensorDeviceError> {
    let mut result = Vec::with_capacity(counts.len() + 1); result.push(0usize);
    for count in counts { result.push(result.last().unwrap().checked_add(*count)
        .ok_or(TensorDeviceError::InvalidBuffer("variable-row prefix overflow"))?); }
    Ok(result)
}
