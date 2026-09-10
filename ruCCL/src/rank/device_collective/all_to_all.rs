use super::*;

impl<T, D> DeviceCollective<'_, T, D>
where
    T: Copy + Send + Sync + 'static,
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn all_to_all(&self, input: &D::Buffer) -> Result<(D::Buffer, CollectiveStats), D::Error> {
        if !self
            .execution
            .buffer_len(input)
            .is_multiple_of(self.world_size() as usize)
        {
            return Err(RankError::InvalidLength(
                "TCP all-to-all input length must be divisible by world size",
            )
            .into());
        }
        let plan = self
            .tuning
            .plan(CollectiveKind::AllToAll, self.execution.buffer_bytes(input));
        if plan.algorithm == CollectiveAlgorithm::Pairwise && self.world_size() > 1 {
            return self.all_to_all_pairwise(input, plan.ring_channels);
        }
        let payload = D::encode(&RankDevice::<T>::copy_from_device(self.execution, input)?);
        let response = self.session.exchange(
            Opcode::AllToAll,
            self.element_type,
            ANY_RANK,
            self.execution.buffer_len(input) as u64,
            payload,
        )?;
        let values = self.decode_exact(&response.payload, self.execution.buffer_len(input))?;
        let output = RankDevice::<T>::alloc(self.execution, self.execution.buffer_len(input))?;
        RankDevice::<T>::copy_to_device(self.execution, &output, &values)?;
        let stats = self.stats(
            CollectiveAlgorithm::Direct,
            self.world_size().saturating_sub(1),
            self.execution.buffer_bytes(input) + response.payload.len(),
            0,
        )?;
        Ok((output, stats))
    }

    pub fn all_to_all_v(
        &self,
        input: &D::Buffer,
        send_counts: &[usize],
    ) -> Result<(VariableCollectiveOutput<D::Buffer>, CollectiveStats), D::Error> {
        if send_counts.len() != self.world_size() as usize {
            return Err(RankError::InvalidLength(
                "TCP all-to-all-v send_counts must have one entry per rank",
            )
            .into());
        }
        let send_total = send_counts
            .iter()
            .try_fold(0_usize, |total, count| total.checked_add(*count))
            .ok_or(RankError::Overflow("TCP all-to-all-v send count"))?;
        if send_total != self.execution.buffer_len(input) {
            return Err(RankError::InvalidLength(
                "TCP all-to-all-v send_counts do not sum to input length",
            )
            .into());
        }
        let plan = self
            .tuning
            .plan(CollectiveKind::AllToAll, self.execution.buffer_bytes(input));
        if plan.algorithm == CollectiveAlgorithm::Pairwise && self.world_size() > 1 {
            return self.all_to_all_v_pairwise(input, send_counts, plan.ring_channels);
        }
        let mut payload = encode_counts(send_counts)?;
        payload.extend_from_slice(&D::encode(&RankDevice::<T>::copy_from_device(
            self.execution,
            input,
        )?));
        let request_bytes = payload.len();
        let response = self.session.exchange_with_options(
            Opcode::AllToAllV,
            self.element_type,
            ExchangeOptions {
                root_rank: ANY_RANK,
                element_count: self.execution.buffer_len(input) as u64,
                flags: FLAG_COUNTS_PREFIX,
                tag: 0,
            },
            payload,
        )?;
        let (receive_counts, data) =
            decode_counts_response(&response, self.world_size() as usize, D::ELEMENT_SIZE)?;
        let output_length = receive_counts
            .iter()
            .try_fold(0_usize, |total, count| total.checked_add(*count))
            .ok_or(RankError::Overflow("TCP all-to-all-v receive count"))?;
        let buffer = if output_length == 0 {
            if !data.is_empty() {
                return Err(NetworkError::InvalidConfiguration(
                    "zero-length all-to-all-v response contains data".into(),
                )
                .into());
            }
            None
        } else {
            let values = self.decode_exact(data, output_length)?;
            let output = RankDevice::<T>::alloc(self.execution, output_length)?;
            RankDevice::<T>::copy_to_device(self.execution, &output, &values)?;
            Some(output)
        };
        let stats = self.stats(
            CollectiveAlgorithm::Direct,
            self.world_size().saturating_sub(1),
            request_bytes + response.payload.len(),
            0,
        )?;
        Ok((
            VariableCollectiveOutput {
                buffer,
                counts: receive_counts,
            },
            stats,
        ))
    }

    fn all_to_all_pairwise(
        &self,
        input: &D::Buffer,
        ring_channels: usize,
    ) -> Result<(D::Buffer, CollectiveStats), D::Error> {
        let world_size = self.world_size() as usize;
        let shard_length = self.execution.buffer_len(input) / world_size;
        let ring = &self.tuning.ring_order;
        let position = self.ring_position(ring)?;
        let output = RankDevice::<T>::alloc(self.execution, self.execution.buffer_len(input))?;
        let self_start = self.rank() as usize * shard_length;
        let self_values = RankDevice::<T>::copy_bytes_from_device_at(
            self.execution,
            input,
            self_start,
            shard_length,
        )?;
        RankDevice::<T>::copy_bytes_to_device_at(
            self.execution,
            &output,
            self_start,
            &self_values,
        )?;
        let operation = self.internal_sequence.fetch_add(1, Ordering::Relaxed);
        let channels = self.effective_ring_channels(ring_channels, shard_length);
        self.ring_agreement(operation, self.execution.buffer_len(input), 0x400, channels)?;
        let mut transferred_bytes = 0_usize;
        for step in 1..world_size {
            let destination = ring[(position + step) % world_size];
            let source = ring[(position + world_size - step) % world_size];
            let send_start = destination as usize * shard_length;
            let channel_results = std::thread::scope(|scope| {
                balanced_ranges(shard_length, channels)
                    .into_iter()
                    .enumerate()
                    .map(|(channel, range)| {
                        let communicator = self.clone();
                        let input = input.clone();
                        scope.spawn(move || {
                            let send_range = send_start + range.start..send_start + range.end;
                            let result = communicator
                                .exchange_ring_chunk_bytes(
                                    &input,
                                    send_range,
                                    range.len(),
                                    destination,
                                    source,
                                    channel,
                                    ring_data_tag(operation, 0, channel, step),
                                )
                                .map(|received| (range, received));
                            communicator.propagate_channel_failure("pairwise all-to-all", result)
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|handle| handle.join().map_err(|_| WorkError::WorkerPanic)?)
                    .collect::<Result<Vec<_>, D::Error>>()
            })?;
            for (range, received) in channel_results {
                RankDevice::<T>::copy_bytes_to_device_at(
                    self.execution,
                    &output,
                    source as usize * shard_length + range.start,
                    &received,
                )?;
                transferred_bytes = transferred_bytes
                    .checked_add(received.len().saturating_mul(2))
                    .ok_or(RankError::Overflow("pairwise all-to-all transferred bytes"))?;
            }
        }
        let stats = self.stats(
            CollectiveAlgorithm::Pairwise,
            (world_size - 1) as u32,
            transferred_bytes,
            0,
        )?;
        Ok((output, stats))
    }

    fn all_to_all_v_pairwise(
        &self,
        input: &D::Buffer,
        send_counts: &[usize],
        ring_channels: usize,
    ) -> Result<(VariableCollectiveOutput<D::Buffer>, CollectiveStats), D::Error> {
        let world_size = self.world_size() as usize;
        let ring = &self.tuning.ring_order;
        let position = self.ring_position(ring)?;
        let operation = self.internal_sequence.fetch_add(1, Ordering::Relaxed);
        let channels = ring_channels.max(1);
        // AllToAllV permits every rank to contribute a different number of elements.
        // The count exchange below establishes the peer-specific layouts, so only the
        // common algorithm/ring/channel contract belongs in the agreement hash.
        self.ring_agreement(operation, 0, 0x500, channels)?;

        let mut receive_counts = vec![0_usize; world_size];
        receive_counts[self.rank() as usize] = send_counts[self.rank() as usize];
        let mut transferred_bytes = 0_usize;
        for step in 1..world_size {
            let destination = ring[(position + step) % world_size];
            let source = ring[(position + world_size - step) % world_size];
            let encoded = u64::try_from(send_counts[destination as usize])
                .map_err(|_| RankError::Overflow("pairwise all-to-all-v send count"))?
                .to_le_bytes()
                .to_vec();
            let received = self.exchange_host_payload(
                ElementType::U8,
                encoded,
                8,
                destination,
                source,
                0,
                ring_data_tag(operation, 0, 0, step),
            )?;
            receive_counts[source as usize] = usize::try_from(u64::from_le_bytes(
                received
                    .as_slice()
                    .try_into()
                    .expect("count exchange returns exactly eight bytes"),
            ))
            .map_err(|_| RankError::Overflow("pairwise all-to-all-v receive count"))?;
            transferred_bytes = transferred_bytes
                .checked_add(16)
                .ok_or(RankError::Overflow(
                    "pairwise all-to-all-v transferred bytes",
                ))?;
        }

        let send_offsets = prefix_offsets(send_counts)?;
        let receive_offsets = prefix_offsets(&receive_counts)?;
        let output_length = receive_counts
            .iter()
            .try_fold(0_usize, |total, count| total.checked_add(*count))
            .ok_or(RankError::Overflow("pairwise all-to-all-v receive count"))?;
        let output = if output_length == 0 {
            None
        } else {
            Some(RankDevice::<T>::alloc(self.execution, output_length)?)
        };
        let self_count = send_counts[self.rank() as usize];
        if self_count != 0 {
            let values = RankDevice::<T>::copy_bytes_from_device_at(
                self.execution,
                input,
                send_offsets[self.rank() as usize],
                self_count,
            )?;
            RankDevice::<T>::copy_bytes_to_device_at(
                self.execution,
                output.as_ref().expect("non-empty self receive has output"),
                receive_offsets[self.rank() as usize],
                &values,
            )?;
        }
        for step in 1..world_size {
            let destination = ring[(position + step) % world_size];
            let source = ring[(position + world_size - step) % world_size];
            let send_start = send_offsets[destination as usize];
            let send_length = send_counts[destination as usize];
            let receive_length = receive_counts[source as usize];
            let send_ranges = balanced_ranges(send_length, channels);
            let receive_ranges = balanced_ranges(receive_length, channels);
            let channel_results = std::thread::scope(|scope| {
                send_ranges
                    .into_iter()
                    .zip(receive_ranges)
                    .enumerate()
                    .map(|(channel, (send_range, receive_range))| {
                        let communicator = self.clone();
                        let input = input.clone();
                        scope.spawn(move || {
                            let absolute_send =
                                send_start + send_range.start..send_start + send_range.end;
                            let result = communicator
                                .exchange_ring_chunk_bytes(
                                    &input,
                                    absolute_send,
                                    receive_range.len(),
                                    destination,
                                    source,
                                    channel,
                                    ring_data_tag(operation, 1, channel, step),
                                )
                                .map(|received| (send_range.len(), receive_range, received));
                            communicator.propagate_channel_failure("pairwise all-to-all-v", result)
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|handle| handle.join().map_err(|_| WorkError::WorkerPanic)?)
                    .collect::<Result<Vec<_>, D::Error>>()
            })?;
            for (sent, receive_range, received) in channel_results {
                if !received.is_empty() {
                    RankDevice::<T>::copy_bytes_to_device_at(
                        self.execution,
                        output.as_ref().expect("non-empty peer receive has output"),
                        receive_offsets[source as usize] + receive_range.start,
                        &received,
                    )?;
                }
                transferred_bytes = transferred_bytes
                    .checked_add(
                        sent.saturating_mul(D::ELEMENT_SIZE)
                            .saturating_add(received.len()),
                    )
                    .ok_or(RankError::Overflow(
                        "pairwise all-to-all-v transferred bytes",
                    ))?;
            }
        }
        let stats = self.stats(
            CollectiveAlgorithm::Pairwise,
            (2 * (world_size - 1)) as u32,
            transferred_bytes,
            0,
        )?;
        Ok((
            VariableCollectiveOutput {
                buffer: output,
                counts: receive_counts,
            },
            stats,
        ))
    }
}
