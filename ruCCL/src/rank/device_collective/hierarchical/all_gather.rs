use super::*;

impl<T, D> DeviceCollective<'_, T, D>
where
    T: Copy + Send + Sync + 'static,
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn all_gather_hierarchical(
        &self,
        input: &D::Buffer,
        ring_channels: usize,
    ) -> Result<(D::Buffer, CollectiveStats), D::Error> {
        let (groups, group_index, _) = self.hierarchy_membership()?;
        let group = &groups[group_index];
        let leader = group[0];
        let leaders = groups.iter().map(|group| group[0]).collect::<Vec<_>>();
        let operation = self.internal_sequence.fetch_add(1, Ordering::Relaxed);
        self.hierarchy_agreement(
            operation,
            self.execution.buffer_len(input),
            0x700,
            ring_channels,
        )?;

        let output_length = self
            .execution
            .buffer_len(input)
            .checked_mul(self.world_size() as usize)
            .ok_or(RankError::Overflow("hierarchical all-gather output length"))?;
        let local = RankDevice::<T>::copy_from_device(self.execution, input)?;
        let steps = hierarchy_steps(groups, leaders.len().saturating_sub(1));

        if self.rank() != leader {
            let payload = D::encode(&local);
            let mut transferred_bytes = payload.len();
            self.session.send(
                leader,
                hierarchy_data_tag(operation, 4, self.rank()),
                self.element_type,
                self.execution.buffer_len(input) as u64,
                payload,
            )?;
            let values = self.receive_values(
                leader,
                hierarchy_data_tag(operation, 6, self.rank()),
                output_length,
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(values.len().saturating_mul(D::ELEMENT_SIZE))
                .ok_or(RankError::Overflow(
                    "hierarchical all-gather transferred bytes",
                ))?;
            let output = RankDevice::<T>::alloc(self.execution, output_length)?;
            RankDevice::<T>::copy_to_device(self.execution, &output, &values)?;
            return Ok((
                output,
                self.stats(
                    CollectiveAlgorithm::Hierarchical,
                    steps,
                    transferred_bytes,
                    0,
                )?,
            ));
        }

        let own_bundle_length = group
            .len()
            .checked_mul(self.execution.buffer_len(input))
            .ok_or(RankError::Overflow(
                "hierarchical all-gather local bundle length",
            ))?;
        let mut own_bundle = Vec::with_capacity(own_bundle_length);
        let mut transferred_bytes = 0_usize;
        for member in group {
            if *member == leader {
                own_bundle.extend_from_slice(&local);
            } else {
                let values = self.receive_values(
                    *member,
                    hierarchy_data_tag(operation, 4, *member),
                    self.execution.buffer_len(input),
                )?;
                transferred_bytes = transferred_bytes
                    .checked_add(values.len().saturating_mul(D::ELEMENT_SIZE))
                    .ok_or(RankError::Overflow(
                        "hierarchical all-gather transferred bytes",
                    ))?;
                own_bundle.extend_from_slice(&values);
            }
        }

        let (bundles, leader_bytes) = self.leader_all_gather_ring(
            &own_bundle,
            groups,
            &leaders,
            group_index,
            self.execution.buffer_len(input),
            operation,
            ring_channels,
        )?;
        transferred_bytes =
            transferred_bytes
                .checked_add(leader_bytes)
                .ok_or(RankError::Overflow(
                    "hierarchical all-gather transferred bytes",
                ))?;

        let mut rank_values = vec![Vec::<T>::new(); self.world_size() as usize];
        for (bundle_group, bundle) in groups.iter().zip(bundles) {
            let mut offset = 0_usize;
            for member in bundle_group {
                let end = offset
                    .checked_add(self.execution.buffer_len(input))
                    .ok_or(RankError::Overflow("hierarchical all-gather bundle offset"))?;
                rank_values[*member as usize].extend_from_slice(&bundle[offset..end]);
                offset = end;
            }
        }
        let mut output_values = Vec::with_capacity(output_length);
        for values in rank_values {
            output_values.extend_from_slice(&values);
        }
        if output_values.len() != output_length {
            return Err(NetworkError::InvalidConfiguration(
                "hierarchical all-gather assembled the wrong output length".into(),
            )
            .into());
        }

        let output_payload = D::encode(&output_values);
        for member in &group[1..] {
            self.session.send(
                *member,
                hierarchy_data_tag(operation, 6, *member),
                self.element_type,
                output_length as u64,
                output_payload.clone(),
            )?;
            transferred_bytes =
                transferred_bytes
                    .checked_add(output_payload.len())
                    .ok_or(RankError::Overflow(
                        "hierarchical all-gather transferred bytes",
                    ))?;
        }
        let output = RankDevice::<T>::alloc(self.execution, output_length)?;
        RankDevice::<T>::copy_to_device(self.execution, &output, &output_values)?;
        Ok((
            output,
            self.stats(
                CollectiveAlgorithm::Hierarchical,
                steps,
                transferred_bytes,
                0,
            )?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn leader_all_gather_ring(
        &self,
        own_bundle: &[T],
        groups: &[Vec<u32>],
        leaders: &[u32],
        position: usize,
        input_length: usize,
        operation: u64,
        ring_channels: usize,
    ) -> Result<(Vec<Vec<T>>, usize), D::Error> {
        let channels = self.effective_ring_channels(ring_channels, input_length);
        let own_channel_ranges = balanced_ranges(own_bundle.len(), channels);
        let channel_results = std::thread::scope(|scope| {
            let handles = own_channel_ranges
                .into_iter()
                .enumerate()
                .map(|(channel, own_range)| {
                    let communicator = self.clone();
                    let groups = groups.to_vec();
                    let leaders = leaders.to_vec();
                    let own_piece = own_bundle[own_range].to_vec();
                    scope.spawn(move || {
                        let result = communicator.leader_all_gather_ring_channel(
                            own_piece,
                            &groups,
                            &leaders,
                            position,
                            input_length,
                            operation,
                            channel,
                            channels,
                        );
                        communicator.propagate_channel_failure("hierarchical all-gather", result)
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().map_err(|_| WorkError::WorkerPanic)?)
                .collect::<Result<Vec<_>, D::Error>>()
        })?;

        let mut bundles = vec![Vec::new(); groups.len()];
        let mut transferred_bytes = 0_usize;
        for (pieces, bytes) in channel_results {
            transferred_bytes = transferred_bytes
                .checked_add(bytes)
                .ok_or(RankError::Overflow(
                    "hierarchical all-gather transferred bytes",
                ))?;
            for (group_index, piece) in pieces.into_iter().enumerate() {
                bundles[group_index].extend_from_slice(&piece.ok_or_else(|| {
                    NetworkError::InvalidConfiguration(
                        "hierarchical all-gather did not receive every leader bundle channel"
                            .into(),
                    )
                })?);
            }
        }
        for (group, bundle) in groups.iter().zip(&bundles) {
            let expected = group
                .len()
                .checked_mul(input_length)
                .ok_or(RankError::Overflow("hierarchical all-gather bundle length"))?;
            if bundle.len() != expected {
                return Err(NetworkError::InvalidConfiguration(format!(
                    "hierarchical all-gather bundle has {} elements, expected {expected}",
                    bundle.len()
                ))
                .into());
            }
        }
        Ok((bundles, transferred_bytes))
    }

    #[allow(clippy::too_many_arguments)]
    fn leader_all_gather_ring_channel(
        &self,
        own_piece: Vec<T>,
        groups: &[Vec<u32>],
        leaders: &[u32],
        position: usize,
        input_length: usize,
        operation: u64,
        channel: usize,
        channels: usize,
    ) -> Result<(LeaderGatherPieces<T>, usize), D::Error> {
        let leader_count = leaders.len();
        let previous_leader = leaders[(position + leader_count - 1) % leader_count];
        let next_leader = leaders[(position + 1) % leader_count];
        let mut pieces = vec![None; leader_count];
        pieces[position] = Some(own_piece);
        let mut transferred_bytes = 0_usize;
        for step in 0..leader_count - 1 {
            let send_group = (position + leader_count - step) % leader_count;
            let receive_group = (position + leader_count - step - 1) % leader_count;
            let send_values = pieces[send_group]
                .as_ref()
                .expect("leader ring forwards a bundle channel it already owns");
            let payload = D::encode(send_values);
            let receive_total = groups[receive_group]
                .len()
                .checked_mul(input_length)
                .ok_or(RankError::Overflow(
                    "hierarchical all-gather leader bundle length",
                ))?;
            let receive_elements = balanced_ranges(receive_total, channels)[channel].len();
            let receive_bytes =
                receive_elements
                    .checked_mul(D::ELEMENT_SIZE)
                    .ok_or(RankError::Overflow(
                        "hierarchical all-gather leader bundle bytes",
                    ))?;
            let send_bytes = payload.len();
            let received = self.exchange_host_payload(
                self.element_type,
                payload,
                receive_bytes,
                next_leader,
                previous_leader,
                channel,
                hierarchy_ring_data_tag(operation, 5, channel, step),
            )?;
            transferred_bytes = transferred_bytes
                .checked_add(send_bytes.saturating_add(received.len()))
                .ok_or(RankError::Overflow(
                    "hierarchical all-gather transferred bytes",
                ))?;
            pieces[receive_group] = Some(decode_exact::<T, D::Error>(
                &received,
                receive_elements,
                D::ELEMENT_SIZE,
                D::decode,
            )?);
        }
        Ok((pieces, transferred_bytes))
    }
}
