use super::*;

impl<T, D, E> InProcessCollective<'_, T, D, E>
where
    T: Copy + Send + Sync + 'static,
    D: InProcessDevice<T>,
    E: From<D::Error> + From<InProcessError>,
{
    /// Equal-count all-to-all. Each input rank stores destination shards in
    /// rank order; each output rank stores source shards in rank order.
    pub fn all_to_all(
        &self,
        input: &DistributedBuffer<T, D>,
    ) -> Result<(DistributedBuffer<T, D>, CollectiveStats), E> {
        self.validate_buffer(input)?;
        if !input.length_per_rank.is_multiple_of(self.world_size()) {
            return Err(InProcessError::InvalidLength(
                "all-to-all rank length must be divisible by world size",
            )
            .into());
        }
        let shard_length = input.length_per_rank / self.world_size();
        let output = self.allocate(input.length_per_rank)?;
        for source in 0..self.world_size() {
            for destination in 0..self.world_size() {
                let values = <D as InProcessDevice<T>>::copy_from_device_at(
                    &self.contexts[source],
                    &input.buffers[source],
                    destination * shard_length,
                    shard_length,
                )?;
                <D as InProcessDevice<T>>::copy_to_device_at(
                    &self.contexts[destination],
                    &output.buffers[destination],
                    source * shard_length,
                    &values,
                )?;
            }
        }
        let mut stats = CollectiveStats::new(CollectiveAlgorithm::Direct);
        stats.steps = self.world_size().saturating_sub(1) as u32;
        stats.transferred_bytes = transferred_bytes(
            self.world_size()
                .saturating_mul(self.world_size().saturating_sub(1)),
            shard_length,
            D::ELEMENT_SIZE,
        )?;
        Ok((output, stats))
    }

    pub fn all_to_all_v(
        &self,
        input: &DistributedBuffer<T, D>,
        send_counts: &[Vec<usize>],
    ) -> Result<(VariableDistributedBuffer<T, D>, CollectiveStats), E> {
        self.validate_buffer(input)?;
        if send_counts.len() != self.world_size()
            || send_counts
                .iter()
                .any(|counts| counts.len() != self.world_size())
        {
            return Err(InProcessError::InvalidLength(
                "all-to-all-v send_counts must be a world_size square matrix",
            )
            .into());
        }
        for counts in send_counts {
            let total = counts
                .iter()
                .try_fold(0_usize, |sum, count| sum.checked_add(*count))
                .ok_or(InProcessError::Overflow("all-to-all-v send count"))?;
            if total != input.length_per_rank {
                return Err(InProcessError::InvalidLength(
                    "each all-to-all-v send-count row must sum to the input rank length",
                )
                .into());
            }
        }
        let receive_lengths = (0..self.world_size())
            .map(|destination| {
                send_counts.iter().try_fold(0_usize, |total, counts| {
                    total.checked_add(counts[destination])
                })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(InProcessError::Overflow("all-to-all-v receive count"))?;
        let output = self.allocate_variable(&receive_lengths)?;
        let mut receive_offsets = vec![0_usize; self.world_size()];
        for (source, counts) in send_counts.iter().enumerate() {
            let values = self.download_rank(input, source)?;
            let mut send_offset = 0_usize;
            for destination in 0..self.world_size() {
                let count = counts[destination];
                let send_end = send_offset
                    .checked_add(count)
                    .ok_or(InProcessError::Overflow("all-to-all-v source range"))?;
                if count != 0 {
                    let destination_buffer = output.buffers[destination]
                        .as_ref()
                        .expect("non-empty all-to-all-v receive has a device buffer");
                    <D as InProcessDevice<T>>::copy_to_device_at(
                        &self.contexts[destination],
                        destination_buffer,
                        receive_offsets[destination],
                        &values[send_offset..send_end],
                    )?;
                }
                receive_offsets[destination] = receive_offsets[destination]
                    .checked_add(count)
                    .ok_or(InProcessError::Overflow("all-to-all-v destination range"))?;
                send_offset = send_end;
            }
        }
        let remote_elements = send_counts
            .iter()
            .enumerate()
            .try_fold(0_usize, |total, (source, counts)| {
                counts
                    .iter()
                    .enumerate()
                    .try_fold(total, |total, (destination, count)| {
                        if source == destination {
                            Some(total)
                        } else {
                            total.checked_add(*count)
                        }
                    })
            })
            .ok_or(InProcessError::Overflow(
                "all-to-all-v transferred elements",
            ))?;
        let mut stats = CollectiveStats::new(CollectiveAlgorithm::Direct);
        stats.steps = self.world_size().saturating_sub(1) as u32;
        stats.transferred_bytes = transferred_bytes(1, remote_elements, D::ELEMENT_SIZE)?;
        Ok((output, stats))
    }
}
