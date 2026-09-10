use super::*;

impl<T, D> DeviceCollective<'_, T, D>
where
    T: Copy + Send + Sync + 'static,
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn broadcast(&self, buffer: &D::Buffer, root: u32) -> Result<CollectiveStats, D::Error> {
        self.validate_root(root)?;
        let payload = if self.rank() == root {
            D::encode(&RankDevice::<T>::copy_from_device(self.execution, buffer)?)
        } else {
            Vec::new()
        };
        let response = self.session.exchange(
            Opcode::Broadcast,
            self.element_type,
            root,
            self.execution.buffer_len(buffer) as u64,
            payload,
        )?;
        let values = self.decode_exact(&response.payload, self.execution.buffer_len(buffer))?;
        RankDevice::<T>::copy_to_device(self.execution, buffer, &values)?;
        self.stats(
            CollectiveAlgorithm::Direct,
            u32::from(self.world_size() > 1),
            self.execution.buffer_bytes(buffer),
            0,
        )
    }

    pub fn all_gather(&self, input: &D::Buffer) -> Result<(D::Buffer, CollectiveStats), D::Error> {
        if self.world_size() > 1 {
            let plan = self.tuning.plan(
                CollectiveKind::AllGather,
                self.execution.buffer_bytes(input),
            );
            match plan.algorithm {
                CollectiveAlgorithm::Hierarchical => {
                    return self.all_gather_hierarchical(input, plan.ring_channels);
                }
                CollectiveAlgorithm::Ring => {
                    return self.all_gather_ring(input, plan.ring_channels);
                }
                CollectiveAlgorithm::Direct | CollectiveAlgorithm::Pairwise => {}
            }
        }
        let payload = D::encode(&RankDevice::<T>::copy_from_device(self.execution, input)?);
        let response = self.session.exchange(
            Opcode::AllGather,
            self.element_type,
            ANY_RANK,
            self.execution.buffer_len(input) as u64,
            payload,
        )?;
        let output_length = self
            .execution
            .buffer_len(input)
            .checked_mul(self.world_size() as usize)
            .ok_or(RankError::Overflow("TCP all-gather output length"))?;
        let values = self.decode_exact(&response.payload, output_length)?;
        let output = RankDevice::<T>::alloc(self.execution, output_length)?;
        RankDevice::<T>::copy_to_device(self.execution, &output, &values)?;
        let stats = self.stats(
            CollectiveAlgorithm::Direct,
            u32::from(self.world_size() > 1),
            self.execution.buffer_bytes(input) + response.payload.len(),
            0,
        )?;
        Ok((output, stats))
    }

    pub fn gather(
        &self,
        input: &D::Buffer,
        root: u32,
    ) -> Result<(Option<D::Buffer>, CollectiveStats), D::Error> {
        self.validate_root(root)?;
        let payload = D::encode(&RankDevice::<T>::copy_from_device(self.execution, input)?);
        let response = self.session.exchange(
            Opcode::Gather,
            self.element_type,
            root,
            self.execution.buffer_len(input) as u64,
            payload,
        )?;
        let output = if self.rank() == root {
            let output_length = self
                .execution
                .buffer_len(input)
                .checked_mul(self.world_size() as usize)
                .ok_or(RankError::Overflow("TCP gather output length"))?;
            let values = self.decode_exact(&response.payload, output_length)?;
            let output = RankDevice::<T>::alloc(self.execution, output_length)?;
            RankDevice::<T>::copy_to_device(self.execution, &output, &values)?;
            Some(output)
        } else {
            if !response.payload.is_empty() {
                return Err(NetworkError::InvalidConfiguration(
                    "non-root gather response contains data".into(),
                )
                .into());
            }
            None
        };
        let stats = self.stats(
            CollectiveAlgorithm::Direct,
            u32::from(self.world_size() > 1),
            self.execution.buffer_bytes(input) + response.payload.len(),
            0,
        )?;
        Ok((output, stats))
    }

    pub fn scatter(
        &self,
        root_input: Option<&D::Buffer>,
        root: u32,
        output_length: usize,
    ) -> Result<(D::Buffer, CollectiveStats), D::Error> {
        self.validate_root(root)?;
        let total_length = output_length
            .checked_mul(self.world_size() as usize)
            .ok_or(RankError::Overflow("TCP scatter input length"))?;
        let payload = match (self.rank() == root, root_input) {
            (true, Some(input)) if self.execution.buffer_len(input) == total_length => {
                D::encode(&RankDevice::<T>::copy_from_device(self.execution, input)?)
            }
            (true, Some(_)) => {
                return Err(RankError::InvalidLength(
                    "TCP scatter root input has the wrong length",
                )
                .into());
            }
            (true, None) => {
                return Err(RankError::InvalidLength(
                    "TCP scatter root must provide the input buffer",
                )
                .into());
            }
            (false, None) => Vec::new(),
            (false, Some(_)) => {
                return Err(RankError::InvalidLength(
                    "non-root TCP scatter rank cannot provide the input buffer",
                )
                .into());
            }
        };
        let response = self.session.exchange(
            Opcode::Scatter,
            self.element_type,
            root,
            total_length as u64,
            payload,
        )?;
        let values = self.decode_exact(&response.payload, output_length)?;
        let output = RankDevice::<T>::alloc(self.execution, output_length)?;
        RankDevice::<T>::copy_to_device(self.execution, &output, &values)?;
        let stats = self.stats(
            CollectiveAlgorithm::Direct,
            u32::from(self.world_size() > 1),
            response.payload.len()
                + root_input.map_or(0, |buffer| self.execution.buffer_bytes(buffer)),
            0,
        )?;
        Ok((output, stats))
    }
}
