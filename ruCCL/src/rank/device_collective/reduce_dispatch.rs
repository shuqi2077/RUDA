use super::*;

impl<T, D> DeviceCollective<'_, T, D>
where
    T: Copy + Send + Sync + 'static,
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn all_reduce(
        &self,
        buffer: &D::Buffer,
        operation: ReductionOperation,
        function: &D::Kernel,
    ) -> Result<CollectiveStats, D::Error> {
        if self.world_size() > 1 {
            let plan = self.tuning.plan(
                CollectiveKind::AllReduce,
                self.execution.buffer_bytes(buffer),
            );
            match plan.algorithm {
                CollectiveAlgorithm::Hierarchical => {
                    return self.all_reduce_hierarchical(
                        buffer,
                        operation,
                        function,
                        plan.ring_channels,
                    );
                }
                CollectiveAlgorithm::Ring => {
                    return self.all_reduce_ring(buffer, operation, function, plan.ring_channels);
                }
                CollectiveAlgorithm::Direct | CollectiveAlgorithm::Pairwise => {}
            }
        }
        let payload = D::encode(&RankDevice::<T>::copy_from_device(self.execution, buffer)?);
        let request_bytes = payload.len();
        let response = self.session.exchange_with_options(
            Opcode::AllReduce,
            self.element_type,
            reduction_exchange_options(
                ANY_RANK,
                self.execution.buffer_len(buffer),
                operation as u32,
            ),
            payload,
        )?;
        let launches = self.reduce_concatenated(
            &response.payload,
            buffer,
            self.execution.buffer_len(buffer),
            function,
        )?;
        self.stats(
            CollectiveAlgorithm::Direct,
            self.world_size().saturating_sub(1),
            request_bytes + response.payload.len(),
            launches,
        )
    }

    pub fn reduce(
        &self,
        input: &D::Buffer,
        root: u32,
        operation: ReductionOperation,
        function: &D::Kernel,
    ) -> Result<(Option<D::Buffer>, CollectiveStats), D::Error> {
        self.validate_root(root)?;
        let payload = D::encode(&RankDevice::<T>::copy_from_device(self.execution, input)?);
        let request_bytes = payload.len();
        let response = self.session.exchange_with_options(
            Opcode::Reduce,
            self.element_type,
            reduction_exchange_options(root, self.execution.buffer_len(input), operation as u32),
            payload,
        )?;
        let (output, launches) = if self.rank() == root {
            let output = RankDevice::<T>::alloc(self.execution, self.execution.buffer_len(input))?;
            let launches = self.reduce_concatenated(
                &response.payload,
                &output,
                self.execution.buffer_len(input),
                function,
            )?;
            (Some(output), launches)
        } else {
            if !response.payload.is_empty() {
                return Err(NetworkError::InvalidConfiguration(
                    "non-root reduce response contains data".into(),
                )
                .into());
            }
            (None, 0)
        };
        let stats = self.stats(
            CollectiveAlgorithm::Direct,
            self.world_size().saturating_sub(1),
            request_bytes + response.payload.len(),
            launches,
        )?;
        Ok((output, stats))
    }

    pub fn reduce_scatter(
        &self,
        input: &D::Buffer,
        operation: ReductionOperation,
        function: &D::Kernel,
    ) -> Result<(D::Buffer, CollectiveStats), D::Error> {
        if !self
            .execution
            .buffer_len(input)
            .is_multiple_of(self.world_size() as usize)
        {
            return Err(RankError::InvalidLength(
                "TCP reduce-scatter input length must be divisible by world size",
            )
            .into());
        }
        let output_length = self.execution.buffer_len(input) / self.world_size() as usize;
        if self.world_size() > 1 {
            let plan = self.tuning.plan(
                CollectiveKind::ReduceScatter,
                self.execution.buffer_bytes(input),
            );
            match plan.algorithm {
                CollectiveAlgorithm::Hierarchical => {
                    return self.reduce_scatter_hierarchical(
                        input,
                        output_length,
                        operation,
                        function,
                        plan.ring_channels,
                    );
                }
                CollectiveAlgorithm::Ring => {
                    return self.reduce_scatter_ring(
                        input,
                        output_length,
                        operation,
                        function,
                        plan.ring_channels,
                    );
                }
                CollectiveAlgorithm::Direct | CollectiveAlgorithm::Pairwise => {}
            }
        }
        let payload = D::encode(&RankDevice::<T>::copy_from_device(self.execution, input)?);
        let request_bytes = payload.len();
        let response = self.session.exchange_with_options(
            Opcode::ReduceScatter,
            self.element_type,
            reduction_exchange_options(
                ANY_RANK,
                self.execution.buffer_len(input),
                operation as u32,
            ),
            payload,
        )?;
        let output = RankDevice::<T>::alloc(self.execution, output_length)?;
        let launches =
            self.reduce_concatenated(&response.payload, &output, output_length, function)?;
        let stats = self.stats(
            CollectiveAlgorithm::Direct,
            self.world_size().saturating_sub(1),
            request_bytes + response.payload.len(),
            launches,
        )?;
        Ok((output, stats))
    }
}
