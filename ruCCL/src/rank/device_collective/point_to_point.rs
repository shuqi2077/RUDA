use super::*;

impl<T, D> DeviceCollective<'_, T, D>
where
    T: Copy + Send + Sync + 'static,
    D: RankDevice<T>,
    D::Error: From<RankError> + From<NetworkError> + From<TopologyError> + From<WorkError>,
{
    pub fn send(
        &self,
        input: &D::Buffer,
        destination: u32,
        tag: u64,
    ) -> Result<CollectiveStats, D::Error> {
        let payload = D::encode(&RankDevice::<T>::copy_from_device(self.execution, input)?);
        let transferred_bytes = payload.len();
        self.session.send(
            destination,
            tag,
            self.element_type,
            self.execution.buffer_len(input) as u64,
            payload,
        )?;
        self.stats(CollectiveAlgorithm::Direct, 1, transferred_bytes, 0)
    }

    pub fn receive(
        &self,
        source: Option<u32>,
        tag: u64,
        length: usize,
    ) -> Result<(ReceivedMessage<D::Buffer>, CollectiveStats), D::Error> {
        let response = self
            .session
            .receive(source, tag, self.element_type, length as u64)?;
        if response.header.tag != tag
            || response.header.element_type != self.element_type
            || response.header.element_count != length as u64
            || source.is_some_and(|source| response.header.source_rank != source)
        {
            return Err(NetworkError::InvalidConfiguration(
                "TCP point-to-point response does not match the receive request".into(),
            )
            .into());
        }
        let values = self.decode_exact(&response.payload, length)?;
        let buffer = RankDevice::<T>::alloc(self.execution, length)?;
        RankDevice::<T>::copy_to_device(self.execution, &buffer, &values)?;
        let stats = self.stats(CollectiveAlgorithm::Direct, 1, response.payload.len(), 0)?;
        Ok((
            ReceivedMessage {
                buffer,
                source_rank: response.header.source_rank,
                tag,
            },
            stats,
        ))
    }
}
