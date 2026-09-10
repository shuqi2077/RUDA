use super::super::error::RankError;
use super::super::payload::{validate_host_reduction_payload, validate_host_reduction_response};
use super::super::{
    CollectiveAlgorithm, CollectiveStats, CollectiveTransport, ElementType, Opcode, RankTransport,
    ReductionOperation, NetworkError, tags,
};
use std::time::Duration;

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct HostStagedExchange<'a> {
    session: &'a dyn RankTransport,
}

impl<'a> HostStagedExchange<'a> {
    pub fn new(session: &'a dyn RankTransport) -> Self {
        Self { session }
    }

    pub fn rank(&self) -> u32 {
        self.session.rank()
    }

    pub fn world_size(&self) -> u32 {
        self.session.world_size()
    }

    pub fn transport(&self) -> CollectiveTransport {
        self.session.transport()
    }

    pub fn heartbeat(&self, timeout: Duration) -> Result<Duration, RankError> {
        Ok(self.session.heartbeat(timeout)?)
    }

    pub fn all_reduce_host_staged(
        &self,
        element_type: ElementType,
        element_count: usize,
        payload: Vec<u8>,
        operation: ReductionOperation,
    ) -> Result<(Vec<u8>, CollectiveStats), RankError> {
        let rank_bytes = validate_host_reduction_payload(element_type, element_count, &payload)?;
        let response = self.session.exchange_with_options(
            Opcode::AllReduce,
            element_type,
            tags::reduction_exchange_options(
                super::super::ANY_RANK,
                element_count,
                operation as u32,
            ),
            payload,
        )?;
        let expected = rank_bytes
            .checked_mul(self.world_size() as usize)
            .ok_or(RankError::Overflow("host all-reduce response bytes"))?;
        validate_host_reduction_response("all-reduce", &response.payload, expected)?;
        let stats = self.stats(
            CollectiveAlgorithm::Direct,
            self.world_size().saturating_sub(1),
            rank_bytes
                .checked_add(response.payload.len())
                .ok_or(RankError::Overflow("host all-reduce transferred bytes"))?,
            0,
        )?;
        Ok((response.payload, stats))
    }

    pub fn reduce_host_staged(
        &self,
        element_type: ElementType,
        element_count: usize,
        payload: Vec<u8>,
        root: u32,
        operation: ReductionOperation,
    ) -> Result<(Option<Vec<u8>>, CollectiveStats), RankError> {
        self.validate_root(root)?;
        let rank_bytes = validate_host_reduction_payload(element_type, element_count, &payload)?;
        let response = self.session.exchange_with_options(
            Opcode::Reduce,
            element_type,
            tags::reduction_exchange_options(root, element_count, operation as u32),
            payload,
        )?;
        let expected = if self.rank() == root {
            rank_bytes
                .checked_mul(self.world_size() as usize)
                .ok_or(RankError::Overflow("host reduce response bytes"))?
        } else {
            0
        };
        validate_host_reduction_response("reduce", &response.payload, expected)?;
        let stats = self.stats(
            CollectiveAlgorithm::Direct,
            self.world_size().saturating_sub(1),
            rank_bytes
                .checked_add(response.payload.len())
                .ok_or(RankError::Overflow("host reduce transferred bytes"))?,
            0,
        )?;
        Ok(((self.rank() == root).then_some(response.payload), stats))
    }

    pub fn reduce_scatter_host_staged(
        &self,
        element_type: ElementType,
        element_count: usize,
        payload: Vec<u8>,
        operation: ReductionOperation,
    ) -> Result<(Vec<u8>, CollectiveStats), RankError> {
        if !element_count.is_multiple_of(self.world_size() as usize) {
            return Err(RankError::InvalidLength(
                "host reduce-scatter input length must be divisible by world size",
            ));
        }
        let rank_bytes = validate_host_reduction_payload(element_type, element_count, &payload)?;
        let response = self.session.exchange_with_options(
            Opcode::ReduceScatter,
            element_type,
            tags::reduction_exchange_options(
                super::super::ANY_RANK,
                element_count,
                operation as u32,
            ),
            payload,
        )?;
        validate_host_reduction_response("reduce-scatter", &response.payload, rank_bytes)?;
        let stats = self.stats(
            CollectiveAlgorithm::Direct,
            self.world_size().saturating_sub(1),
            rank_bytes
                .checked_add(response.payload.len())
                .ok_or(RankError::Overflow("host reduce-scatter transferred bytes"))?,
            0,
        )?;
        Ok((response.payload, stats))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn exchange_host_payload(
        &self,
        collective_rail_order: &[usize],
        element_type: ElementType,
        payload: Vec<u8>,
        receive_bytes: usize,
        destination: u32,
        source: u32,
        rail_hint: usize,
        tag: u64,
    ) -> Result<Vec<u8>, RankError> {
        let element_bytes = element_type.byte_width();
        if element_bytes == 0
            || !payload.len().is_multiple_of(element_bytes)
            || !receive_bytes.is_multiple_of(element_bytes)
        {
            return Err(RankError::InvalidLength(
                "pairwise host payload does not align to its element type",
            ));
        }
        let send_elements = payload.len() / element_bytes;
        let receive_elements = receive_bytes / element_bytes;
        let rail = collective_rail_order[rail_hint % collective_rail_order.len()];
        self.session.send_on_rail(
            rail,
            destination,
            tag,
            element_type,
            send_elements as u64,
            payload,
        )?;
        let response = self.session.receive_on_rail(
            rail,
            Some(source),
            tag,
            element_type,
            receive_elements as u64,
        )?;
        if response.payload.len() != receive_bytes {
            return Err(NetworkError::InvalidConfiguration(format!(
                "pairwise host response has {} bytes, expected {receive_bytes}",
                response.payload.len()
            ))
            .into());
        }
        Ok(response.payload)
    }

    pub fn barrier(&self) -> Result<CollectiveStats, RankError> {
        self.session.barrier()?;
        self.stats(
            CollectiveAlgorithm::Direct,
            u32::from(self.world_size() > 1),
            0,
            0,
        )
    }

    pub fn set_timeout(&self, timeout: Duration) -> Result<(), RankError> {
        self.session.set_timeout(timeout)?;
        Ok(())
    }

    pub fn abort(&self, message: &str) -> Result<(), RankError> {
        self.session.abort(message)?;
        Ok(())
    }

    pub fn validate_root(&self, root: u32) -> Result<(), RankError> {
        if root >= self.world_size() {
            return Err(RankError::RankOutOfRange {
                rank: root as usize,
                world_size: self.world_size() as usize,
            });
        }
        Ok(())
    }

    pub fn stats(
        &self,
        algorithm: CollectiveAlgorithm,
        steps: u32,
        transferred_bytes: usize,
        reduction_kernel_launches: u32,
    ) -> Result<CollectiveStats, RankError> {
        Ok(CollectiveStats {
            algorithm,
            transport: self.transport(),
            steps,
            transferred_bytes: u64::try_from(transferred_bytes)
                .map_err(|_| RankError::Overflow("TCP transferred bytes"))?,
            reduction_kernel_launches,
        })
    }
}
