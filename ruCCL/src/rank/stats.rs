use super::{CollectiveAlgorithm, CollectiveTransport};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectiveStats {
    pub algorithm: CollectiveAlgorithm,
    pub transport: CollectiveTransport,
    pub steps: u32,
    pub transferred_bytes: u64,
    pub reduction_kernel_launches: u32,
}

impl CollectiveStats {
    pub fn new(algorithm: CollectiveAlgorithm) -> Self {
        Self {
            algorithm,
            transport: CollectiveTransport::HostStaged,
            steps: 0,
            transferred_bytes: 0,
            reduction_kernel_launches: 0,
        }
    }
}
