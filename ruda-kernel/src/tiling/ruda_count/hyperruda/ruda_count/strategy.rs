use crate::tiling::ruda_count::SmAllocation;

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Front-facing configuration when crafting a TilingBlueprint
/// Allows choosing a strategy before knowing actual values
pub enum RudaCountStrategy {
    #[default]
    /// X: num rudas in m, Y: num rudas in n, Z: num rudas in batch
    FromProblem,

    /// If not rudas_first: X: num SMs, Y: num rudas per SM
    /// If rudas_first: X: num rudas per SM, Y: num SMs
    Sm {
        rudas_first: bool,
        num_sms: u32,
        sm_usage: SmAllocation,
    },

    /// X: total rudas flattened (num SMs * num rudas per SM)
    Flattened,

    /// Heuristically find a balance for X, Y, Z that respects hardware limits
    Spread,
}
