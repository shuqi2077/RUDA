use crate::tiling::ruda_count::{
    RudaCountStrategy, GlobalOrder, hyperruda::builder::HyperrudaBlueprintBuilder,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Determines how to launch the hyperruda, i.e. anything
/// relevant to RudaCount and where a Ruda at a ruda position should work
pub struct HyperrudaBlueprint {
    pub global_order: GlobalOrder,
    pub ruda_count_strategy: RudaCountStrategy,
}

impl HyperrudaBlueprint {
    /// Create a builder for HyperrudaBlueprint
    pub fn builder() -> HyperrudaBlueprintBuilder {
        HyperrudaBlueprintBuilder::new()
    }
}
