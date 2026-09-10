use crate::tiling::ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint};

/// Builder for creating a [HyperrudaBlueprint]
pub struct HyperrudaBlueprintBuilder {
    global_order: Option<GlobalOrder>,
    ruda_count_strategy: Option<RudaCountStrategy>,
}

impl HyperrudaBlueprintBuilder {
    pub(crate) fn new() -> Self {
        Self {
            global_order: None,
            ruda_count_strategy: None,
        }
    }

    /// Set the [GlobalOrder]
    pub fn global_order(mut self, global_order: GlobalOrder) -> Self {
        self.global_order = Some(global_order);
        self
    }

    /// Set the [RudaCountStrategy]
    pub fn ruda_count_strategy(mut self, ruda_count_strategy: RudaCountStrategy) -> Self {
        self.ruda_count_strategy = Some(ruda_count_strategy);
        self
    }

    /// Build the HyperrudaBlueprint
    pub fn build(self) -> HyperrudaBlueprint {
        HyperrudaBlueprint {
            global_order: self.global_order.unwrap_or_default(),
            ruda_count_strategy: self.ruda_count_strategy.unwrap_or_default(),
        }
    }
}
