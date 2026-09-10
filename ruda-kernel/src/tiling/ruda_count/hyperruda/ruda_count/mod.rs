//! User defines a [RudaCountStrategy], which, once problem is known,
//! becomes a [RudaCountPlan] where all information is known.
//! Then the [RudaCountPlan] is split into:
//! - The RudaCount
//! - The [RudaMapping] which maps a Ruda to where it will work

// mod mapping;
mod plan;
mod strategy;

// pub use mapping::{RudaMapping, RudaMappingLaunch};
pub use plan::{Count3d, RudaCountPlan, RudaCountPlanKind};
pub use strategy::RudaCountStrategy;
