use crate::cpp::{Dialect, shared::Variable};

use super::BuiltInAttribute;

impl<D: Dialect> Variable<D> {
    pub fn attribute(&self) -> BuiltInAttribute {
        match self {
            Variable::AbsolutePosBaseName => BuiltInAttribute::ThreadPositionInGrid,
            Variable::RudaCountBaseName => BuiltInAttribute::ThreadgroupsPerGrid,
            Variable::RudaDimBaseName => BuiltInAttribute::ThreadsPerThreadgroup,
            Variable::RudaPosBaseName => BuiltInAttribute::ThreadgroupPositionInGrid,
            Variable::PlaneDim => BuiltInAttribute::ThreadsPerSIMDgroup,
            Variable::PlanePos => BuiltInAttribute::SIMDgroupIndexInThreadgroup,
            Variable::PlaneCount => BuiltInAttribute::SIMDgroupsPerThreadgroup,
            Variable::UnitPosBaseName => BuiltInAttribute::ThreadPositionInThreadgroup,
            Variable::UnitPos => BuiltInAttribute::ThreadIndexInThreadgroup,
            Variable::UnitPosPlane => BuiltInAttribute::ThreadIndexInSIMDgroup,
            _ => BuiltInAttribute::None,
        }
    }
}
