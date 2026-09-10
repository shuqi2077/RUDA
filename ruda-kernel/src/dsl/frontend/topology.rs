//! In this file we use a trick where the constant has the same name as the module containing
//! the expand function, so that a user implicitly imports the expand function when importing the constant.

use ruda_core::ir::{ManagedVariable, Scope};

use crate::dsl::prelude::RudaPrimitive;

use super::NativeExpand;

macro_rules! constant {
    ($ident:ident, $var:expr, $doc:expr) => {
        #[doc = $doc]
        pub const $ident: u32 = 2;

        #[allow(non_snake_case)]
        #[doc = $doc]
        pub mod $ident {
            use super::*;

            /// Expansion of the constant variable.
            pub fn expand(scope: &mut Scope) -> NativeExpand<u32> {
                NativeExpand::new(ManagedVariable::Plain(crate::dsl::ir::Variable::builtin(
                    $var,
                    u32::as_type(scope).storage_type(),
                )))
            }
        }
    };
}

macro_rules! constant_usize {
    ($ident:ident, $var:expr, $doc:expr) => {
        #[doc = $doc]
        pub const $ident: usize = 2;

        #[allow(non_snake_case)]
        #[doc = $doc]
        pub mod $ident {
            use super::*;

            /// Expansion of the constant variable.
            pub fn expand(scope: &mut Scope) -> NativeExpand<usize> {
                NativeExpand::new(ManagedVariable::Plain(crate::dsl::ir::Variable::builtin(
                    $var,
                    usize::as_type(scope).storage_type(),
                )))
            }
        }
    };
}

constant!(
    PLANE_DIM,
    crate::dsl::ir::Builtin::PlaneDim,
    r"
The total amount of working units in a plane.
"
);

constant!(
    PLANE_POS,
    crate::dsl::ir::Builtin::PlanePos,
    r"
The position of the plane within the ruda (plane/warp/subgroup index).
"
);

constant!(
    PLANE_COUNT,
    crate::dsl::ir::Builtin::PlaneCount,
    r"
The number of planes in the current ruda.
"
);

constant!(
    UNIT_POS_PLANE,
    crate::dsl::ir::Builtin::UnitPosPlane,
    r"
The relative position of the working unit inside the plane, without regards to ruda dimensions.
"
);

constant!(
    UNIT_POS,
    crate::dsl::ir::Builtin::UnitPos,
    r"
The position of the working unit inside the ruda, without regards to axis.
"
);

constant!(
    UNIT_POS_X,
    crate::dsl::ir::Builtin::UnitPosX,
    r"
The position of the working unit inside the ruda along the X axis.
"
);

constant!(
    UNIT_POS_Y,
    crate::dsl::ir::Builtin::UnitPosY,
    r"
The position of the working unit inside the ruda along the Y axis.
"
);

constant!(
    UNIT_POS_Z,
    crate::dsl::ir::Builtin::UnitPosZ,
    r"
The position of the working unit inside the ruda along the Z axis.
"
);

constant!(
    RUDA_CLUSTER_DIM,
    crate::dsl::ir::Builtin::ClusterDim,
    r"
The total amount of rudas in a cluster.
"
);

constant!(
    RUDA_CLUSTER_DIM_X,
    crate::dsl::ir::Builtin::ClusterDimX,
    r"
The dimension of the cluster along the X axis.
"
);

constant!(
    RUDA_CLUSTER_DIM_Y,
    crate::dsl::ir::Builtin::ClusterDimY,
    r"
The dimension of the cluster along the Y axis.
"
);

constant!(
    RUDA_CLUSTER_DIM_Z,
    crate::dsl::ir::Builtin::ClusterDimZ,
    r"
The dimension of the cluster along the Z axis.
"
);

constant!(
    RUDA_DIM,
    crate::dsl::ir::Builtin::RudaDim,
    r"
The total amount of working units in a ruda.
"
);

constant!(
    RUDA_DIM_X,
    crate::dsl::ir::Builtin::RudaDimX,
    r"
The dimension of the ruda along the X axis.
"
);

constant!(
    RUDA_DIM_Y,
    crate::dsl::ir::Builtin::RudaDimY,
    r"
The dimension of the ruda along the Y axis.
"
);

constant!(
    RUDA_DIM_Z,
    crate::dsl::ir::Builtin::RudaDimZ,
    r"
The dimension of the ruda along the Z axis.
"
);

constant_usize!(
    RUDA_POS,
    crate::dsl::ir::Builtin::RudaPos,
    r"
The ruda position, without regards to axis.
"
);

constant!(
    RUDA_POS_X,
    crate::dsl::ir::Builtin::RudaPosX,
    r"
The ruda position along the X axis.
"
);

constant!(
    RUDA_POS_Y,
    crate::dsl::ir::Builtin::RudaPosY,
    r"
The ruda position along the Y axis.
"
);

constant!(
    RUDA_POS_Z,
    crate::dsl::ir::Builtin::RudaPosZ,
    r"
The ruda position along the Z axis.
"
);

constant!(
    RUDA_POS_CLUSTER,
    crate::dsl::ir::Builtin::RudaPosCluster,
    r"
The ruda position within the cluster.
"
);

constant!(
    RUDA_POS_CLUSTER_X,
    crate::dsl::ir::Builtin::RudaPosClusterX,
    r"
The ruda position in the cluster along the X axis.
"
);

constant!(
    RUDA_POS_CLUSTER_Y,
    crate::dsl::ir::Builtin::RudaPosClusterY,
    r"
The ruda position in the cluster along the Y axis.
"
);

constant!(
    RUDA_POS_CLUSTER_Z,
    crate::dsl::ir::Builtin::RudaPosClusterZ,
    r"
The ruda position in the cluster along the Z axis.
"
);

constant_usize!(
    RUDA_COUNT,
    crate::dsl::ir::Builtin::RudaCount,
    r"
The number of rudas launched.
"
);

constant!(
    RUDA_COUNT_X,
    crate::dsl::ir::Builtin::RudaCountX,
    r"
The number of rudas launched along the X axis.
"
);

constant!(
    RUDA_COUNT_Y,
    crate::dsl::ir::Builtin::RudaCountY,
    r"
The number of rudas launched along the Y axis.
"
);

constant!(
    RUDA_COUNT_Z,
    crate::dsl::ir::Builtin::RudaCountZ,
    r"
The number of rudas launched along the Z axis.
"
);

constant_usize!(
    ABSOLUTE_POS,
    crate::dsl::ir::Builtin::AbsolutePos,
    r"
The position of the working unit in the whole ruda kernel, without regards to rudas and axis.
"
);

constant!(
    ABSOLUTE_POS_X,
    crate::dsl::ir::Builtin::AbsolutePosX,
    r"
The index of the working unit in the whole ruda kernel along the X axis, without regards to rudas.
"
);

constant!(
    ABSOLUTE_POS_Y,
    crate::dsl::ir::Builtin::AbsolutePosY,
    r"
The index of the working unit in the whole ruda kernel along the Y axis, without regards to rudas.
"
);

constant!(
    ABSOLUTE_POS_Z,
    crate::dsl::ir::Builtin::AbsolutePosZ,
    r"
The index of the working unit in the whole ruda kernel along the Z axis, without regards to rudas.
"
);
