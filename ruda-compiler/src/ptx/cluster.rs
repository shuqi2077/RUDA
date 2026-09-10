use super::{PtxTarget, Result, emit::Emitter, invalid, types::Scalar, unsupported};
use ruda_core::{ir::Builtin, launch::RudaDim};

pub(super) fn directive(dim: Option<RudaDim>, target: PtxTarget) -> Result<String> {
    let Some(dim) = dim else { return Ok(String::new()); };
    if target.sm < 90 || target.version < (7, 8) {
        return Err(unsupported("cluster launch requires SM >= 90 and PTX >= 7.8"));
    }
    if dim.x == 0 || dim.y == 0 || dim.z == 0 {
        return Err(invalid("cluster dimensions must be nonzero"));
    }
    dim.x.checked_mul(dim.y).and_then(|xy| xy.checked_mul(dim.z))
        .ok_or_else(|| invalid("cluster block count exceeds U32"))?;
    Ok(format!(".reqnctapercluster {}, {}, {}\n", dim.x, dim.y, dim.z))
}

impl Emitter {
    pub(super) fn cluster_builtin(&mut self, builtin: Builtin, ty: Scalar) -> Option<String> {
        let (register, singleton) = match builtin {
            Builtin::RudaPosCluster => ("%cluster_ctarank", "0"),
            Builtin::RudaPosClusterX => ("%cluster_ctaid.x", "0"),
            Builtin::RudaPosClusterY => ("%cluster_ctaid.y", "0"),
            Builtin::RudaPosClusterZ => ("%cluster_ctaid.z", "0"),
            Builtin::ClusterDim => ("%cluster_nctarank", "1"),
            Builtin::ClusterDimX => ("%cluster_nctaid.x", "1"),
            Builtin::ClusterDimY => ("%cluster_nctaid.y", "1"),
            Builtin::ClusterDimZ => ("%cluster_nctaid.z", "1"),
            _ => return None,
        };
        let source = if self.target.sm >= 90 && self.target.version >= (7, 8) {
            register
        } else {
            singleton
        };
        Some(self.special(source, ty))
    }
}
