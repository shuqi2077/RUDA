use alloc::{string::String, vec::Vec};
use crate::{ir::{Id, Scope, StorageType, Type}, launch::RudaDim};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct KernelDefinition {
    pub buffers: Vec<KernelArg>,
    pub tensor_maps: Vec<KernelArg>,
    pub scalars: Vec<ScalarKernelArg>,
    pub ruda_dim: RudaDim,
    pub body: Scope,
    pub options: KernelOptions,
}

#[derive(Default, Clone, Debug, Hash, PartialEq, Eq)]
/// Options for a specific kernel compilation
pub struct KernelOptions {
    /// The name of the kernel
    pub kernel_name: String,
    /// Whether to include debug symbols
    pub debug_symbols: bool,
    /// CUDA Cluster dim, if any
    pub cluster_dim: Option<RudaDim>,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
/// Global argument of a kernel.
pub struct KernelArg {
    /// The kernel id.
    pub id: Id,
    /// Whether the global argument can only be accessed for reading, or if it can also be accessed
    /// for write.
    pub visibility: Visibility,
    /// The type of the argument.
    pub ty: Type,
    /// The size of the argument.
    pub size: Option<usize>,
    /// Whether the argument has metadata.
    pub has_extended_meta: bool,
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScalarKernelArg {
    pub ty: StorageType,
    pub count: usize,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum Visibility {
    Read,
    ReadWrite,
}
