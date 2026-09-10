// SPDX-License-Identifier: Apache-2.0
use core::fmt;

/// Configuration, metadata or explicit parameter-group error for Muon.
/// Device execution faults remain backend errors; no host fallback is attempted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MuonError {
    /// Invalid hyperparameter or unsupported numerical mode.
    InvalidConfig(&'static str),
    /// Muon operates on a complete matrix, not an arbitrary-rank tensor.
    ExpectedMatrix {
        /// Actual tensor rank.
        rank: usize,
    },
    /// A matrix dimension is zero.
    EmptyMatrix,
    /// Gradient or restored momentum has incompatible geometry.
    ShapeMismatch(&'static str),
    /// Gradient or restored momentum has a different element format.
    DTypeMismatch(&'static str),
    /// Gradient or restored momentum is on a different device.
    DeviceMismatch(&'static str),
    /// The Muon group was left empty.
    EmptyMuonGroup,
    /// A selected id was supplied twice.
    DuplicateParameter(u64),
    /// A selected id is absent from the module.
    UnknownParameter(u64),
    /// A selected Muon parameter is frozen.
    FrozenParameter(u64),
    /// A module's ids/shapes changed, or tied aliases disagree.
    ModelChanged,
    /// A gradient refers to an unknown or frozen parameter.
    UnusedGradients,
    /// A record belongs to another grouping, geometry, configuration or schema.
    IncompatibleRecord,
    /// This API is deliberately limited to full, synchronized local matrices.
    UnsupportedDistributed,
}

impl fmt::Display for MuonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid Muon configuration: {message}"),
            Self::ExpectedMatrix { rank } => write!(f, "Newton-Schulz iteration requires 2D tensors, got {rank}D"),
            Self::EmptyMatrix => f.write_str("Muon requires a nonempty matrix"),
            Self::ShapeMismatch(what) => write!(f, "Muon {what} shape does not match its parameter"),
            Self::DTypeMismatch(what) => write!(f, "Muon {what} dtype does not match its parameter"),
            Self::DeviceMismatch(what) => write!(f, "Muon {what} device does not match its parameter"),
            Self::EmptyMuonGroup => f.write_str("select at least one hidden matrix explicitly for Muon"),
            Self::DuplicateParameter(id) => write!(f, "duplicate Muon parameter id {id}"),
            Self::UnknownParameter(id) => write!(f, "unknown Muon parameter id {id}"),
            Self::FrozenParameter(id) => write!(f, "Muon parameter {id} does not require gradients"),
            Self::ModelChanged => f.write_str("model parameter ids/shapes changed or tied aliases disagree"),
            Self::UnusedGradients => f.write_str("gradients include an unknown or frozen parameter"),
            Self::IncompatibleRecord => f.write_str("Muon/AdamW record configuration, grouping, geometry or version mismatch"),
            Self::UnsupportedDistributed => f.write_str("Muon requires complete synchronized matrices; implicit sharded/step_multi updates are not supported"),
        }
    }
}

impl core::error::Error for MuonError {}
