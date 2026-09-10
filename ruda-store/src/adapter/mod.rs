//! Module adapters for transforming tensor snapshots during save/load
//!
//! This module provides adapters for:
//! - PyTorch/Ruda format conversion (weight transposition, parameter renaming)
//! - Mixed-precision storage (F32/F16 dtype casting via [`HalfPrecisionAdapter`])
//! - Adapter chaining for composing multiple transformations

use crate::TensorSnapshot;

use alloc::boxed::Box;
use alloc::format;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec;

use ruda_tensor::api::shape;
use ruda_tensor::api::{DType, TensorData};
use hashbrown::HashSet;

// Module type names as they appear in the container_type field
// These come from the Module derive macro which uses stringify! on the struct name
// Format: "Struct:TypeName" for user-defined structs
mod module_names {
    // The actual string constants that match what the Module derive macro produces
    pub const LINEAR: &str = "Struct:Linear";
    pub const BATCH_NORM: &str = "Struct:BatchNorm";
    pub const LAYER_NORM: &str = "Struct:LayerNorm";
    pub const GROUP_NORM: &str = "Struct:GroupNorm";
    pub const EMBEDDING: &str = "Struct:Embedding";
    pub const CONV1D: &str = "Struct:Conv1d";
    pub const CONV2D: &str = "Struct:Conv2d";
    pub const CONV3D: &str = "Struct:Conv3d";
    pub const CONV_TRANSPOSE1D: &str = "Struct:ConvTranspose1d";
    pub const CONV_TRANSPOSE2D: &str = "Struct:ConvTranspose2d";
    pub const CONV_TRANSPOSE3D: &str = "Struct:ConvTranspose3d";
    pub const DEFORM_CONV2D: &str = "Struct:DeformConv2d";
    pub const INSTANCE_NORM: &str = "Struct:InstanceNorm";
    pub const RMS_NORM: &str = "Struct:RmsNorm";
    pub const PRELU: &str = "Struct:PRelu";
}

/// Trait for adapting tensor snapshots between different module formats
pub trait ModuleAdapter: Send + Sync {
    /// Adapt a tensor snapshot based on its container type and parameter name
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot;

    /// Get alternative parameter name to try during matching
    ///
    /// When looking for a parameter in a module, this method provides an alternative
    /// name to try if the direct name doesn't match. This enables matching parameters
    /// with different naming conventions (e.g., PyTorch's "weight" vs Ruda's "gamma").
    ///
    /// # Arguments
    /// * `param_name` - The parameter name we're looking for
    /// * `container_type` - The type of container module (e.g., "BatchNorm")
    ///
    /// # Returns
    /// Alternative parameter name to try, or None if no alternative exists
    fn get_alternative_param_name(
        &self,
        _param_name: &str,
        _container_type: &str,
    ) -> Option<String> {
        None
    }

    /// Clone the adapter into a boxed trait object
    fn clone_box(&self) -> Box<dyn ModuleAdapter>;

    /// Chain adapters together, applying `self` first and then `next`.
    ///
    /// This is useful when multiple transformations are required when importing model weights
    /// (e.g. PyTorch -> Ruda layout conversion, then dtype casting, then custom remapping).
    ///
    /// The semantics follow a simple pipeline:
    /// - `adapt`: `next.adapt(&self.adapt(snapshot))`
    /// - `get_alternative_param_name`: try `self` first; if it returns an alternative name,
    ///   try `next` with that name, otherwise return the first alternative name.
    fn chain<A>(self, next: A) -> ChainAdapter
    where
        Self: Sized + 'static,
        A: ModuleAdapter + 'static,
    {
        ChainAdapter::new(self, next)
    }
}

impl Clone for Box<dyn ModuleAdapter> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Adapter that applies two adapters in sequence.
///
/// This allows composing smaller adapters instead of creating one large monolithic adapter.
#[derive(Clone)]
pub struct ChainAdapter {
    first: Box<dyn ModuleAdapter>,
    second: Box<dyn ModuleAdapter>,
}

impl ChainAdapter {
    /// Create a new adapter chain.
    pub fn new<A, B>(first: A, second: B) -> Self
    where
        A: ModuleAdapter + 'static,
        B: ModuleAdapter + 'static,
    {
        Self {
            first: Box::new(first),
            second: Box::new(second),
        }
    }
}

impl ModuleAdapter for ChainAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let snapshot = self.first.adapt(snapshot);
        self.second.adapt(&snapshot)
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        if let Some(name) = self
            .first
            .get_alternative_param_name(param_name, container_type)
        {
            self.second
                .get_alternative_param_name(&name, container_type)
                .or(Some(name))
        } else {
            self.second
                .get_alternative_param_name(param_name, container_type)
        }
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Identity adapter that passes tensors through unchanged
#[derive(Debug, Clone, Default)]
pub struct IdentityAdapter;

impl ModuleAdapter for IdentityAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        snapshot.clone()
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Returns the default set of module types that `HalfPrecisionAdapter` converts.
///
/// Includes: Linear, Embedding, all Conv variants, LayerNorm, GroupNorm,
/// InstanceNorm, RmsNorm, PRelu.
///
/// Excludes BatchNorm by default because `running_var` underflows in F16.
fn default_half_precision_modules() -> HashSet<String> {
    let modules = [
        module_names::LINEAR,
        module_names::EMBEDDING,
        module_names::CONV1D,
        module_names::CONV2D,
        module_names::CONV3D,
        module_names::CONV_TRANSPOSE1D,
        module_names::CONV_TRANSPOSE2D,
        module_names::CONV_TRANSPOSE3D,
        module_names::DEFORM_CONV2D,
        module_names::LAYER_NORM,
        module_names::GROUP_NORM,
        module_names::INSTANCE_NORM,
        module_names::RMS_NORM,
        module_names::PRELU,
    ];
    modules.iter().map(|s| s.to_string()).collect()
}

/// Adapter for mixed-precision (F32/F16) model storage.
///
/// Auto-detects conversion direction from the snapshot's dtype:
/// - F32 source -> cast to F16 (typical for saving)
/// - F16 source -> cast to F32 (typical for loading)
/// - Other dtypes -> passed through unchanged
///
/// The same instance works for both `with_to_adapter` (save) and `with_from_adapter` (load).
///
/// By default, converts weights in: Linear, Embedding, Conv*, LayerNorm, GroupNorm,
/// InstanceNorm, RmsNorm, PRelu. BatchNorm is excluded because `running_var` underflows in F16.
///
/// # Examples
///
/// Default usage (same adapter for save and load):
/// ```rust
/// # use ruda_store::HalfPrecisionAdapter;
/// let adapter = HalfPrecisionAdapter::new();
/// // store.with_to_adapter(adapter.clone());  // F32 -> F16 on save
/// // store.with_from_adapter(adapter);        // F16 -> F32 on load
/// ```
///
/// Exclude a module type:
/// ```rust
/// # use ruda_store::HalfPrecisionAdapter;
/// let adapter = HalfPrecisionAdapter::new()
///     .without_module("LayerNorm");
/// ```
///
/// Add a custom module type:
/// ```rust
/// # use ruda_store::HalfPrecisionAdapter;
/// let adapter = HalfPrecisionAdapter::new()
///     .with_module("CustomLayer");
/// ```
#[derive(Debug, Clone)]
pub struct HalfPrecisionAdapter {
    modules: HashSet<String>,
}

impl HalfPrecisionAdapter {
    /// Create a new adapter with the default set of modules.
    pub fn new() -> Self {
        Self {
            modules: default_half_precision_modules(),
        }
    }

    /// Add a module type to convert. Accepts both short (`"MyLayer"`) and
    /// qualified (`"Struct:MyLayer"`) forms.
    ///
    /// Note: short names are mapped to `"Struct:Name"`. If you have an Enum-based
    /// module, use the qualified form `"Enum:MyModule"` explicitly.
    pub fn with_module(mut self, module_type: impl Into<String>) -> Self {
        let name = module_type.into();
        if name.contains(':') {
            self.modules.insert(name);
        } else {
            self.modules.insert(format!("Struct:{}", name));
        }
        self
    }

    /// Remove a module type from conversion. Accepts both short and qualified forms.
    pub fn without_module(mut self, module_type: impl Into<String>) -> Self {
        let name = module_type.into();
        let key = if name.contains(':') {
            name
        } else {
            format!("Struct:{}", name)
        };
        assert!(
            self.modules.contains(&key),
            "without_module called with '{}' which is not in the module set",
            key
        );
        self.modules.remove(&key);
        self
    }

    /// Check whether the tensor belongs to a module that should be converted.
    fn should_convert(&self, snapshot: &TensorSnapshot) -> bool {
        snapshot
            .module_type()
            .is_some_and(|mt| self.modules.contains(&mt))
    }
}

impl Default for HalfPrecisionAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleAdapter for HalfPrecisionAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        // Determine target dtype from source: F32 -> F16, F16 -> F32, anything else -> skip
        let target_dtype = match snapshot.dtype {
            DType::F32 => DType::F16,
            DType::F16 => DType::F32,
            _ => return snapshot.clone(),
        };

        if !self.should_convert(snapshot) {
            return snapshot.clone();
        }

        let original_data_fn = snapshot.clone_data_fn();

        let cast_data_fn = Rc::new(move || {
            let data = original_data_fn()?;
            Ok(data.convert_dtype(target_dtype))
        });

        TensorSnapshot::from_closure(
            cast_data_fn,
            target_dtype,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Adapter for converting from PyTorch format to Ruda format
///
/// Handles:
/// - Linear layer weight transposition (PyTorch: [out, in] → Ruda: [in, out])
/// - Normalization parameter renaming (weight → gamma, bias → beta)
#[derive(Debug, Clone, Default)]
pub struct PyTorchToRudaAdapter;

impl ModuleAdapter for PyTorchToRudaAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        adapt_pytorch_tensor(snapshot, PyTorchConversionDirection::PyTorchToRuda)
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        // For PyTorch->Ruda: When looking for Ruda names (gamma/beta), try PyTorch names (weight/bias)
        if is_normalization_layer(container_type) {
            ruda_norm_param_to_pytorch(param_name).map(|s| s.to_string())
        } else {
            None
        }
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Adapter for converting from Ruda format to PyTorch format
///
/// Handles:
/// - Linear layer weight transposition (Ruda: [in, out] → PyTorch: [out, in])
/// - Normalization parameter renaming (gamma → weight, beta → bias)
#[derive(Debug, Clone, Default)]
pub struct RudaToPyTorchAdapter;

impl ModuleAdapter for RudaToPyTorchAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        adapt_pytorch_tensor(snapshot, PyTorchConversionDirection::RudaToPyTorch)
    }

    fn get_alternative_param_name(&self, param_name: &str, container_type: &str) -> Option<String> {
        // For Ruda->PyTorch: When looking for PyTorch names (weight/bias), try Ruda names (gamma/beta)
        if is_normalization_layer(container_type) {
            pytorch_norm_param_to_ruda(param_name).map(|s| s.to_string())
        } else {
            None
        }
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

/// Direction of PyTorch conversion for parameter naming
#[derive(Debug, Clone, Copy)]
enum PyTorchConversionDirection {
    PyTorchToRuda,
    RudaToPyTorch,
}

/// Check if container type is a normalization layer
fn is_normalization_layer(container_type: &str) -> bool {
    matches!(
        container_type,
        module_names::BATCH_NORM | module_names::LAYER_NORM | module_names::GROUP_NORM
    )
}

/// Map PyTorch normalization parameter name to Ruda
fn pytorch_norm_param_to_ruda(param_name: &str) -> Option<&'static str> {
    match param_name {
        "weight" => Some("gamma"),
        "bias" => Some("beta"),
        _ => None,
    }
}

/// Map Ruda normalization parameter name to PyTorch
fn ruda_norm_param_to_pytorch(param_name: &str) -> Option<&'static str> {
    match param_name {
        "gamma" => Some("weight"),
        "beta" => Some("bias"),
        _ => None,
    }
}

/// Core tensor adaptation logic for PyTorch format conversions
fn adapt_pytorch_tensor(
    snapshot: &TensorSnapshot,
    direction: PyTorchConversionDirection,
) -> TensorSnapshot {
    // Extract path and parameter name
    let (path_stack, param_name) = match get_path_and_param(snapshot) {
        Some(result) => result,
        None => return snapshot.clone(),
    };

    // Get module type for matching (ignores Vec/Array wrappers)
    let module_type = match snapshot.module_type() {
        Some(mt) => mt,
        None => return snapshot.clone(), // No user-defined module found
    };

    // Linear: transpose weight (bidirectional - same operation both ways)
    if module_type == module_names::LINEAR && param_name == "weight" && snapshot.shape.len() == 2 {
        return transpose_2d_tensor(snapshot);
    }

    // Normalization layers: rename parameters based on direction
    if is_normalization_layer(&module_type) {
        let new_name = match direction {
            PyTorchConversionDirection::PyTorchToRuda => pytorch_norm_param_to_ruda(param_name),
            PyTorchConversionDirection::RudaToPyTorch => ruda_norm_param_to_pytorch(param_name),
        };

        if let Some(new_name) = new_name {
            return rename_parameter(snapshot, path_stack, new_name);
        }
    }

    snapshot.clone()
}

/// Extract path stack and parameter name from snapshot
fn get_path_and_param(snapshot: &TensorSnapshot) -> Option<(&[String], &str)> {
    let path_stack = snapshot.path_stack.as_ref()?;
    let param_name = path_stack.last()?.as_str();
    Some((path_stack.as_slice(), param_name))
}

/// Rename a parameter in the snapshot
fn rename_parameter(
    snapshot: &TensorSnapshot,
    path_stack: &[String],
    new_name: &str,
) -> TensorSnapshot {
    let mut new_path = path_stack.to_vec();
    *new_path.last_mut().unwrap() = new_name.to_string();

    TensorSnapshot::from_closure(
        snapshot.clone_data_fn(),
        snapshot.dtype,
        snapshot.shape.clone(),
        new_path,
        snapshot.container_stack.clone().unwrap_or_default(),
        snapshot.tensor_id.unwrap_or_default(),
    )
}

/// Transpose a 2D tensor
fn transpose_2d_tensor(snapshot: &TensorSnapshot) -> TensorSnapshot {
    if snapshot.shape.len() != 2 {
        return snapshot.clone();
    }

    let original_data_fn = snapshot.clone_data_fn();
    let dtype = snapshot.dtype;
    let transposed_shape = shape![snapshot.shape[1], snapshot.shape[0]];

    // Create a lazy closure that transposes when called
    let transposed_data_fn = Rc::new(move || {
        let data = original_data_fn()?;
        Ok(transpose_tensor_data(data))
    });

    TensorSnapshot::from_closure(
        transposed_data_fn,
        dtype,
        transposed_shape,
        snapshot.path_stack.clone().unwrap_or_default(),
        snapshot.container_stack.clone().unwrap_or_default(),
        snapshot.tensor_id.unwrap_or_default(),
    )
}

/// Transpose tensor data (assumes 2D shape is already validated)
fn transpose_tensor_data(data: TensorData) -> TensorData {
    let shape = &data.shape;
    let rows = shape[0];
    let cols = shape[1];
    let transposed_shape = vec![cols, rows];

    // Get the raw bytes and element size
    let bytes = data.as_bytes();
    let element_size = data.dtype.size();

    // Create a new buffer for transposed data
    let mut transposed_bytes = vec![0u8; bytes.len()];

    // Transpose at the byte level - works for any data type
    for i in 0..rows {
        for j in 0..cols {
            let src_idx = (i * cols + j) * element_size;
            let dst_idx = (j * rows + i) * element_size;

            // Copy the bytes for this element
            transposed_bytes[dst_idx..dst_idx + element_size]
                .copy_from_slice(&bytes[src_idx..src_idx + element_size]);
        }
    }

    // Create new TensorData from transposed bytes
    TensorData::from_bytes_vec(transposed_bytes, transposed_shape, data.dtype)
}

#[cfg(test)]
mod tests;
