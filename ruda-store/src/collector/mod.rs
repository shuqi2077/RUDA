use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use ruda_tensor::api::{Bool, Int, Tensor, backend::Backend};

use crate::{ModuleAdapter, PathFilter, TensorSnapshot};
use ruda_model::module::{ModuleVisitor, Param, ParamId};

/// Collects tensor views from modules without copying data.
///
/// This collector traverses a module hierarchy and creates lightweight views
/// of tensors that can be materialized to `TensorData` on demand.
///
/// # Examples
///
/// ## Collect all tensors
/// ```rust,no_run
/// # use ruda_store::Collector;
/// let collector = Collector::new(None, None, false);
/// // Use with module.visit(&mut collector);
/// let all_tensors = collector.tensors;
/// ```
///
/// ## Filter with single pattern
/// ```rust,no_run
/// # use ruda_store::{Collector, PathFilter};
/// let filter = PathFilter::new().with_regex(r"^encoder\..*");
/// let collector = Collector::new(Some(filter), None, false);
/// // Use with module.visit(&mut collector);
/// // Only collects tensors starting with "encoder."
/// ```
///
/// ## Filter with multiple patterns (OR union)
/// ```rust,no_run
/// # use ruda_store::{Collector, PathFilter};
/// let filter = PathFilter::new()
///     .with_regex(r"^encoder\..*")  // Match all encoder tensors
///     .with_regex(r".*\.bias$");    // OR match any bias tensors
/// let collector = Collector::new(Some(filter), None, false);
/// // Use with module.visit(&mut collector);
/// // Collects tensors matching ANY of the patterns
/// ```
pub struct Collector {
    /// Collection of tensor views
    pub tensors: Vec<TensorSnapshot>,
    path_stack: Vec<String>,
    container_stack: Vec<String>,
    filter: Option<PathFilter>,
    adapter: Option<Box<dyn ModuleAdapter>>,
    /// Skip enum variant names when building paths
    /// When true, enum variant names are not included in tensor paths
    skip_enum_variants: bool,
}

impl Default for Collector {
    fn default() -> Self {
        Self::new(None, None, false)
    }
}

impl Collector {
    /// Create a new tensor view collector with an optional filter and adapter.
    ///
    /// # Arguments
    ///
    /// * `filter` - An optional [`PathFilter`] to determine which tensors to collect.
    ///   When `None`, all tensors are collected.
    /// * `adapter` - Optional adapter to transform tensors based on container types.
    ///   Applied to all collected tensors before returning.
    /// * `skip_enum_variants` - Skip enum variant names when building paths.
    ///   When true, paths will not include enum variant names (e.g., "feature.weight"
    ///   instead of "feature.BaseConv.weight"). Useful when exporting to formats
    ///   like PyTorch that don't use enum variants.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use ruda_store::{Collector, PathFilter};
    /// // Collect all tensors without adapter
    /// let collector = Collector::new(None, None, false);
    ///
    /// // Use PathFilter builder
    /// let filter = PathFilter::new()
    ///     .with_regex(r"^encoder\..*")
    ///     .with_full_path("decoder.weight");
    /// let collector = Collector::new(Some(filter), None, false);
    ///
    /// // Skip enum variants for PyTorch export
    /// let collector = Collector::new(None, None, true);
    /// ```
    pub fn new(
        filter: Option<PathFilter>,
        adapter: Option<Box<dyn ModuleAdapter>>,
        skip_enum_variants: bool,
    ) -> Self {
        Self {
            tensors: Vec::new(),
            path_stack: Vec::new(),
            container_stack: Vec::new(),
            filter,
            adapter,
            skip_enum_variants,
        }
    }

    /// Apply the adapter to collected tensors and return the result.
    pub fn into_tensors(self) -> Vec<TensorSnapshot> {
        if let Some(adapter) = self.adapter {
            self.tensors
                .into_iter()
                .map(|snapshot| adapter.adapt(&snapshot))
                .collect()
        } else {
            self.tensors
        }
    }

    fn should_collect(&self, path: &[String], container_stack: &[String]) -> bool {
        // If filter is present, use it; otherwise collect all
        match &self.filter {
            None => true,
            Some(f) => f.matches_with_container_path(path, container_stack),
        }
    }
}

impl<B: Backend> ModuleVisitor<B> for Collector {
    fn enter_module(&mut self, name: &str, container_type: &str) {
        // Always track the container type for proper filtering and module type detection
        self.container_stack.push(container_type.to_string());

        // Only add to path if it's not an enum variant (when skip_enum_variants is enabled)
        // This ensures paths are built without enum variant names from the start
        if !self.skip_enum_variants || !container_type.starts_with("Enum:") {
            self.path_stack.push(name.to_string());
        }
    }

    fn exit_module(&mut self, _name: &str, container_type: &str) {
        self.container_stack.pop();

        // Only pop from path if we added it (not an enum variant when skip_enum_variants is enabled)
        if !self.skip_enum_variants || !container_type.starts_with("Enum:") {
            self.path_stack.pop();
        }
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if self.should_collect(&self.path_stack, &self.container_stack) {
            self.tensors.push(TensorSnapshot::from_float(
                &param.transform_for_save().val(),
                self.path_stack.clone(),
                self.container_stack.clone(),
                param.id,
            ));
        }
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
        if self.should_collect(&self.path_stack, &self.container_stack) {
            self.tensors.push(TensorSnapshot::from_int(
                &param.transform_for_save().val(),
                self.path_stack.clone(),
                self.container_stack.clone(),
                param.id,
            ));
        }
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {
        if self.should_collect(&self.path_stack, &self.container_stack) {
            self.tensors.push(TensorSnapshot::from_bool(
                &param.transform_for_save().val(),
                self.path_stack.clone(),
                self.container_stack.clone(),
                param.id,
            ));
        }
    }

    fn visit_float_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        tensor: &Tensor<B, D>,
    ) {
        // For path-based visits, we use the current container stack for filtering
        if self.should_collect(path, &self.container_stack) {
            self.tensors.push(TensorSnapshot::from_float(
                tensor,
                path.to_vec(),
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_int_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        tensor: &Tensor<B, D, Int>,
    ) {
        if self.should_collect(path, &self.container_stack) {
            self.tensors.push(TensorSnapshot::from_int(
                tensor,
                path.to_vec(),
                self.container_stack.clone(),
                id,
            ));
        }
    }

    fn visit_bool_with_path<const D: usize>(
        &mut self,
        path: &[String],
        id: ParamId,
        tensor: &Tensor<B, D, Bool>,
    ) {
        if self.should_collect(path, &self.container_stack) {
            self.tensors.push(TensorSnapshot::from_bool(
                tensor,
                path.to_vec(),
                self.container_stack.clone(),
                id,
            ));
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests;
