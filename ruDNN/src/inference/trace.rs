//! Explicit metadata-only operator definitions. Numerical input values, device
//! addresses and kernel bodies are never serialized by this interface.
use super::InferenceError;
use ruda_core::tensor::DType;
use ruda_kernel::{dsl::prelude::Runtime, tensor::RudaTensor};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};
fn is_false(value: &bool) -> bool {
    !*value
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Axis {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Argument {
    pub shape: Option<Vec<String>>,
    pub dtype: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub optional: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Definition {
    pub name: String,
    pub description: String,
    pub op_type: String,
    pub tags: Vec<String>,
    pub axes: BTreeMap<String, Axis>,
    pub inputs: BTreeMap<String, Argument>,
    pub outputs: BTreeMap<String, Argument>,
}
impl Definition {
    pub fn new(name: impl Into<String>, op_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            op_type: op_type.into(),
            tags: vec!["ruda:native".into()],
            axes: BTreeMap::new(),
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }
    }
    /// Dimension symbols describe the public operator contract. Variable axes
    /// omit an instance value; constant axes retain the shape specialization.
    pub fn tensor<R: Runtime>(
        &mut self,
        name: impl Into<String>,
        tensor: &RudaTensor<R>,
        dimensions: &[&str],
        variable: &[&str],
        output: bool,
        optional: bool,
    ) -> Result<(), InferenceError> {
        if dimensions.len() != tensor.meta.num_dims() {
            return Err(InferenceError("trace axis count must match tensor rank"));
        }
        for (axis, &size) in dimensions.iter().zip(tensor.meta.shape().iter()) {
            let varying = variable.contains(axis);
            if let Some(existing) = self.axes.get(*axis) {
                if !varying && existing.value.is_some_and(|n| n != size) {
                    return Err(InferenceError(
                        "trace shared dimension has inconsistent extents",
                    ));
                }
            } else {
                self.axes.insert(
                    (*axis).into(),
                    Axis {
                        kind: if varying { "var" } else { "const" }.into(),
                        value: if varying { None } else { Some(size) },
                    },
                );
            }
        }
        let dtype = match tensor.dtype {
            DType::F32 => "float32",
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::Bool(_) => "bool",
            _ => {
                return Err(InferenceError(
                    "trace tensor dtype requires an explicit raw storage descriptor",
                ));
            }
        };
        let descriptor = Argument {
            shape: Some(dimensions.iter().map(|d| (*d).into()).collect()),
            dtype: dtype.into(),
            optional,
        };
        if output {
            self.outputs.insert(name.into(), descriptor);
        } else {
            self.inputs.insert(name.into(), descriptor);
        }
        Ok(())
    }
    pub fn scalar(
        &mut self,
        name: impl Into<String>,
        dtype: impl Into<String>,
        output: bool,
        optional: bool,
    ) {
        let value = Argument {
            shape: None,
            dtype: dtype.into(),
            optional,
        };
        if output {
            self.outputs.insert(name.into(), value);
        } else {
            self.inputs.insert(name.into(), value);
        }
    }
    pub fn json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let data = self.json().map_err(std::io::Error::other)?;
        std::fs::write(path, data)
    }
}
/// Describe/apply a native operation without a vendor decorator, hidden tensor
/// readback, global environment mutation or benchmark reference implementation.
pub fn trace_apply<T>(
    definition: &Definition,
    emit: impl FnOnce(&Definition),
    operation: impl FnOnce() -> T,
) -> T {
    emit(definition);
    operation()
}
pub fn fi_trace(definition: &Definition) -> Result<String, serde_json::Error> {
    definition.json()
}

/// Typed native equivalent of opt-in Trace Apply kernel substitution. A registry
/// belongs to the application, not process-global Python monkey patching. Each
/// replacement receives the same device arguments as the original operation.
pub struct ApplyRegistry<A, O> {
    replacements: Vec<Replacement<A, O>>,
}
struct Replacement<A, O> {
    name: String,
    operator: String,
    constants: BTreeMap<String, usize>,
    priority: i32,
    function: std::sync::Arc<dyn Fn(&A) -> O + Send + Sync>,
}
impl<A, O> Default for ApplyRegistry<A, O> {
    fn default() -> Self {
        Self {
            replacements: Vec::new(),
        }
    }
}
impl<A, O> ApplyRegistry<A, O> {
    /// A replacement specializes one definition name and its constant axes.
    /// More-specific matches win at equal priority; registration order resolves
    /// the remaining ties. No input values or device addresses are inspected.
    pub fn register(
        &mut self,
        definition: &Definition,
        priority: i32,
        function: impl Fn(&A) -> O + Send + Sync + 'static,
    ) -> Result<(), InferenceError> {
        let mut constants = BTreeMap::new();
        for (name, axis) in &definition.axes {
            match axis.kind.as_str() {
                "var" => {}
                "const" => {
                    constants.insert(
                        name.clone(),
                        axis.value
                            .ok_or(InferenceError("constant trace axis needs an extent"))?,
                    );
                }
                _ => return Err(InferenceError("trace axis must be const or var")),
            }
        }
        self.replacements.push(Replacement {
            name: definition.name.clone(),
            operator: definition.op_type.clone(),
            constants,
            priority,
            function: std::sync::Arc::new(function),
        });
        Ok(())
    }
    pub fn clear(&mut self) {
        self.replacements.clear();
    }
    pub fn len(&self) -> usize {
        self.replacements.len()
    }
    pub fn is_empty(&self) -> bool {
        self.replacements.is_empty()
    }
    pub fn apply(
        &self,
        definition: &Definition,
        arguments: &A,
        emit: impl FnOnce(&Definition),
        original: impl FnOnce(&A) -> O,
    ) -> O {
        emit(definition);
        let mut chosen: Option<&Replacement<A, O>> = None;
        for candidate in &self.replacements {
            if candidate.name != definition.name
                || candidate.operator != definition.op_type
                || candidate.constants.iter().any(|(axis, value)| {
                    definition.axes.get(axis).and_then(|a| a.value) != Some(*value)
                })
            {
                continue;
            }
            if chosen.is_none_or(|old| {
                (candidate.priority, candidate.constants.len())
                    > (old.priority, old.constants.len())
            }) {
                chosen = Some(candidate);
            }
        }
        if let Some(replacement) = chosen {
            (replacement.function)(arguments)
        } else {
            original(arguments)
        }
    }
}
