mod config;

pub use config::AwqQuantizationConfig;

use super::{
    CONFIG_FILE, HuggingFaceLoadError, SINGLE_WEIGHTS_FILE, SafetensorsIndex, WEIGHTS_INDEX_FILE,
    canonical_directory, discover_weight_files, read_json,
};
use rublas::tensor_int4::AwqGemm;
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::transfer::from_data;
use ruda_store::{ModuleStore, SafetensorsStore, TensorSnapshot};
use ruda_tensor::api::{DType, Shape, TensorData};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct CheckpointConfig {
    quantization_config: AwqQuantizationConfig,
}

pub struct AwqCheckpoint {
    directory: PathBuf,
    weight_files: Vec<PathBuf>,
    quantization: AwqQuantizationConfig,
    tensors: BTreeMap<String, TensorSnapshot>,
}

impl AwqCheckpoint {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, HuggingFaceLoadError> {
        let directory = canonical_directory(directory.as_ref())?;
        let config: CheckpointConfig = read_json(&directory.join(CONFIG_FILE))?;
        config.quantization_config.validate()?;
        let weight_files = discover_weight_files(&directory)?;
        let index = if directory.join(SINGLE_WEIGHTS_FILE).is_file() {
            None
        } else {
            Some(read_json::<SafetensorsIndex>(
                &directory.join(WEIGHTS_INDEX_FILE),
            )?)
        };
        let mut tensors = BTreeMap::new();
        for file in &weight_files {
            let mut store = SafetensorsStore::from_file(file);
            let snapshots = store
                .get_all_snapshots()
                .map_err(|error| HuggingFaceLoadError(format!("{}: {error}", file.display())))?;
            for (name, snapshot) in snapshots {
                if let Some(index) = &index {
                    let shard = index.weight_map.get(name).ok_or_else(|| {
                        HuggingFaceLoadError(format!(
                            "tensor {name} is missing from the Safetensors index"
                        ))
                    })?;
                    let indexed_file = directory
                        .join(shard)
                        .canonicalize()
                        .map_err(super::io_error)?;
                    if &indexed_file != file {
                        return Err(HuggingFaceLoadError(format!(
                            "tensor {name} is in a different shard than its index entry"
                        )));
                    }
                }
                if tensors.insert(name.clone(), snapshot.clone()).is_some() {
                    return Err(HuggingFaceLoadError(format!(
                        "duplicate checkpoint tensor: {name}"
                    )));
                }
            }
        }
        if let Some(index) = index {
            for name in index.weight_map.keys() {
                if !tensors.contains_key(name) {
                    return Err(HuggingFaceLoadError(format!(
                        "indexed tensor is missing from its shard: {name}"
                    )));
                }
            }
        }
        Ok(Self {
            directory,
            weight_files,
            quantization: config.quantization_config,
            tensors,
        })
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn weight_files(&self) -> &[PathBuf] {
        &self.weight_files
    }

    pub fn quantization(&self) -> &AwqQuantizationConfig {
        &self.quantization
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor_data(&self, name: &str) -> Result<TensorData, HuggingFaceLoadError> {
        let snapshot = self
            .tensors
            .get(name)
            .ok_or_else(|| HuggingFaceLoadError(format!("missing checkpoint tensor: {name}")))?;
        snapshot
            .to_data()
            .map_err(|error| HuggingFaceLoadError(format!("{name}: {error}")))
    }

    pub fn tensor_info(&self, name: &str) -> Option<(DType, &Shape)> {
        self.tensors.get(name).map(|tensor| (tensor.dtype, &tensor.shape))
    }

    fn validate_tensor(
        &self,
        name: &str,
        shape: Shape,
        dtype: DType,
    ) -> Result<(), HuggingFaceLoadError> {
        let snapshot = self
            .tensors
            .get(name)
            .ok_or_else(|| HuggingFaceLoadError(format!("missing checkpoint tensor: {name}")))?;
        if snapshot.dtype != dtype || snapshot.shape != shape {
            return Err(HuggingFaceLoadError(format!(
                "{name}: expected {dtype:?} {shape:?}, found {:?} {:?}",
                snapshot.dtype, snapshot.shape
            )));
        }
        Ok(())
    }

    pub fn load_linear<R: Runtime>(
        &self,
        prefix: &str,
        input_features: usize,
        output_features: usize,
        has_bias: bool,
        device: &R::Device,
    ) -> Result<AwqGemm<R>, HuggingFaceLoadError> {
        let layout = self.quantization.layout(input_features, output_features)?;
        let qweight = format!("{prefix}.qweight");
        let qzeros = format!("{prefix}.qzeros");
        let scales = format!("{prefix}.scales");
        let bias = format!("{prefix}.bias");
        self.validate_tensor(
            &qweight,
            [input_features, output_features / 8].into(),
            DType::I32,
        )?;
        self.validate_tensor(
            &qzeros,
            [layout.groups(), output_features / 8].into(),
            DType::I32,
        )?;
        self.validate_tensor(
            &scales,
            [layout.groups(), output_features].into(),
            DType::F16,
        )?;
        if has_bias {
            self.validate_tensor(&bias, [output_features].into(), DType::F16)?;
        } else if self.tensors.contains_key(&bias) {
            return Err(HuggingFaceLoadError(format!("unexpected bias: {bias}")));
        }
        for suffix in ["weight", "g_idx"] {
            let name = format!("{prefix}.{suffix}");
            if self.tensors.contains_key(&name) {
                return Err(HuggingFaceLoadError(format!(
                    "unexpected AWQ GEMM tensor: {name}"
                )));
            }
        }
        AwqGemm::new(
            from_data(self.tensor_data(&qweight)?, device),
            from_data(self.tensor_data(&qzeros)?, device),
            from_data(self.tensor_data(&scales)?, device),
            if has_bias {
                Some(from_data(self.tensor_data(&bias)?, device))
            } else {
                None
            },
            layout.group_size,
        )
        .map_err(|error| HuggingFaceLoadError(format!("{prefix}: {error}")))
    }
}

#[cfg(all(test, feature = "nvidia"))]
mod tests;
