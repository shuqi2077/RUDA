use super::*;
use ruda_store::{ModuleStore, TensorSnapshot};
use ruda_tensor::api::{DType, Tensor, backend::Backend};

pub(crate) struct Checkpoint {
    pub files: Vec<PathBuf>,
    pub tensors: BTreeMap<String, TensorSnapshot>,
    pub consumed: BTreeSet<String>,
}

impl Checkpoint {
    pub fn open(directory: &Path) -> Result<Self, HuggingFaceLoadError> {
        let directory = canonical_directory(directory)?;
        let files = discover_weight_files(&directory)?;
        let index = if directory.join(SINGLE_WEIGHTS_FILE).exists() {
            None
        } else {
            Some(read_json::<SafetensorsIndex>(
                &directory.join(WEIGHTS_INDEX_FILE),
            )?)
        };
        let mut tensors = BTreeMap::new();
        for file in &files {
            let mut store = SafetensorsStore::from_file(file);
            for (name, snapshot) in store
                .get_all_snapshots()
                .map_err(|e| HuggingFaceLoadError(format!("{}: {e}", file.display())))?
            {
                if let Some(index) = &index {
                    let shard = index
                        .weight_map
                        .get(name)
                        .ok_or_else(|| HuggingFaceLoadError(format!("unindexed tensor {name}")))?;
                    if directory.join(shard).canonicalize().map_err(io_error)? != *file {
                        return Err(HuggingFaceLoadError(format!("wrong shard for {name}")));
                    }
                }
                if tensors.insert(name.clone(), snapshot.clone()).is_some() {
                    return Err(HuggingFaceLoadError(format!("duplicate tensor {name}")));
                }
            }
        }
        if let Some(index) = index {
            for name in index.weight_map.keys() {
                if !tensors.contains_key(name) {
                    return Err(HuggingFaceLoadError(format!(
                        "missing indexed tensor {name}"
                    )));
                }
            }
        }
        Ok(Self {
            files,
            tensors,
            consumed: BTreeSet::new(),
        })
    }

    pub fn tensor<B: Backend, const D: usize>(
        &mut self,
        name: &str,
        shape: [usize; D],
        device: &B::Device,
    ) -> Result<Tensor<B, D>, HuggingFaceLoadError> {
        let snapshot = self
            .tensors
            .get(name)
            .ok_or_else(|| HuggingFaceLoadError(format!("missing tensor {name}")))?;
        if snapshot.shape != shape.into()
            || !matches!(snapshot.dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return Err(HuggingFaceLoadError(format!(
                "{name}: expected floating {shape:?}, found {:?} {:?}",
                snapshot.dtype, snapshot.shape
            )));
        }
        let data = snapshot
            .to_data()
            .map_err(|e| HuggingFaceLoadError(format!("{name}: {e}")))?;
        self.consumed.insert(name.into());
        Ok(Tensor::from_data(data, (device, snapshot.dtype)))
    }
}
