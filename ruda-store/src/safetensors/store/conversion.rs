use super::*;

/// Convert TensorSnapshots to safetensors format lazily.
pub(super) fn snapshots_to_safetensors(
    snapshots: Vec<TensorSnapshot>,
) -> Result<Vec<(String, TensorSnapshotAdapter)>, SafetensorsStoreError> {
    let mut tensors = Vec::new();

    for snapshot in snapshots {
        let name = snapshot.full_path();
        // No need to materialize data - TensorSnapshot now has dtype and shape cached!
        tensors.push((name, TensorSnapshotAdapter(snapshot)));
    }

    Ok(tensors)
}

/// Convert safetensors to TensorSnapshots with lazy loading.
pub(super) fn safetensors_to_snapshots_lazy(
    data_arc: Arc<Vec<u8>>,
) -> Result<Vec<TensorSnapshot>, SafetensorsStoreError> {
    // Parse to get metadata
    let tensors = safetensors::SafeTensors::deserialize(&data_arc)?;
    let mut snapshots = Vec::new();

    for (name, tensor_snapshot) in tensors.tensors() {
        // Extract metadata without materializing data
        let dtype = safetensor_dtype_to_ruda(tensor_snapshot.dtype())?;
        let shape = tensor_snapshot.shape();
        let path_parts: Vec<String> = name.split('.').map(|s| s.to_string()).collect();

        // Create a lazy closure that will deserialize only this tensor when needed
        #[cfg(target_has_atomic = "ptr")]
        let data_clone = Arc::clone(&data_arc);
        #[cfg(not(target_has_atomic = "ptr"))]
        let data_clone = data_arc.clone();
        let name_clone = name.to_string();
        let data_fn = alloc::rc::Rc::new(move || {
            // Re-deserialize when needed (this is cheap, just parsing header)
            let tensors = safetensors::SafeTensors::deserialize(&data_clone).map_err(|e| {
                crate::TensorSnapshotError::IoError(format!(
                    "Failed to re-deserialize safetensors: {}",
                    e
                ))
            })?;

            // Find our specific tensor
            let tensor = tensors.tensor(&name_clone).map_err(|e| {
                crate::TensorSnapshotError::DataError(format!(
                    "Tensor '{}' not found: {}",
                    name_clone, e
                ))
            })?;

            // Now materialize just this tensor's data
            let bytes = ruda_tensor::api::Bytes::from_bytes_vec(tensor.data().to_vec());
            Ok(TensorData {
                bytes,
                shape: tensor.shape().into(),
                dtype: safetensor_dtype_to_ruda(tensor.dtype())
                    .map_err(|_| crate::TensorSnapshotError::DataError("Invalid dtype".into()))?,
            })
        });

        let snapshot = TensorSnapshot::from_closure(
            data_fn,
            dtype,
            shape.into(),
            path_parts,
            vec![], // Empty container_stack - will be filled during module traversal
            ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    Ok(snapshots)
}

/// Convert safetensors to TensorSnapshots with true on-demand loading from file.
/// This reads only the header initially, then loads tensor data on demand.
#[cfg(feature = "std")]
pub(super) fn safetensors_to_snapshots_lazy_file(
    path: &std::path::Path,
) -> Result<Vec<TensorSnapshot>, SafetensorsStoreError> {
    // Always use memory mapping for the most efficient access
    use memmap2::MmapOptions;

    // Memory map the file for efficient access
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let mmap_arc = Arc::new(mmap);

    // Parse just to get metadata (safetensors won't copy data with mmap)
    let tensors = safetensors::SafeTensors::deserialize(&mmap_arc)?;
    let mut snapshots = Vec::new();

    for (name, tensor_snapshot) in tensors.tensors() {
        let dtype = safetensor_dtype_to_ruda(tensor_snapshot.dtype())?;
        let shape = tensor_snapshot.shape();
        let path_parts: Vec<String> = name.split('.').map(|s| s.to_string()).collect();

        // Create a lazy closure that accesses the mmap'd data
        let mmap_clone = Arc::clone(&mmap_arc);
        let name_clone = name.to_string();

        let data_fn = alloc::rc::Rc::new(move || {
            // Re-parse to get the tensor snapshot (this is cheap with mmap)
            let tensors = safetensors::SafeTensors::deserialize(&mmap_clone).map_err(|e| {
                crate::TensorSnapshotError::IoError(format!("Failed to deserialize: {}", e))
            })?;
            let tensor = tensors.tensor(&name_clone).map_err(|e| {
                crate::TensorSnapshotError::DataError(format!(
                    "Tensor '{}' not found: {}",
                    name_clone, e
                ))
            })?;

            // Only now do we actually copy the tensor data
            Ok(TensorData {
                bytes: ruda_tensor::api::Bytes::from_bytes_vec(tensor.data().to_vec()),
                shape: tensor.shape().into(),
                dtype: safetensor_dtype_to_ruda(tensor.dtype())
                    .map_err(|_| crate::TensorSnapshotError::DataError("Invalid dtype".into()))?,
            })
        });

        let snapshot = TensorSnapshot::from_closure(
            data_fn,
            dtype,
            shape.into(),
            path_parts,
            vec![], // Empty container_stack - will be filled during module traversal
            ParamId::new(),
        );
        snapshots.push(snapshot);
    }

    Ok(snapshots)
}

/// Helper to convert safetensors Dtype to ruda DType.
fn safetensor_dtype_to_ruda(dtype: safetensors::Dtype) -> Result<DType, SafetensorsStoreError> {
    use safetensors::Dtype;

    match dtype {
        Dtype::F64 => Ok(DType::F64),
        Dtype::F32 => Ok(DType::F32),
        Dtype::F16 => Ok(DType::F16),
        Dtype::BF16 => Ok(DType::BF16),
        Dtype::I64 => Ok(DType::I64),
        Dtype::I32 => Ok(DType::I32),
        Dtype::I16 => Ok(DType::I16),
        Dtype::I8 => Ok(DType::I8),
        Dtype::U64 => Ok(DType::U64),
        Dtype::U32 => Ok(DType::U32),
        Dtype::U8 => Ok(DType::U8),
        Dtype::BOOL => Ok(DType::Bool(BoolStore::Native)),
        _ => Err(SafetensorsStoreError::Other(format!(
            "Unsupported dtype: {:?}",
            dtype
        ))),
    }
}

/// Helper to convert DType to safetensors Dtype.
pub(super) fn dtype_to_safetensors(dtype: DType) -> Result<safetensors::Dtype, SafetensorsStoreError> {
    use safetensors::Dtype;

    match dtype {
        DType::F64 => Ok(Dtype::F64),
        DType::F32 | DType::Flex32 => Ok(Dtype::F32), // Flex32 is stored as F32
        DType::F16 => Ok(Dtype::F16),
        DType::BF16 => Ok(Dtype::BF16),
        DType::I64 => Ok(Dtype::I64),
        DType::I32 => Ok(Dtype::I32),
        DType::I16 => Ok(Dtype::I16),
        DType::I8 => Ok(Dtype::I8),
        DType::U64 => Ok(Dtype::U64),
        DType::U32 => Ok(Dtype::U32),
        DType::U16 => Err(SafetensorsStoreError::Other(
            "U16 dtype not yet supported in safetensors".to_string(),
        )),
        DType::U8 => Ok(Dtype::U8),
        DType::Bool(BoolStore::Native) => Ok(Dtype::BOOL),
        DType::Bool(BoolStore::U32) => Ok(Dtype::U32),
        DType::Bool(BoolStore::U8) => Ok(Dtype::U8),
        DType::QFloat(_) => Err(SafetensorsStoreError::Other(
            "Quantized tensors not yet supported in safetensors".to_string(),
        )),
    }
}
