use super::*;

pub(super) fn rebuild_from_type_v2(
    o: Object,
    memo: &mut HashMap<u32, Object>,
    data_source: &Option<Arc<LazyDataSource>>,
) -> Result<Object> {
    let args = if let Object::Tuple(args) = o {
        if args.is_empty() {
            return Err(PickleError::InvalidData(
                "rebuild_from_type_v2: empty args".to_string(),
            ));
        }
        args
    } else {
        return Err(PickleError::InvalidData(format!(
            "rebuild_from_type_v2: expected tuple got {:?}",
            o
        )));
    };
    let func = &args[0];
    match func {
        Object::Class { module_name, name } => {
            let module_name = module_name.as_str();
            let name = name.as_str();
            // For rebuild_tensor_v2, the args might already be in a tuple
            let actual_args = if args.len() == 2 && matches!(&args[1], Object::Tuple(_)) {
                // If there's only one arg and it's a tuple, use it directly
                args[1].clone()
            } else {
                // Otherwise, wrap the remaining args in a tuple
                Object::Tuple(args[1..].to_vec())
            };
            if module_name == "torch._utils" && name == "_rebuild_tensor_v2" {
                rebuild_tensor_v2(actual_args, memo, data_source)
            } else if module_name == "torch._utils" && name == "_rebuild_tensor" {
                // Legacy _rebuild_tensor (PyTorch < 1.6)
                // Same as v2 but with fewer arguments: (storage, storage_offset, size, stride)
                rebuild_tensor(actual_args, memo, data_source)
            } else if module_name == "torch._tensor" && name == "_rebuild_from_type_v2" {
                rebuild_from_type_v2(actual_args, memo, data_source)
            } else if module_name == "torch._utils" && name == "_rebuild_parameter" {
                rebuild_parameter(actual_args, memo, data_source)
            } else if module_name == "collections" && name == "OrderedDict" {
                // OrderedDict is treated as a regular Dict in our implementation
                Ok(Object::Dict(HashMap::new()))
            } else {
                Err(PickleError::UnsupportedType(format!(
                    "{}.{}",
                    module_name, name
                )))
            }
        }
        _ => Err(PickleError::InvalidData(format!(
            "rebuild_from_type_v2: expected class got {:?}",
            func
        ))),
    }
}

fn rebuild_parameter(
    args: Object,
    memo: &mut HashMap<u32, Object>,
    data_source: &Option<Arc<LazyDataSource>>,
) -> Result<Object> {
    let args = if let Object::Tuple(args) = args {
        if args.is_empty() {
            return Err(PickleError::InvalidData(
                "rebuild_parameter: empty args".to_string(),
            ));
        }
        args
    } else {
        return Err(PickleError::InvalidData(format!(
            "rebuild_parameter: expected tuple got {:?}",
            args
        )));
    };
    let data = &args[0];
    let tensor = match data {
        Object::Reduce {
            callable: _,
            args: _,
        } => rebuild_from_type_v2(data.clone(), memo, data_source)?,
        _ => data.clone(),
    };
    Ok(tensor)
}

/// Parse storage argument and extract storage info and tuple.
fn parse_storage_arg(arg: &Object, fn_name: &str) -> Result<(Vec<u8>, Option<Vec<Object>>)> {
    match arg {
        Object::Persistent(data) => Ok((data.clone(), None)),
        Object::PersistentTuple(tuple) => Ok((vec![], Some(tuple.clone()))),
        // Also accept regular Tuple for TAR format compatibility
        Object::Tuple(tuple) => Ok((vec![], Some(tuple.clone()))),
        _ => Err(PickleError::InvalidData(format!(
            "{}: expected persistent id got {:?}",
            fn_name, arg
        ))),
    }
}

/// Parse shape argument.
fn parse_shape_arg(arg: &Object, fn_name: &str) -> Result<Vec<usize>> {
    match arg {
        Object::Tuple(shape) => shape
            .iter()
            .map(|x| match x {
                Object::Int(i) => Ok(*i as usize),
                _ => Err(PickleError::InvalidData(
                    "shape must contain ints".to_string(),
                )),
            })
            .collect::<Result<Vec<_>>>(),
        _ => Err(PickleError::InvalidData(format!(
            "{}: expected shape tuple got {:?}",
            fn_name, arg
        ))),
    }
}

/// Legacy _rebuild_tensor function for PyTorch < 1.6.
/// Thin wrapper that parses 4 arguments and calls rebuild_tensor_impl.
fn rebuild_tensor(
    args: Object,
    _memo: &mut HashMap<u32, Object>,
    data_source: &Option<Arc<LazyDataSource>>,
) -> Result<Object> {
    let args = if let Object::Tuple(args) = args {
        args
    } else {
        return Err(PickleError::InvalidData(format!(
            "rebuild_tensor: expected tuple got {:?}",
            args
        )));
    };

    if args.len() < 4 {
        return Err(PickleError::InvalidData(format!(
            "rebuild_tensor: expected at least 4 args, got {}",
            args.len()
        )));
    }

    let (storage_info, storage_tuple) = parse_storage_arg(&args[0], "rebuild_tensor")?;
    let storage_offset = match &args[1] {
        Object::Int(offset) => *offset as usize,
        _ => 0,
    };
    let shape = parse_shape_arg(&args[2], "rebuild_tensor")?;

    rebuild_tensor_impl(
        storage_info,
        storage_tuple,
        storage_offset,
        shape,
        data_source,
    )
}

/// Modern _rebuild_tensor_v2 function for PyTorch >= 1.6.
/// Thin wrapper that parses 5+ arguments and calls rebuild_tensor_impl.
fn rebuild_tensor_v2(
    args: Object,
    _memo: &mut HashMap<u32, Object>,
    data_source: &Option<Arc<LazyDataSource>>,
) -> Result<Object> {
    let args = if let Object::Tuple(args) = args {
        args
    } else {
        return Err(PickleError::InvalidData(format!(
            "rebuild_tensor_v2: expected tuple got {:?}",
            args
        )));
    };

    if args.len() < 5 {
        return Err(PickleError::InvalidData(format!(
            "rebuild_tensor_v2: expected at least 5 args, got {}",
            args.len()
        )));
    }

    let (storage_info, storage_tuple) = parse_storage_arg(&args[0], "rebuild_tensor_v2")?;
    let storage_offset = match &args[1] {
        Object::Int(offset) => *offset as usize,
        _ => 0,
    };
    let shape = parse_shape_arg(&args[2], "rebuild_tensor_v2")?;
    // args[3] is stride (unused)
    // args[4] is requires_grad (unused)
    // args[5] is backward_hooks (unused)

    rebuild_tensor_impl(
        storage_info,
        storage_tuple,
        storage_offset,
        shape,
        data_source,
    )
}

/// Helper to convert storage type name to DType.
fn storage_type_to_dtype(storage_type: &str) -> Result<DType> {
    match storage_type {
        "FloatStorage" => Ok(DType::F32),
        "DoubleStorage" => Ok(DType::F64),
        "HalfStorage" => Ok(DType::F16),
        "BFloat16Storage" => Ok(DType::BF16),
        "LongStorage" => Ok(DType::I64),
        "IntStorage" => Ok(DType::I32),
        "ShortStorage" => Ok(DType::I16),
        "CharStorage" => Ok(DType::I8),
        "ByteStorage" => Ok(DType::U8),
        "BoolStorage" => Ok(DType::Bool(BoolStore::Native)),
        _ => Err(PickleError::InvalidData(format!(
            "Unknown storage type: {}",
            storage_type
        ))),
    }
}

/// Core implementation for rebuilding tensors.
/// Shared by both rebuild_tensor (legacy) and rebuild_tensor_v2 (modern).
fn rebuild_tensor_impl(
    storage_info: Vec<u8>,
    storage_tuple: Option<Vec<Object>>,
    storage_offset: usize,
    shape: Vec<usize>,
    data_source: &Option<Arc<LazyDataSource>>,
) -> Result<Object> {
    // Parse the storage info to extract dtype and storage key
    // The persistent ID is typically a tuple like: ('storage', 'FloatStorage', '0', 'cpu', 4)
    let (dtype, storage_key, storage_total_elements) = if let Some(tuple) = storage_tuple {
        // Direct tuple access
        if tuple.len() >= 3 {
            let storage_type = match &tuple[1] {
                Object::String(s) => s.as_str(),
                Object::Class {
                    module_name: _,
                    name,
                } => name.as_str(),
                other => {
                    return Err(PickleError::InvalidData(format!(
                        "Expected storage type as String or Class, got {:?}",
                        other
                    )));
                }
            };
            let dtype = storage_type_to_dtype(storage_type)?;
            let key = match &tuple[2] {
                Object::String(s) => s.clone(),
                other => {
                    return Err(PickleError::InvalidData(format!(
                        "Expected storage key as String, got {:?}",
                        other
                    )));
                }
            };
            // Extract total element count from index 4
            let total_elements = match tuple.get(4) {
                Some(Object::Int(n)) => Some(*n as usize),
                _ => None,
            };
            (dtype, key, total_elements)
        } else {
            return Err(PickleError::InvalidData(format!(
                "Storage tuple too short, expected at least 3 elements, got {}",
                tuple.len()
            )));
        }
    } else if !storage_info.is_empty() {
        // Legacy string-based parsing
        let storage_str = String::from_utf8_lossy(&storage_info);
        if storage_str.starts_with("Tuple(") {
            // Parse from the debug representation we stored
            let parts: Vec<&str> = storage_str
                .trim_start_matches("Tuple(")
                .trim_end_matches(")")
                .split(", ")
                .map(|s| {
                    let trimmed = s.trim_matches('"');
                    if let Some(inner) = trimmed
                        .strip_prefix("Object::String(\"")
                        .and_then(|s| s.strip_suffix("\")"))
                    {
                        inner
                    } else {
                        trimmed
                    }
                })
                .collect();

            if parts.len() >= 3 {
                let dtype = storage_type_to_dtype(parts[1])?;
                (dtype, parts[2].to_string(), None)
            } else {
                return Err(PickleError::InvalidData(format!(
                    "Storage info tuple too short, expected at least 3 parts, got {}",
                    parts.len()
                )));
            }
        } else {
            return Err(PickleError::InvalidData(format!(
                "Invalid storage info format: {}",
                storage_str
            )));
        }
    } else {
        return Err(PickleError::InvalidData(
            "No storage information available".to_string(),
        ));
    };

    // If no data source, we can't load tensor data
    let data_source = match data_source {
        Some(ds) => ds.clone(),
        None => {
            return Err(PickleError::InvalidData(
                "Cannot load tensor data without a data source".to_string(),
            ));
        }
    };

    // Create clones for the closure
    let data_source_clone = data_source.clone();
    let shape_clone = shape.clone();

    // Find the correct data file key
    let data_file_key = {
        let exact_key = format!("data/{}", storage_key);
        if data_source.contains(&exact_key) {
            exact_key
        } else {
            // Try other patterns
            data_source
                .keys()
                .into_iter()
                .find(|key| {
                    key.ends_with(&format!("/data/{}", storage_key))
                        || (key.contains("/data/") && key.rsplit('/').next() == Some(&storage_key))
                })
                .unwrap_or_else(|| format!("data/{}", storage_key))
        }
    };

    // There is no guarantee that the storage size equals the python object size.
    // The size of the object can be smaller than the storage (e.g., uncloned views
    // of a tensor are saved to the .pt/.pth files). Thus, the actual storage size
    // should be used, which can be computed using the total number of elements
    // in the storage.
    if let LazyDataSource::LegacyMultiStorage(ref source) = *data_source {
        let source = source
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let bytes_needed = match storage_total_elements {
            Some(total) => total * dtype.size(),
            None => (storage_offset + shape.iter().product::<usize>()) * dtype.size(),
        };
        source.track_storage_usage(&storage_key, 0, bytes_needed);
    }

    // Create a TensorSnapshot with a closure that loads the actual data on-demand
    Ok(Object::TorchParam(TensorSnapshot::from_closure(
        Rc::new(move || {
            // Load data only when needed
            if let Ok(data) = data_source_clone.read(&data_file_key) {
                // Parse the binary data based on dtype
                let num_elements = shape_clone.iter().product::<usize>().max(1);

                // Use dtype.size() to get element size in bytes
                let element_size = dtype.size();

                // Apply storage offset
                let offset_bytes = storage_offset * element_size;
                if offset_bytes >= data.len() {
                    return Ok(TensorData::new(
                        vec![0.0f32; num_elements],
                        shape_clone.clone(),
                    ));
                }

                let data_slice = &data[offset_bytes..];
                let available_elements = data_slice.len() / element_size;
                let elements_to_read = num_elements.min(available_elements);

                // Convert bytes to the appropriate type
                match dtype {
                    DType::F32 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let bytes = [
                                data_slice[i * element_size],
                                data_slice[i * element_size + 1],
                                data_slice[i * element_size + 2],
                                data_slice[i * element_size + 3],
                            ];
                            values.push(f32::from_le_bytes(bytes));
                        }
                        // Pad with zeros if needed
                        values.resize(num_elements, 0.0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::F64 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 8];
                            bytes.copy_from_slice(
                                &data_slice[i * element_size..(i + 1) * element_size],
                            );
                            values.push(f64::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, 0.0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::I64 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 8];
                            bytes.copy_from_slice(
                                &data_slice[i * element_size..(i + 1) * element_size],
                            );
                            values.push(i64::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::I32 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 4];
                            bytes.copy_from_slice(
                                &data_slice[i * element_size..(i + 1) * element_size],
                            );
                            values.push(i32::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::I16 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 2];
                            bytes.copy_from_slice(
                                &data_slice[i * element_size..(i + 1) * element_size],
                            );
                            values.push(i16::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::I8 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for &byte in data_slice.iter().take(elements_to_read) {
                            values.push(byte as i8);
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::Bool(BoolStore::Native) => {
                        let mut values = Vec::with_capacity(num_elements);
                        for &byte in data_slice.iter().take(elements_to_read) {
                            values.push(byte != 0);
                        }
                        values.resize(num_elements, false);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::F16 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 2];
                            bytes.copy_from_slice(
                                &data_slice[i * element_size..(i + 1) * element_size],
                            );
                            values.push(f16::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, f16::ZERO);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::BF16 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 2];
                            bytes.copy_from_slice(
                                &data_slice[i * element_size..(i + 1) * element_size],
                            );
                            values.push(bf16::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, bf16::ZERO);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::U8 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for &byte in data_slice.iter().take(elements_to_read) {
                            values.push(byte);
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::U16 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 2];
                            bytes.copy_from_slice(
                                &data_slice[i * element_size..(i + 1) * element_size],
                            );
                            values.push(u16::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::U32 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 4];
                            bytes.copy_from_slice(
                                &data_slice[i * element_size..(i + 1) * element_size],
                            );
                            values.push(u32::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    DType::U64 => {
                        let mut values = Vec::with_capacity(num_elements);
                        for i in 0..elements_to_read {
                            let mut bytes = [0u8; 8];
                            bytes.copy_from_slice(
                                &data_slice[i * element_size..(i + 1) * element_size],
                            );
                            values.push(u64::from_le_bytes(bytes));
                        }
                        values.resize(num_elements, 0);
                        Ok(TensorData::new(values, shape_clone.clone()))
                    }
                    _ => {
                        // For any remaining unsupported types, return an error
                        Err(crate::TensorSnapshotError::DataError(format!(
                            "Unsupported dtype for tensor data reading: {:?}",
                            dtype
                        )))
                    }
                }
            } else {
                // If no data file found, return zeros of the appropriate type
                let num_elements = shape_clone.iter().product::<usize>().max(1);
                match dtype {
                    DType::F32 => Ok(TensorData::new(
                        vec![0.0f32; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::F64 => Ok(TensorData::new(
                        vec![0.0f64; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::F16 => Ok(TensorData::new(
                        vec![f16::ZERO; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::BF16 => Ok(TensorData::new(
                        vec![bf16::ZERO; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::I64 => Ok(TensorData::new(
                        vec![0i64; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::I32 => Ok(TensorData::new(
                        vec![0i32; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::I16 => Ok(TensorData::new(
                        vec![0i16; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::I8 => Ok(TensorData::new(
                        vec![0i8; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::U8 => Ok(TensorData::new(
                        vec![0u8; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::U16 => Ok(TensorData::new(
                        vec![0u16; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::U32 => Ok(TensorData::new(
                        vec![0u32; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::U64 => Ok(TensorData::new(
                        vec![0u64; num_elements],
                        shape_clone.clone(),
                    )),
                    DType::Bool(BoolStore::Native) => Ok(TensorData::new(
                        vec![false; num_elements],
                        shape_clone.clone(),
                    )),
                    _ => {
                        // For any remaining unsupported types, return an error
                        Err(crate::TensorSnapshotError::DataError(format!(
                            "Unsupported dtype for tensor data reading: {:?}",
                            dtype
                        )))
                    }
                }
            }
        }),
        dtype,
        shape.into(),
        vec![],         // path_stack
        vec![],         // container_stack
        ParamId::new(), // tensor_id
    )))
}

