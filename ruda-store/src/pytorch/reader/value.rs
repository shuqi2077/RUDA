use super::*;

/// Read pickle data from a PyTorch file as a simplified value
pub(super) fn read_pickle_as_value(path: &Path, top_level_key: Option<&str>) -> Result<PickleValue> {
    use crate::pytorch::lazy_data::LazyDataSource;
    use crate::pytorch::pickle_reader::{read_pickle, read_pickle_with_data};
    use std::sync::Arc;

    // Try to open as ZIP first
    if let Ok(file) = File::open(path)
        && let Ok(mut archive) = zip::ZipArchive::new(BufReader::new(file))
    {
        // Read pickle data from ZIP
        let mut pickle_data = Vec::new();

        // Try standard locations
        for pickle_path in &["data.pkl", "archive/data.pkl"] {
            if let Ok(mut pickle_file) = archive.by_name(pickle_path) {
                pickle_file.read_to_end(&mut pickle_data)?;
                break;
            }
        }

        // If not found, search for any .pkl file
        if pickle_data.is_empty() {
            for i in 0..archive.len() {
                let file = archive.by_index(i)?;
                let name = file.name().to_string();
                drop(file);

                if name.ends_with("data.pkl") {
                    let mut file = archive.by_index(i)?;
                    file.read_to_end(&mut pickle_data)?;
                    break;
                }
            }
        }

        if !pickle_data.is_empty() {
            // Create a data source for the ZIP file
            let data_source = LazyDataSource::from_zip(path)?;
            let data_source_arc = Arc::new(data_source);

            let mut reader = BufReader::new(pickle_data.as_slice());
            let obj = read_pickle_with_data(&mut reader, data_source_arc)?;
            return convert_object_to_value(obj, top_level_key);
        }
    }

    // Try as plain pickle file
    // First attempt without data source (for pure metadata files)
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    match read_pickle(&mut reader) {
        Ok(obj) => convert_object_to_value(obj, top_level_key),
        Err(e)
            if e.to_string()
                .contains("Cannot load tensor data without a data source") =>
        {
            // File contains tensors, need to use full PytorchReader
            // Use the regular reader to get proper tensor handling
            let reader = PytorchReader::new(path)?;

            // Convert tensors to PickleValue structure
            let mut result = std::collections::HashMap::new();
            for key in reader.keys() {
                // For pickle value extraction, we just need the structure, not the actual data
                result.insert(
                    key.clone(),
                    PickleValue::String(format!("<Tensor:{}>", key)),
                );
            }

            if let Some(key) = top_level_key {
                Ok(PickleValue::Dict(
                    [(key.to_string(), PickleValue::Dict(result))]
                        .into_iter()
                        .collect(),
                ))
            } else {
                Ok(PickleValue::Dict(result))
            }
        }
        Err(e) => Err(PytorchError::Pickle(e)),
    }
}

/// Convert internal Object to public PickleValue
fn convert_object_to_value(obj: Object, top_level_key: Option<&str>) -> Result<PickleValue> {
    use crate::pytorch::pickle_reader::Object;

    // If a top-level key is specified, extract it first
    if let Some(key) = top_level_key
        && let Object::Dict(dict) = obj
    {
        if let Some(value) = dict.get(key) {
            return object_to_pickle_value(value.clone());
        } else {
            return Err(PytorchError::KeyNotFound(format!(
                "Key '{}' not found in pickle data",
                key
            )));
        }
    }

    object_to_pickle_value(obj)
}

/// Convert Object to PickleValue
fn object_to_pickle_value(obj: Object) -> Result<PickleValue> {
    use crate::pytorch::pickle_reader::Object;

    Ok(match obj {
        Object::None => PickleValue::None,
        Object::Bool(b) => PickleValue::Bool(b),
        Object::Int(i) => PickleValue::Int(i),
        Object::Float(f) => PickleValue::Float(f),
        Object::String(s) => PickleValue::String(s),
        Object::Persistent(data) => {
            // Persistent data is raw bytes
            PickleValue::Bytes(data)
        }
        Object::PersistentTuple(tuple) => {
            // Convert persistent tuples to lists
            let mut values = Vec::new();
            for item in tuple {
                values.push(object_to_pickle_value(item)?);
            }
            PickleValue::List(values)
        }
        Object::List(list) => {
            let mut values = Vec::new();
            for item in list {
                values.push(object_to_pickle_value(item)?);
            }
            PickleValue::List(values)
        }
        Object::Dict(dict) => {
            let mut map = HashMap::new();
            for (k, v) in dict {
                map.insert(k, object_to_pickle_value(v)?);
            }
            PickleValue::Dict(map)
        }
        Object::Tuple(tuple) => {
            // Convert tuples to lists in the public API
            let mut values = Vec::new();
            for item in tuple {
                values.push(object_to_pickle_value(item)?);
            }
            PickleValue::List(values)
        }
        Object::TorchParam(_) => {
            // Skip tensor parameters in config reading
            PickleValue::None
        }
        Object::Class { .. } | Object::Build { .. } | Object::Reduce { .. } => {
            // Complex objects are represented as None for simplicity
            PickleValue::None
        }
    })
}

/// Convert PickleValue to NestedValue for deserialization
pub(super) fn convert_pickle_to_nested_value(value: PickleValue) -> Result<NestedValue> {
    Ok(match value {
        PickleValue::None => NestedValue::Default(None),
        PickleValue::Bool(b) => NestedValue::Bool(b),
        PickleValue::Int(i) => NestedValue::I64(i),
        PickleValue::Float(f) => NestedValue::F64(f),
        PickleValue::String(s) => NestedValue::String(s),
        PickleValue::List(list) => {
            let mut vec = Vec::new();
            for item in list {
                vec.push(convert_pickle_to_nested_value(item)?);
            }
            NestedValue::Vec(vec)
        }
        PickleValue::Dict(dict) => {
            let mut map = HashMap::new();
            for (k, v) in dict {
                map.insert(k, convert_pickle_to_nested_value(v)?);
            }
            NestedValue::Map(map)
        }
        PickleValue::Bytes(data) => {
            // Convert bytes to a list of u8 values
            let vec: Vec<NestedValue> = data.into_iter().map(NestedValue::U8).collect();
            NestedValue::Vec(vec)
        }
    })
}
