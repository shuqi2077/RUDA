use super::*;
use half::f16;
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::readback::into_data_sync;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ruda-awq-test-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(
            path.join(CONFIG_FILE),
            serde_json::to_vec(&json!({"quantization_config": {
                "quant_method": "awq", "bits": 4, "group_size": -1,
                "zero_point": true, "version": "GEMM", "modules_to_not_convert": ["lm_head"]
            }}))
            .unwrap(),
        )
        .unwrap();
        Self(path)
    }

    fn write_shard(&self, file: &str, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = payload.len();
            payload.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        std::fs::write(self.0.join(file), bytes).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        for entry in std::fs::read_dir(&self.0).unwrap() {
            std::fs::remove_file(entry.unwrap().path()).unwrap();
        }
        std::fs::remove_dir(&self.0).unwrap();
    }
}

fn tensors() -> Vec<(&'static str, &'static str, Vec<usize>, Vec<u8>)> {
    vec![
        (
            "model.proj.qweight",
            "I32",
            vec![2, 1],
            [0x75316420u32.to_le_bytes(), 0xfdb9eca8u32.to_le_bytes()].concat(),
        ),
        (
            "model.proj.qzeros",
            "I32",
            vec![1, 1],
            0x11111111u32.to_le_bytes().to_vec(),
        ),
        (
            "model.proj.scales",
            "F16",
            vec![1, 8],
            (0..8)
                .flat_map(|_| f16::from_f32(0.5).to_bits().to_le_bytes())
                .collect(),
        ),
    ]
}

#[test]
fn awq_configuration_rejects_other_formats_and_keeps_full_channel_groups() {
    let mut config = AwqQuantizationConfig {
        quant_method: "awq".into(),
        bits: 4,
        group_size: -1,
        zero_point: true,
        version: "GEMM".into(),
        modules_to_not_convert: None,
    };
    assert_eq!(config.layout(128, 16).unwrap().group_size, 128);
    config.group_size = 0;
    assert!(config.validate().is_err());
    config.group_size = 32;
    config.version = "gemv".into();
    assert!(config.validate().is_err());
    config.version = "gemm".into();
    config.bits = 8;
    assert!(config.validate().is_err());
    config.bits = 4;
    config.zero_point = false;
    assert!(config.validate().is_err());
}

#[test]
fn awq_checkpoint_keeps_packed_bytes_and_executes_native_projection() {
    let fixture = Fixture::new();
    fixture.write_shard(SINGLE_WEIGHTS_FILE, &tensors());
    let checkpoint = AwqCheckpoint::open(&fixture.0).unwrap();
    let packed = checkpoint.tensor_data("model.proj.qweight").unwrap();
    assert_eq!(packed.dtype, DType::I32);
    assert_eq!(
        packed.to_vec::<i32>().unwrap(),
        vec![0x75316420i32, 0xfdb9eca8u32 as i32]
    );
    let device = CudaDevice::default();
    let linear = checkpoint
        .load_linear::<CudaRuntime>("model.proj", 2, 8, false, &device)
        .unwrap();
    drop(checkpoint);
    let input = from_data(
        TensorData::new(vec![f16::from_f32(1.0), f16::from_f32(2.0)], [1, 2]),
        &device,
    );
    let actual = into_data_sync(linear.forward(input).unwrap())
        .to_vec::<f16>()
        .unwrap();
    let expected = (0..8)
        .map(|column| f16::from_f32(1.5 * column as f32 + 6.5))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn awq_checkpoint_loads_split_projection_and_checks_index_ownership() {
    let fixture = Fixture::new();
    let tensors = tensors();
    fixture.write_shard("a.safetensors", &tensors[..1]);
    fixture.write_shard("b.safetensors", &tensors[1..]);
    let mut map = json!({
        "model.proj.qweight": "a.safetensors",
        "model.proj.qzeros": "b.safetensors",
        "model.proj.scales": "b.safetensors"
    });
    let write_index = |map: &serde_json::Value| {
        std::fs::write(
            fixture.0.join(WEIGHTS_INDEX_FILE),
            serde_json::to_vec(&json!({"weight_map": map})).unwrap(),
        )
        .unwrap()
    };
    write_index(&map);
    let checkpoint = AwqCheckpoint::open(&fixture.0).unwrap();
    assert_eq!(checkpoint.weight_files().len(), 2);
    assert_eq!(checkpoint.tensor_names().count(), 3);
    drop(checkpoint);
    map["model.proj.qweight"] = json!("b.safetensors");
    write_index(&map);
    assert!(AwqCheckpoint::open(&fixture.0).is_err());
}

#[test]
fn awq_checkpoint_rejects_missing_or_malformed_projection_before_upload() {
    let fixture = Fixture::new();
    let mut tensors = tensors();
    tensors[2].2 = vec![2, 4];
    fixture.write_shard(SINGLE_WEIGHTS_FILE, &tensors);
    let checkpoint = AwqCheckpoint::open(&fixture.0).unwrap();
    let device = CudaDevice::default();
    assert!(
        checkpoint
            .load_linear::<CudaRuntime>("model.proj", 2, 8, false, &device)
            .is_err()
    );
    assert!(
        checkpoint
            .load_linear::<CudaRuntime>("absent", 2, 8, false, &device)
            .is_err()
    );
}
