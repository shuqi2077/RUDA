# Data pipelines and model storage

[Documentation](README.md) · [Training](training.md) · [中文](../zh/data-and-storage.md)

## From samples to tensor batches

`ruda-dataset` supplies the `Dataset<I>` abstraction: `get(index)`, `len()`, and iteration. `ruda-model` adds batching and data loaders under its `dataset` feature. A `Batcher<B, I, O>` receives samples and the destination backend device, then builds the batch.

Create a binary project and use:

```toml
[dependencies]
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-host = { version = "0.21", default-features = false, features = ["std"] }
ruda-model = { version = "0.21", default-features = false, features = ["std", "dataset"] }
```

Put this program in `src/main.rs` and run `cargo run`:

```rust
use ruda_model::data::{
    dataloader::{batcher::Batcher, DataLoaderBuilder},
    dataset::InMemDataset,
};
use ruda_tensor::{api::Tensor, DType, TensorData};
use ruda_tensor_host::{Host, HostDevice};

struct PairBatcher;

impl Batcher<Host, [f32; 2], Tensor<Host, 2>> for PairBatcher {
    fn batch(&self, items: Vec<[f32; 2]>, device: &HostDevice) -> Tensor<Host, 2> {
        let rows = items.len();
        let values: Vec<f32> = items.into_iter().flatten().collect();
        Tensor::from_data(TensorData::new(values, [rows, 2]), (device, DType::F32))
    }
}

fn main() {
    let dataset = InMemDataset::new(vec![
        [1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],
    ]);
    let loader = DataLoaderBuilder::<Host, [f32; 2], Tensor<Host, 2>>::new(PairBatcher)
        .batch_size(2)
        .shuffle(42)
        .num_workers(0)
        .set_device(HostDevice)
        .build(dataset);
    let mut rows = 0;
    for batch in loader.iter() {
        assert_eq!(batch.dims()[1], 2);
        rows += batch.dims()[0];
    }
    assert_eq!(rows, 5);
}
```

The final batch has one row; derive its shape from `items.len()` rather than the configured batch size. `num_workers(0)` batches on the calling thread; a positive number enables background workers. `set_device` selects the device passed to the batcher, not a separate automatic transfer after batching.

`shuffle(seed)` controls the loader's shuffle generator. Iterating the same loader again advances that generator; the seed by itself is not a full mid-epoch training checkpoint.

## Dataset choices

| Input | Entry point | Behavior |
| --- | --- | --- |
| Existing in-memory samples | `InMemDataset::new(Vec<I>)` | Owns the samples; retrieves cloned items |
| Another dataset | `InMemDataset::from_dataset` | Materializes all samples in RAM |
| JSON Lines | `InMemDataset::from_json_rows` | Deserializes one JSON sample per line |
| CSV | `InMemDataset::from_csv` | Deserializes rows with the supplied CSV reader configuration |
| SQLite | `SqliteDataset` | Enable `sqlite` or `sqlite-bundled` |
| Custom source | Implement `Dataset<I>` | Define indexed access and dataset length |

`ruda-dataset::transform` contains adapters for composing datasets; keep sample decoding in the dataset/transform and tensor assembly in the batcher. For downloading resources, `ruda-io` exposes `network::downloader::download_file_as_bytes` with its `network` feature; model/data storage is a separate responsibility.

## Saving and loading a model

Add these dependencies to the project above:

```toml
ruda-nn = { version = "0.21", default-features = false, features = ["std"] }
ruda-store = { version = "0.21", default-features = false, features = ["std", "rudapack"] }
```

Replace `src/main.rs` with the following round trip. It writes `linear.bpk` in the working directory, replacing that file on repeated runs.

```rust
use ruda_nn::LinearConfig;
use ruda_store::{ModuleSnapshot, RudapackStore};
use ruda_tensor::{api::Tensor, DType};
use ruda_tensor_host::{Host, HostDevice};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = HostDevice;
    let model = LinearConfig::new(2, 1).init::<Host>(&device);
    let input = Tensor::<Host, 2>::from_data([[1.0f32, 2.0]], (&device, DType::F32));
    let expected = model.forward(input.clone()).into_data();

    let mut output = RudapackStore::from_file("linear.bpk").overwrite(true);
    model.save_into(&mut output)?;

    let mut restored = LinearConfig::new(2, 1).init::<Host>(&device);
    let mut source = RudapackStore::from_file("linear.bpk");
    let result = restored.load_from(&mut source)?;
    assert!(result.errors.is_empty());
    assert!(result.missing.is_empty());
    let actual = restored.forward(input).into_data();
    assert_eq!(actual.as_slice::<f32>()?, expected.as_slice::<f32>()?);
    Ok(())
}
```

Construct the same model architecture before loading. `ModuleSnapshot` visits the model's parameters; a weights file does not reconstruct arbitrary Rust model code.

## Choose the storage format

| Store | Feature | Use |
| --- | --- | --- |
| `RudapackStore` | `rudapack` | RUDA tensor snapshots with parameter IDs; default extension `.bpk` |
| `SafetensorsStore` | `safetensors` | Read/write safetensors weights and metadata |
| `PytorchStore` | `pytorch` | Import PyTorch checkpoints and state dictionaries |

All three stores are enabled by `ruda-store` defaults. For PyTorch dictionaries nested beneath a key, use `with_top_level_key("state_dict")`. Use the PyTorch adapter path for layout/name conventions rather than assuming a raw byte copy is enough.

Filters such as `with_regex` select parameter paths. Safetensors and PyTorch stores use `with_key_remapping` to map parameter names; Rudapack uses `with_remap_pattern` or `remap(KeyRemapper)`. `allow_partial(true)` explicitly permits a partial load; the default is false. Examine `ApplyResult.applied`, `skipped`, `missing`, `unused`, and `errors` to distinguish loaded weights from omitted or incompatible entries. Shape and dtype mismatches are different errors.

`HalfPrecisionAdapter` converts floating-point snapshots for storage. Apply it with `with_to_adapter` when saving and `with_from_adapter` when loading. Its `without_module("LayerNorm")` option can keep normalization parameters at full precision; see the [complete half-precision example](../../ruda-store/examples/half_precision.rs).

## Weights versus training state

Loading weights is sufficient to restore model parameters, not an exact training position. Resume workflows also need the optimizer state, scheduler/step, and the data-order/random state appropriate to the training loop. Rudapack preserves parameter IDs, but does not itself capture the entire application's state. Use the record and optimizer restoration workflow in [Training and saving state](training.md).

## API entry points

- [Dataset implementations](../../ruda-dataset/src/dataset/mod.rs) and [transform adapters](../../ruda-dataset/src/transform/mod.rs).
- [Batcher contract](../../ruda-model/src/data/dataloader/batcher.rs) and [loader builder](../../ruda-model/src/data/dataloader/builder.rs).
- [Storage API exports](../../ruda-store/src/lib.rs), [Rudapack store](../../ruda-store/src/rudapack/store.rs), and [PyTorch store](../../ruda-store/src/pytorch/store.rs).
