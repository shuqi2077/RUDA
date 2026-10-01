# 数据管线与模型存储

[文档首页](README.md) · [训练指南](training.md) · [English](../en/data-and-storage.md)

## 从样本到张量批次

`ruda-dataset` 提供 `Dataset<I>` 抽象，包括 `get(index)`、`len()` 与迭代。`ruda-model` 的 `dataset` feature 提供组批和数据加载器。`Batcher<B, I, O>` 接收样本及目标后端设备，构造输出批次。

创建二进制项目并配置：

```toml
[dependencies]
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-host = { version = "0.21", default-features = false, features = ["std"] }
ruda-model = { version = "0.21", default-features = false, features = ["std", "dataset"] }
```

将下面程序放入 `src/main.rs`，执行 `cargo run`：

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

最后一批只有一行，因此形状应根据 `items.len()` 生成，而不是始终使用配置的 batch size。`num_workers(0)` 在调用线程组批，正数启用后台 worker。`set_device` 决定传给 batcher 的设备，不代表组批结束后还有一次自动设备迁移。

`shuffle(seed)` 控制加载器的洗牌随机数生成器。再次迭代同一个加载器会推进生成器状态；仅保存 seed 不能完整恢复到训练中的某个批次。

## 数据集选择

| 输入 | 入口 | 行为 |
| --- | --- | --- |
| 已有内存样本 | `InMemDataset::new(Vec<I>)` | 持有样本，读取时克隆条目 |
| 其他数据集 | `InMemDataset::from_dataset` | 把全部样本载入内存 |
| JSON Lines | `InMemDataset::from_json_rows` | 每行反序列化一个 JSON 样本 |
| CSV | `InMemDataset::from_csv` | 根据传入的 CSV reader 配置反序列化各行 |
| SQLite | `SqliteDataset` | 启用 `sqlite` 或 `sqlite-bundled` |
| 自定义来源 | 实现 `Dataset<I>` | 定义按索引访问与数据集长度 |

`ruda-dataset::transform` 提供数据集组合适配器。样本解码放在 dataset/transform 中，张量组装放在 batcher 中。下载资源可使用启用了 `network` feature 的 `ruda-io` 所提供的 `network::downloader::download_file_as_bytes`；它与模型、数据存储是不同职责。

## 保存和加载模型

在上述项目中追加依赖：

```toml
ruda-nn = { version = "0.21", default-features = false, features = ["std"] }
ruda-store = { version = "0.21", default-features = false, features = ["std", "rudapack"] }
```

将 `src/main.rs` 替换为下面的保存／恢复程序。它在工作目录写入 `linear.bpk`，重复运行会覆盖该文件。

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

加载前需要构造相同架构的模型。`ModuleSnapshot` 遍历模型参数；权重文件不会自动重建任意 Rust 模型代码。

## 存储格式选择

| Store | Feature | 用途 |
| --- | --- | --- |
| `RudapackStore` | `rudapack` | 保存参数 ID 的 RUDA 张量快照，默认扩展名 `.bpk` |
| `SafetensorsStore` | `safetensors` | 读写 safetensors 权重和元数据 |
| `PytorchStore` | `pytorch` | 导入 PyTorch 检查点与 state dictionary |

`ruda-store` 默认启用三种格式。PyTorch 权重嵌套在键下时，使用 `with_top_level_key("state_dict")`。布局和命名惯例通过 PyTorch 适配路径处理，不能假定只复制原始字节即可完成转换。

`with_regex` 等过滤器选择参数路径。Safetensors 和 PyTorch Store 用 `with_key_remapping` 映射参数名；Rudapack 使用 `with_remap_pattern` 或 `remap(KeyRemapper)`。`allow_partial(true)` 显式允许部分加载，默认值是 false。检查 `ApplyResult.applied`、`skipped`、`missing`、`unused` 和 `errors`，区分已载入、未匹配及不兼容的条目。形状不匹配和 dtype 不匹配是不同错误。

`HalfPrecisionAdapter` 对浮点快照进行存储精度转换：保存时使用 `with_to_adapter`，读取时使用 `with_from_adapter`。`without_module("LayerNorm")` 可使归一化参数保留全精度，见[完整半精度示例](../../ruda-store/examples/half_precision.rs)。

## 权重与训练状态的区别

加载权重恢复的是模型参数，不是完整训练位置。续训还需要训练循环使用的优化器状态、调度器／步数，以及数据顺序和随机状态。Rudapack 保留参数 ID，但不会自行捕获应用的全部状态。Record 与优化器恢复流程见[训练与状态保存](training.md)。

## API 入口

- [数据集实现](../../ruda-dataset/src/dataset/mod.rs)与[变换适配器](../../ruda-dataset/src/transform/mod.rs)。
- [Batcher 契约](../../ruda-model/src/data/dataloader/batcher.rs)与[加载器构造器](../../ruda-model/src/data/dataloader/builder.rs)。
- [存储 API 导出](../../ruda-store/src/lib.rs)、[Rudapack Store](../../ruda-store/src/rudapack/store.rs)和 [PyTorch Store](../../ruda-store/src/pytorch/store.rs)。
