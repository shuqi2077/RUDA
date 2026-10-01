# 张量实用示例

[文档首页](README.md) · [后端组合](backend-composition.md) · [English](../en/tensor-recipes.md)

## 完整的 CPU 张量程序

执行 `cargo new tensor-demo` 创建二进制项目，在其 `Cargo.toml` 中添加：

```toml
[dependencies]
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-host = { version = "0.21", default-features = false, features = ["std"] }
ruda-autodiff = { version = "0.21", default-features = false, features = ["std"] }
```

将 `src/main.rs` 替换为：

```rust
use ruda_autodiff::Autodiff;
use ruda_tensor::{api::Tensor, DType};
use ruda_tensor_host::{Host, HostDevice};

fn main() {
    let device = HostDevice;
    let a = Tensor::<Host, 2>::from_data(
        [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
        (&device, DType::F32),
    );
    let b = Tensor::<Host, 2>::from_data(
        [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]],
        (&device, DType::F32),
    );
    let product = a.clone().matmul(b);
    assert_eq!(product.dims(), [2, 2]);
    assert_eq!(
        product.into_data().as_slice::<f32>().unwrap(),
        &[22.0, 28.0, 49.0, 64.0],
    );

    let transposed = a.clone().transpose();
    assert_eq!(transposed.dims(), [3, 2]);
    assert_eq!(
        transposed.into_data().as_slice::<f32>().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );
    let last_row = a.slice([1..2, 0..3]).reshape([3]);
    assert_eq!(
        last_row.into_data().as_slice::<f32>().unwrap(),
        &[4.0, 5.0, 6.0],
    );

    type Train = Autodiff<Host>;
    let x = Tensor::<Train, 1>::from_data(
        [1.0f32, 2.0, 3.0, 4.0], (&device, DType::F32),
    ).require_grad();
    let loss = (x.clone() * x.clone()).mean();
    let grads = loss.backward();
    let dx = x.grad(&grads).expect("x participates in loss");
    assert_eq!(
        dx.into_data().as_slice::<f32>().unwrap(),
        &[0.5, 1.0, 1.5, 2.0],
    );
}
```

执行 `cargo run`。该程序明确使用 CPU，不选择 GPU，也不需要 GPU 工具链。需要相应执行路径时，可为 Host 后端启用 `simd` 和 `rayon` feature。

## 秩、形状、类别与数据类型

| 概念 | 含义 | 示例 |
| --- | --- | --- |
| Backend | 存储和算子的实现 | `Host`、`Autodiff<Host>` |
| 秩 | 轴数量，体现在 Rust 类型中 | `Tensor<B, 2>` |
| 形状 | 各轴在运行时的长度 | `dims()` 返回的 `[2, 3]` |
| 类别 | 浮点、整数或布尔算子族 | `Tensor<B, 1, Int>` |
| Dtype | 运行时元素表示 | `DType::F32`、`DType::F64` |

`Tensor<B, 2>` 不表示固定大小的矩阵。矩阵乘仍要求内维匹配，reshape 必须保持元素总数，切片区间不包含右端点。

Host 后端使用默认类型 `Host`，不要用 `Host<f64, i64>` 指定精度。通过创建选项指定：

```rust
use ruda_tensor::{api::Tensor, DType};
use ruda_tensor_host::{Host, HostDevice};

fn main() {
    let device = HostDevice;
    let x = Tensor::<Host, 1>::from_data(
        [1.0f64, 2.0], (&device, DType::F64),
    );
    assert_eq!(x.dtype(), DType::F64);
    assert_eq!(x.into_data().as_slice::<f64>().unwrap(), &[1.0, 2.0]);
}
```

只传 `&device` 时使用设备的 dtype 策略，输入数组的 Rust 元素类型本身不强制决定张量存储精度。回读数据时，切片元素类型需要与 dtype 一致。

## 所有权、视图与回读

张量算子通常消耗 `self`。后续还需要原张量时，先克隆句柄，如上例的 `a` 和 `x`。克隆不保证得到独立的深拷贝；具体存储共享和写时复制由后端决定。

转置和切片会改变逻辑布局。除非底层内核接受实际步长，否则不能把视图的存储直接当作连续稠密数组传入。框架算子通过后端处理布局。

`into_data()` 消耗张量，返回主机可读的 `TensorData`。设备后端回读涉及执行完成与数据传输；应把中间结果留在设备上，而不是每做一步就回读。

## 梯度与模型训练

`Autodiff<B>` 包装执行后端。对需要梯度的输入叶子调用 `require_grad()`，构建损失后执行 `backward()`，再用 `grad(&grads)` 取出梯度。返回的梯度张量使用内层后端；只消费一次时，可用 `grad_remove(&mut grads)` 移出梯度。

上例对四个平方值的均值求导，因此梯度为 `2*x/4`。模型参数、优化器状态、梯度累积与检查点恢复见[训练与状态保存](training.md)。

## API 入口

- [张量创建、形状、切片与迁移](../../ruda-tensor/src/api/base.rs)。
- [算术、归约与矩阵乘](../../ruda-tensor/src/api/numeric.rs)。
- [自动微分张量方法](../../ruda-tensor/src/api/autodiff.rs)。
- [创建选项与 dtype 策略](../../ruda-tensor/src/api/options.rs)。
- [Host 后端与运行时 dtype 分发](../../ruda-tensor-host/src/backend.rs)。
