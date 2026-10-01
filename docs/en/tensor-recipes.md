# Tensor recipes

[Documentation](README.md) · [Backend composition](backend-composition.md) · [中文](../zh/tensor-recipes.md)

## A complete CPU tensor program

Create a binary project with `cargo new tensor-demo`. Add these dependencies to its `Cargo.toml`:

```toml
[dependencies]
ruda-tensor = { version = "0.21", default-features = false, features = ["api-std"] }
ruda-tensor-host = { version = "0.21", default-features = false, features = ["std"] }
ruda-autodiff = { version = "0.21", default-features = false, features = ["std"] }
```

Replace `src/main.rs` with:

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

Run with `cargo run`. This program uses the CPU explicitly; it does not select a GPU or require a GPU toolkit. Enable the host backend's `simd` and `rayon` features when those execution paths are wanted.

## Rank, shape, kind, and dtype

| Concept | Meaning | Example |
| --- | --- | --- |
| Backend | Storage and operation implementation | `Host`, `Autodiff<Host>` |
| Rank | Number of axes, checked in the Rust type | `Tensor<B, 2>` |
| Shape | Runtime extent of each axis | `[2, 3]` from `dims()` |
| Kind | Float, integer, or boolean operation family | `Tensor<B, 1, Int>` |
| Dtype | Runtime element representation | `DType::F32`, `DType::F64` |

`Tensor<B, 2>` is not a fixed-size matrix type. Matrix multiplication still requires compatible inner dimensions; reshape must preserve the number of elements. Slice ranges are end-exclusive.

For `Host`, use the default type `Host`, not `Host<f64, i64>`. Select precision through creation options:

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

Passing only `&device` uses the device's dtype policy. The Rust type of the input array alone does not force the tensor's storage dtype. Read data with an element type that matches its dtype.

## Ownership, views, and readback

Tensor operations commonly consume `self`. Clone a tensor handle before an operation if it is still needed, as with `a` and `x` above. A clone is not a promise of an independent deep copy; physical storage and copy-on-write behavior are backend-specific.

Transpose and slicing change logical layout. Do not pass their storage to a low-level kernel as a dense contiguous array unless that kernel accepts the actual strides. The framework operation handles layout through its backend.

`into_data()` consumes the tensor and returns host-readable `TensorData`. On a device backend, readback involves completion and data transfer. Keep intermediate results on the device rather than reading them after every operation.

## Gradients and model training

`Autodiff<B>` wraps the execution backend. Mark input leaves with `require_grad()`, build a loss, call `backward()`, and retrieve gradients with `grad(&grads)`. The returned gradient tensor uses the inner backend. `grad_remove(&mut grads)` removes the gradient when it only needs to be consumed once.

The example differentiates the mean of four squares, so its gradient is `2*x/4`. Model parameters, optimizer state, gradient accumulation, and checkpoint restoration are covered in [Training and saving state](training.md).

## API entry points

- [Tensor creation, shape, slicing, and transfers](../../ruda-tensor/src/api/base.rs).
- [Arithmetic, reductions, and matrix multiplication](../../ruda-tensor/src/api/numeric.rs).
- [Autodiff tensor methods](../../ruda-tensor/src/api/autodiff.rs).
- [Creation options and dtype policy](../../ruda-tensor/src/api/options.rs).
- [Host backend and runtime dtype dispatch](../../ruda-tensor-host/src/backend.rs).
