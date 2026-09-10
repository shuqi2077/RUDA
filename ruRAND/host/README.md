# ruRAND-host

Host random-state management and random tensor generation for Ruda. Unlike the device `ruRAND` package, these functions construct tensors in host memory.

## Interfaces

- `seed` initializes the shared host random sequence; `HostRng` names its generator type.
- `float_random(shape, distribution, dtype)` produces a floating-point host tensor.
- `int_random(shape, distribution, dtype)` produces an integer host tensor; generation advances the shared RNG state.

## Usage

Cargo package: `ruRAND-host`. Rust import: `rurand_host`.

```toml
[dependencies]
ruRAND-host = "0.1"
```

## Features

Default features: `std`.

| Feature | Purpose |
| --- | --- |
| `std` | Enable standard-library RNG and numeric support. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruRAND/host/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruRAND/host/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/rurand.md)
