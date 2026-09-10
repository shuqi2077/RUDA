# ruPRIM-host

CPU tensor primitives shared by Ruda host-domain libraries. This package implements host elementwise operations, reductions, cumulative operations, indexing, and tensor rearrangement.

## Interfaces

- `binary`, `unary`, and `comparison` implement elementwise primitives.
- `reduce`, `cumulative`, and `sort` implement aggregation and ordering.
- `gather_scatter`, `slice`, `mask`, `cat`, and `unfold` implement indexing and layout operations.

## Usage

Cargo package: `ruPRIM-host`. Rust import: `ruprim_host`.

```toml
[dependencies]
ruPRIM-host = "0.1"
```

## Features

Default features: `std`, `simd`, `rayon`.

| Feature | Purpose |
| --- | --- |
| `simd` | Enable SIMD primitives. |
| `rayon` | Enable parallel host tensor processing. |
| `std` | Enable standard-library numeric support. |

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruPRIM/host/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruPRIM/host/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/ruprim.md)
