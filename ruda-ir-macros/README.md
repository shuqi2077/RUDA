# ruda-ir-macros

Procedural derives for Ruda's intermediate-representation types. These macros generate operation argument conversion, reflection, opcodes, and type hashes; they do not compile or execute GPU kernels.

## Interfaces

- `OperationArgs`: derive argument-list conversion for operation fields.
- `OperationReflect` and `OperationCode`: derive operation reflection and opcode enums.
- `TypeHash`: derive a type hash with `type_hash` attributes.

## Usage

Cargo package: `ruda-ir-macros`. Rust import: `ruda_ir_macros`.

```toml
[dependencies]
ruda-ir-macros = "0.1"
```

## Links

- [Package source](https://github.com/shuqi2077/RUDA/tree/main/ruda-ir-macros/src)
- [Cargo manifest](https://github.com/shuqi2077/RUDA/blob/main/ruda-ir-macros/Cargo.toml)
- [Ruda guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/compiler-guide.md)
