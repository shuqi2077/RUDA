# Rust-authored Ascend device programs

The algorithm is in `src/kernel.rs` and `src/scheduler.rs`. It contains actual
persistent scheduling, tile/address calculations, staged transfers, pipe events,
BF16 matrix multiply-accumulate and direct output writes, not a call into a
foreign DeepGEMM implementation. `src/ascend.rs` is an instruction printer.

Supported subset: direct-store identity BF16 Dense/Batched (NN/NT/TN/TT), aligned
MGrouped NT, BF16/F32 output. The other reference-project kernels are not ported.

The Rust source builds structured device IR. The target currently lowers that IR
to CANN CCE intrinsic source and uses Bisheng. It is NOT a rustc backend producing
Ascend ISA directly and NOT a generic Ruda Kernel compiler.

`cargo test -p ruda-ascend-kernels` runs the same IR in a test-only tile memory and
pipe-token model. This checks algorithm-level ordering and addressing, not CANN
intrinsic semantics, hardware races, performance or generated machine code.

`cargo run -p ruda-ascend-kernels --bin ruda-ascend-emit -- --out NEW_DIR` generates
18 candidate translation units. It does not produce a loadable artifact manifest.
Use the checked workspace build/validation tools for actual NPU compilation.
