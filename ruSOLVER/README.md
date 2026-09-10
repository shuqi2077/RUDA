# ruSOLVER

Experimental numerical algebra with explicit execution and error contracts.
Default is synchronous host FP64 and has no dependencies. Native device kernels,
ruSPARSE format adapters, and ruCCL transport adapters are separately enabled.

Host: pivoted LU; Cholesky; pivoted QR/least squares; Jacobi thin SVD/minimum-norm
solve/pseudoinverse; real symmetric eigen; complex FP64 LU, Cholesky and QR; sparse
LU without densification; row-partitioned CG; analytic reverse-mode pullbacks.

Device: opt-in FP32 small batched LU, Cholesky, QR, symmetric eigen and dense CG.
One device thread per system: a correctness-first baseline, NOT a cuSOLVER
performance claim. Every batch has a status tensor that must be checked.

[Extended guide (中文)](../docs/zh/libraries/science-extended.md) ·
[English guide](../docs/en/libraries/science-extended.md) ·
[Original solver guide](../docs/en/libraries/rusolver.md)

There is no
LAPACK/cuSOLVER ABI compatibility, distributed sparse-direct factorization, or
implicit transfer of GPU matrices to a CPU solver.
