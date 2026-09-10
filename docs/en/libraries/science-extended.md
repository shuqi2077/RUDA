# Extended numerical science (experimental)

This extends the existing `rusolver`,
`ruintegrate` and opt-in `ruda-autodiff/solver-host`.

## Execution contracts

- **Host FP64:** thin one-sided Jacobi SVD, pseudoinverse and minimum-norm solve;
  complex LU/adjoint solve, Hermitian Cholesky, thin Householder QR; sparse LU
  with partial row pivoting and explicit fill budget; row-partitioned CG;
  analytic solve/Cholesky/SVD/symmetric-eigen/QR reverse-mode pullbacks.
- **Existing Ruda autodiff graph:** optional FP64 `Autodiff<Host>` functions
  `solve_host`, `cholesky_host`, `singular_values_host`,
  `symmetric_eigenvalues_host`. First-order only. GPU tensors cannot be silently
  transferred to this API. Full SVD/QR/eigenvector VJPs are explicit low-level
  pullbacks, not multi-output graph wrappers yet.
- **Native device FP32:** LU+solve (n<=32, nrhs<=8), QR (n<=32,m<=64,m>=n), real
  symmetric eigen (n<=32), dense SPD CG (n<=128, one RHS). All are contiguous
  small-batch, one-thread-per-system baselines, no host numerical fallback.
  They complement the existing device Cholesky. Call `check_status_sync()`;
  kernel submission is not success.
- **Integrators:** adaptive backward Euler/BDF1 with Newton and analytic or
  finite-difference Jacobian; sign-change events for RK45/BDF1; rational
  half/infinite-domain transformations over the existing GK15 integrator.
  Host FP64 only. `ruintegrate` now depends on local host `rusolver` for LU.

`Complex64` uses TWO f64 components (128 storage bits), unlike CUDA's 64-bit
single-precision complex storage name. SVD returns thin U(m,k), s(k), VT(k,n).
Default rank cutoff is `max(abs, rel * FrobeniusNorm(A))`; truncated singular
values are explicitly zero. Rank-deficient factors may thus be approximations.
A nonconverged Jacobi sweep budget is an error, not an accepted factorization.

Sparse LU never allocates a dense n*n factor; BTreeMap fill and row pivot scans
are not equivalent to an optimized supernodal/ordering-based solver. The fill
limit counts factor nonzeros, not total allocator bytes. `sparse` optionally
borrows existing ruSPARSE CSR formats. Distributed CG keeps matrix rows local,
all-gathers full search vectors and reduces scalars through ordered all-gather.
It is host FP64, O(n) vectors/rank, not a GPU halo-exchange or distributed direct
factorization. `collective` reuses existing ruCCL TCP via `RucclCommunicator`.
It needs a dedicated session; failures abort the group. Retry with a new session.

Spectral gradients reject repeated/near-degenerate spectra; SVD additionally
requires full rank and separated nonzero singular values. QR differentiates with
its forward pivot permutation fixed, not the discrete pivot selection. Symmetric
input gradients use the full symmetric-matrix convention. Graph backward uses
the existing non-fallible Backward contract; numerical errors panic explicitly,
not return fabricated zero gradients. No complex/sparse/distributed/GPU graph
integration or higher derivatives are provided by this patch.

BDF1 uses a full step vs two half steps for error estimation, and accepts the two
half steps without extrapolation. It is first order, dense Newton/LU, default
max dimension 256, not variable-order BDF/Radau. Event location uses cubic
Hermite interpolation of accepted step endpoint states/derivatives and bisection.
Direction is physical-time direction even when integrating backward. Tangencies
and multiple crossings inside a step may be missed; limit max_step as needed.
Root tolerance is not a global ODE error bound. Infinite transforms integrate
the two tails separately, never claim Cauchy principal-value cancellation as
convergence. Estimated errors do not prove integrability.

Examples: `advanced-solver`, `advanced-integrate`, `solver-cuda-advanced` and
`distributed-cg`. Existing numerical and tensor-LU APIs remain in place.
`OdeStatus` adds `Event`: downstream exhaustive matches must add a case.

Mathematical references (not source-code copies): LAPACK Jacobi SVD/DGESVJ and
complex Householder QR documentation; SciPy solve_ivp/quad references for scope
and validation; PyTorch linalg autograd documentation for spectral restrictions.
See THIRD_PARTY_NOTICES.md and [中文完整指南](../../zh/libraries/science-extended.md).
