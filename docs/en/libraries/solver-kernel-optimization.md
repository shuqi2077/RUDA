# Experimental warp-shared LU and Cholesky

[中文 implementation/verification guide](../../zh/libraries/solver-kernel-optimization.md)

Explicit new APIs in `rusolver::tensor` behind the OFF-by-default
`warp-solvers` feature (`cuda,warp-solvers` for NVIDIA; the runner enables both):

- `cholesky_solve_batched_warp(a, b, BatchedCholeskyOptions)`
- `lu_solve_batched_warp(a, b, BatchedLuOptions)`

One full 32-thread block owns each small FP32 system. Striped/coalesced global
loads feed shared matrix/RHS scratch; independent rows and RHS columns are
assigned to lanes. LU uses warp max/min pivot selection, preserving the first
maximum tie rule. Odd shared pitch avoids duplicate banks for fixed-column
32-bit row accesses in a 32-bank model. All memory handoffs use block barriers;
no implicit warp lockstep or lane-local exits around synchronization.

n=1..32, nrhs=1..8, contiguous row-major, same device/queue. Requires fixed
32-lane plane operations and sufficient block/shared/grid limits. Inputs are
unchanged, outputs are new allocations, numeric failure zeros the entire failed
system and returns the existing status codes. No CPU fallback, casting or
hidden diagonal jitter. The existing serial APIs remain the default.

At n=32/nrhs=8, shared declarations consume 5248 bytes/block for Cholesky and
5376 for LU. Intermediate global scratch accesses are replaced with shared
accesses. These are source-level facts, NOT DRAM counters or measured speedup.
One warp/block can reduce occupancy at a resident-block limit; tiny systems
may be slower. No automatic crossover policy is installed.

```bash
python tools/run_kernel_optimization_regressions.py --suite plan
python tools/run_kernel_optimization_regressions.py --suite build
python tools/run_kernel_optimization_regressions.py --suite cuda --compiler both
python tools/run_kernel_optimization_regressions.py --suite sanitizer --compiler both
python tools/run_kernel_optimization_regressions.py --suite bench --compiler both --batch 256 --order 16 --rhs 4
```

`plan` compiles the real standalone Rust launch planner; `build` expands device
macros via Cargo; CUDA tests include the original baseline tests. Sanitizer runs
memcheck/racecheck/synccheck with nonzero error exits. The benchmark warms both
paths, alternates measurement order, checks every output vs FP64 solve and
factor/residual references, and reports synchronized API wall time INCLUDING
allocation/submission/completion. No GPU-only event timing is claimed.
