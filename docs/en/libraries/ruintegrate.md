# ruINTEGRATE — numerical integration and initial-value ODEs

[Library index](README.md) · [中文](../../zh/libraries/ruintegrate.md)

**Experimental source.**
Package `ruintegrate`, directory `ruINTEGRATE`. Host FP64, not a GPU backend; the extended stiff method depends on the local host rusolver crate.

- `integrate`: finite-interval adaptive Gauss–Kronrod 15/7 quadrature, largest-error-first bisection, absolute/relative tolerance and explicit evaluation/partition budgets.
- `solve_ivp`: explicit Dormand–Prince 5(4) for real non-stiff systems, forward/backward time, rejection, adaptive step sizes, endpoint derivative reuse, bounded optional trajectory.

```rust
use ruintegrate::{integrate, solve_ivp};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = integrate(|x| Ok((-x*x).exp()), -2.0, 2.0, Default::default())?;
    if !q.converged() { return Err(format!("quadrature: {:?}", q.status).into()); }
    let s = solve_ivp(|_, y, d| { d[0] = -y[0]; Ok(()) },
        0.0, 5.0, &[1.0], Default::default())?;
    if !s.reached_end() { return Err(format!("ODE: {:?}", s.status).into()); }
    println!("integral={}, y(5)={:?}", q.integral, s.state);
    Ok(())
}
```

Callbacks are explicit deterministic host functions; no implicit device readback. They can run at trial/rejected points and must not advance external simulation/training state. ODE callbacks must overwrite all derivative entries; missing/nonfinite outputs are errors.

Quadrature accepts when total estimated absolute error <= max(atol, rtol*abs(integral)). Initial cost is 15 calls, each split 30. Totals are recomputed over the current partition, so very large partition-count bookkeeping is not yet optimized. Error estimates can miss narrow/oscillatory/singular features; they are not mathematical guarantees. Nonfinite and unrepresentable-width inputs return errors rather than false zero integrals.

RK45 accepts using the maximum componentwise error divided by atol+rtol*max(abs(y),abs(y_next)). It uses the fifth-order step and embedded fourth-order error estimate, with six new derivative calls per attempt after initialization. Rejected steps do not change committed state. Workspace is reused, final-only output is the default, and saved trajectories are bounded.

Always inspect convergence/status: max steps, evaluations, output points, or representable-step limits are not success. The [extension](science-extended.md) adds first-order implicit BDF1 with optional finite-difference Jacobian, event location and infinite-domain transforms. PDE discretization, dedicated singular/oscillatory quadrature, public dense-output interpolation, complex ODE and ODE autograd remain outside this implementation.

```bash
cargo run --locked -p ruintegrate --example integrate-demo
```

Algorithm references: [SciPy RK45](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html), [Netlib QUADPACK](https://www.netlib.org/quadpack/). Implementations are independently written, not copied from those packages.
