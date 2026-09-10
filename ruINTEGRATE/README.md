# ruINTEGRATE

**English** | [简体中文](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/ruintegrate.md)

Numerical integration and ordinary differential equation solvers for Ruda.

- Cargo package: `ruintegrate`
- Rust crate: `ruintegrate`

This crate is part of the RUDA workspace. Build from the [RUDA workspace](https://github.com/shuqi2077/RUDA), which provides shared configuration and the local `rusolver` dependency.

## Features

| Interface | Operations |
| --- | --- |
| `integrate` | Adaptive finite-interval Gauss-Kronrod 15/7 quadrature |
| `integrate_infinite` | Half-infinite and whole-line integration through rational transformations |
| `solve_ivp` | Dormand-Prince 5(4), or RK45, for non-stiff initial-value ODEs |
| `solve_bdf1` | Adaptive backward Euler with Newton solves for stiff systems |
| `solve_ivp_events`, `solve_bdf1_events` | Sign-change event location on accepted ODE steps |

All interfaces use host FP64. The library is experimental and has no tensor, GPU, or Python runtime dependency. BDF1 uses host LU from `rusolver`. The default Cargo feature set is empty; the interfaces above do not require additional features.

## Quick Start

Build and run from the RUDA workspace:

```sh
git clone https://github.com/shuqi2077/RUDA.git
cd RUDA
cargo build --locked -p ruintegrate
cargo run --locked -p ruintegrate --example integrate-demo
```

## Documentation

- [User guide](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/ruintegrate.md)
- [Extended numerical science](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/science-extended.md)
- [Cargo configuration](https://github.com/shuqi2077/RUDA/blob/main/ruINTEGRATE/Cargo.toml) · [Module exports](https://github.com/shuqi2077/RUDA/blob/main/ruINTEGRATE/src/lib.rs)

## ruINTEGRATE User Guide

[Compute libraries](https://github.com/shuqi2077/RUDA/blob/main/docs/en/libraries/README.md) · [中文](https://github.com/shuqi2077/RUDA/blob/main/docs/zh/libraries/science-extended.md)

### 1. Integrate a function and solve an ODE

This example integrates a Gaussian over a finite interval and solves `y' = -y` with `y(0) = 1`:

```rust
use ruintegrate::{integrate, solve_ivp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = integrate(|x| Ok((-x * x).exp()), -2.0, 2.0, Default::default())?;
    if !q.converged() {
        return Err(format!("quadrature stopped: {:?}", q.status).into());
    }

    let s = solve_ivp(
        |_, y, derivative| { derivative[0] = -y[0]; Ok(()) },
        0.0, 5.0, &[1.0], Default::default(),
    )?;
    if !s.reached_end() {
        return Err(format!("ODE stopped: {:?}", s.status).into());
    }

    println!("integral = {}, estimated error = {}",
        q.integral, q.estimated_absolute_error);
    println!("y(5) = {:?}", s.state);
    Ok(())
}
```

Callbacks are explicit host functions. They may run at trial points belonging to rejected steps, so they must not advance external simulation or training state. ODE callbacks must overwrite every derivative entry with a finite value. There is no implicit device readback.

### 2. Finite and infinite intervals

`integrate` bisects the interval with the largest estimated error. It accepts when the total estimated absolute error is at most `max(atol, rtol * abs(integral))`. `QuadratureOptions` sets absolute/relative tolerances, evaluation limits, and partition limits.

The initial interval costs 15 function evaluations; each split adds 30. Reversed and zero-length finite intervals are supported. Nonfinite values and unrepresentable interval widths return errors rather than a false zero integral.

`integrate_infinite` transforms half-infinite or whole-line domains to finite intervals. For `InfiniteInterval::WholeLine { split }`, the two tails are checked separately; cancellation of divergent tails is not accepted as a Cauchy principal value.

Error estimates are not guaranteed bounds. Narrow peaks, singularities, and oscillations can be missed; infinite integrals must genuinely converge. Dedicated singular, oscillatory, and principal-value quadrature methods are not provided.

Source: [Finite quadrature](https://github.com/shuqi2077/RUDA/blob/main/ruINTEGRATE/src/quadrature.rs) and [infinite-domain transforms](https://github.com/shuqi2077/RUDA/blob/main/ruINTEGRATE/src/improper.rs).

### 3. Non-stiff and stiff ODEs

| Solver | Method and options |
| --- | --- |
| `solve_ivp` | Explicit RK45 with adaptive step size; configure `Rk45Options` |
| `solve_bdf1` | First-order implicit Euler/BDF1 with Newton iteration and backtracking; configure `Bdf1Options` |

RK45 uses the fifth-order solution and an embedded fourth-order error estimate. Its acceptance scale is `atol + rtol * max(abs(y), abs(y_next))` per component, with the maximum normalized error across components. Rejected steps do not update the committed state.

BDF1 compares one full step with two half steps and accepts the two-half-step result without extrapolation. Supply an analytic Jacobian or use finite differences. Analytic Jacobian callbacks must fill all finite row-major matrix entries. Newton systems use dense host LU; the default maximum dimension is 256. This is not a variable-order BDF or Radau solver.

Both solvers support forward/backward integration and step/evaluation budgets. Final-state-only output is the default. Enable `save_trajectory` for bounded trajectory storage. Reaching a step, evaluation, or output limit does not mean the endpoint was reached.

Source: [RK45](https://github.com/shuqi2077/RUDA/blob/main/ruINTEGRATE/src/ode.rs) and [BDF1](https://github.com/shuqi2077/RUDA/blob/main/ruINTEGRATE/src/stiff.rs).

### 4. Events and convergence status

`solve_ivp_events` and `solve_bdf1_events` accept event callbacks and `EventSpec` entries. Each specification selects `Any`, `Increasing`, or `Decreasing` direction and whether the event terminates integration. Direction is defined in physical time, including when integrating backward.

Event roots are located using cubic Hermite interpolation of accepted-step endpoint states/derivatives and bisection. Initial exact zeros are reported. Tangencies or multiple crossings within one step may be missed; root-location tolerance is not a global ODE solution error bound.

Inspect the returned status: `QuadratureReport::converged()` checks quadrature, and `OdeReport::reached_end()` checks endpoint completion. A terminal event is reported as `OdeStatus::Event`, not endpoint completion. Event reports contain the located times, states, and event indices.

The [advanced example](https://github.com/shuqi2077/RUDA/blob/main/ruINTEGRATE/examples/advanced_integrate.rs) combines whole-line Gaussian integration, BDF1, and a terminating oscillator event:

```sh
cargo run --locked -p ruintegrate --example advanced-integrate
```

Source: [Event interfaces](https://github.com/shuqi2077/RUDA/blob/main/ruINTEGRATE/src/events.rs).

### 5. Execution scope

ruINTEGRATE operates on real FP64 host values and callback functions. It does not provide GPU integration, complex ODEs, automatic differentiation, PDE discretization, or a public dense-output interpolation interface. It is not a complete SciPy/GSL API replacement.
