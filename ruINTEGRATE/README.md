# ruINTEGRATE

Experimental host FP64 numerical integration, without tensor/GPU dependencies.
Includes finite-interval Gauss-Kronrod 15/7, rational infinite-domain transforms,
non-stiff Dormand-Prince RK45, adaptive backward Euler (BDF1) with Newton solves,
and sign-change event location on RK45/BDF1 accepted steps. BDF1 uses the host LU
in the local `rusolver` crate; it does not call a GPU or Python runtime.

[Extended guide (中文)](../docs/zh/libraries/science-extended.md) ·
[English guide](../docs/en/libraries/science-extended.md) ·
[Original integrator guide](../docs/en/libraries/ruintegrate.md)

Check every convergence status and set resource budgets. Error estimates and
root-location tolerances are NOT guaranteed global error bounds. BDF1 is first
order, not a variable-order BDF/Radau replacement. Tangential/multiple roots
within one step may be missed; infinite integrals must genuinely converge.
