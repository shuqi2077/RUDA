use super::{Backend, Tensor, ToElement};
use alloc::vec;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Cubic Interpolate
///
/// Uses two points (x1, f1), (x2, f2) and their first derivatives g1,g2 to construct
/// a cubic interpolant and return its minimum within the given bounds.
pub(super) fn cubic_interpolate(
    x1: f64,
    f1: f64,
    g1: f64,
    x2: f64,
    f2: f64,
    g2: f64,
    bounds: Option<(f64, f64)>,
) -> f64 {
    // Compute bounds of interpolation area
    let (min_bound, max_bound) = bounds.unwrap_or(if x1 <= x2 { (x1, x2) } else { (x2, x1) });
    // Code for most common case: cubic interpolation of 2 points
    // with function and derivative values for both
    // Solution in this case (where x2 is the farthest point)
    // d1 = g1 + g2 - 3*(f1 - f2) / (x1-x2);
    // d2 = sqrt(d1^2 - g1 * g2);
    // min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    // t_new = min(max(min_pos,min_bound), max_bound);
    let d1 = g1 + g2 - 3.0 * (f1 - f2) / (x1 - x2);
    let d2_square = d1 * d1 - g1 * g2;

    if d2_square >= 0.0 {
        let d2 = d2_square.sqrt();
        let min_pos = if x1 <= x2 {
            x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2.0 * d2))
        } else {
            x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2.0 * d2))
        };
        min_pos.max(min_bound).min(max_bound)
    } else {
        (min_bound + max_bound) / 2.0
    }
}
/// Auxiliary Struct For Strong_Wolfe
struct LineSearchSample<B: Backend> {
    // step size
    t: f64,
    // loss
    f: f64,
    // gradient
    g: Tensor<B, 1>,
    // directional derivative
    gtd: f64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn strong_wolfe<B: Backend, F>(
    // obj_func(x,step size,direction) -> (loss,grad)
    obj_func: &mut F,
    x: &Tensor<B, 1>,
    // initial step size
    mut t: f64,
    d: &Tensor<B, 1>,
    f: f64,
    g: Tensor<B, 1>,
    gtd: f64,
    c1: f64,
    c2: f64,
    tolerance_change: f64,
    max_ls: usize,
) -> (f64, Tensor<B, 1>, f64, usize)
where
    F: FnMut(&Tensor<B, 1>, f64, &Tensor<B, 1>) -> (f64, Tensor<B, 1>),
{
    if max_ls == 0 {
        return (f, g, 0.0, 0);
    }
    let d_norm = d.clone().abs().max().into_scalar().to_f64();

    // evaluate objective and gradient using initial step
    let (mut f_new, mut g_new) = obj_func(x, t, d);
    let mut ls_func_evals = 1;
    let mut gtd_new = g_new.clone().dot(d.clone()).into_scalar().to_f64();

    // bracket an interval [t_prev,t] containing a point satisfying the Wolfe criteria
    let (mut t_prev, mut f_prev, mut g_prev, mut gtd_prev) = (0.0, f, g.clone(), gtd);
    let mut done = false;
    let mut ls_iter = 0;

    // the interval [low,high] using for Zoom phase
    let mut bracket: Option<[LineSearchSample<B>; 2]> = None;
    // point which satisfy the wolfe condition
    let mut wolfe_bracket: Option<LineSearchSample<B>> = None;
    loop {
        // Checking Conditions.

        // Checking the Armijo Condition and function value increasing condition.
        // Armijo: f(x+t*d) <= f(x) + c_1 t gtd
        if f_new > (f + c1 * t * gtd) || (ls_iter > 1 && f_new >= f_prev) {
            bracket = Some([
                LineSearchSample {
                    t: t_prev,
                    f: f_prev,
                    g: g_prev,
                    gtd: gtd_prev,
                },
                LineSearchSample {
                    t,
                    f: f_new,
                    g: g_new.clone(),
                    gtd: gtd_new,
                },
            ]);
            break;
        }

        // Checking Strong Wolfe Condition
        // |gtd_new| <= -c_2 gtd
        if gtd_new.abs() <= -c2 * gtd {
            wolfe_bracket = Some(LineSearchSample {
                t,
                f: f_new,
                g: g_new.clone(),
                gtd: gtd_new,
            });
            done = true;
            break;
        }

        // gtd_new >=0 , there must be a local minimum in the interval.
        if gtd_new >= 0.0 {
            bracket = Some([
                LineSearchSample {
                    t: t_prev,
                    f: f_prev,
                    g: g_prev,
                    gtd: gtd_prev,
                },
                LineSearchSample {
                    t,
                    f: f_new,
                    g: g_new.clone(),
                    gtd: gtd_new,
                },
            ]);
            break;
        }

        if ls_func_evals >= max_ls {
            break;
        }

        // interpolate
        let min_step = t + 0.01 * (t - t_prev);
        let max_step = t * 10.0;
        let t_next = cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            Some((min_step, max_step)),
        );
        t_prev = t;
        f_prev = f_new;
        g_prev = g_new;
        gtd_prev = gtd_new;

        // next step
        t = t_next;
        (f_new, g_new) = obj_func(x, t, d);
        ls_func_evals += 1;
        gtd_new = g_new.clone().dot(d.clone()).into_scalar().to_f64();
        ls_iter += 1;
    }
    if let Some(sample) = wolfe_bracket {
        return (sample.f, sample.g, sample.t, ls_func_evals);
    }

    let mut bracket = bracket.unwrap_or_else(|| {
        [
            LineSearchSample {
                t: 0.0,
                f,
                g: g.clone(),
                gtd,
            },
            LineSearchSample {
                t,
                f: f_new,
                g: g_new.clone(),
                gtd: gtd_new,
            },
        ]
    });

    // zoom phase
    let mut insuf_progress = false;

    // find high and low points in bracket
    let (mut low_idx, mut high_idx) = if bracket[0].f <= bracket[1].f {
        (0, 1)
    } else {
        (1, 0)
    };

    while !done && ls_func_evals < max_ls {
        let diff = (bracket[1].t - bracket[0].t).abs();
        // line-search bracket is so small
        if diff * d_norm < tolerance_change {
            break;
        }

        // compute new trial value
        t = cubic_interpolate(
            bracket[0].t,
            bracket[0].f,
            bracket[0].gtd,
            bracket[1].t,
            bracket[1].f,
            bracket[1].gtd,
            None,
        );

        let b_min = bracket[0].t.min(bracket[1].t);
        let b_max = bracket[0].t.max(bracket[1].t);
        let eps = 0.1 * (b_max - b_min);

        if (b_max - t).min(t - b_min) < eps {
            // interpolation close to boundary
            if insuf_progress || t >= b_max || t <= b_min {
                t = if (t - b_max).abs() < (t - b_min).abs() {
                    b_max - eps
                } else {
                    b_min + eps
                };
                insuf_progress = false;
            } else {
                insuf_progress = true;
            }
        } else {
            insuf_progress = false;
        }

        // Evaluate new point
        (f_new, g_new) = obj_func(x, t, d);

        ls_func_evals += 1;
        gtd_new = g_new.clone().dot(d.clone()).into_scalar().to_f64();

        let armijo_holds = f_new <= (f + c1 * t * gtd) && f_new < bracket[low_idx].f;

        if !armijo_holds {
            bracket[high_idx] = LineSearchSample {
                t,
                f: f_new,
                g: g_new,
                gtd: gtd_new,
            };
        } else {
            if gtd_new.abs() <= -c2 * gtd {
                return (f_new, g_new, t, ls_func_evals);
            }

            if gtd_new * (bracket[high_idx].t - bracket[low_idx].t) >= 0.0 {
                bracket[high_idx] = LineSearchSample {
                    t: bracket[low_idx].t,
                    f: bracket[low_idx].f,
                    g: bracket[low_idx].g.clone(),
                    gtd: bracket[low_idx].gtd,
                };
            }
            bracket[low_idx] = LineSearchSample {
                t,
                f: f_new,
                g: g_new,
                gtd: gtd_new,
            };
        }

        if bracket[0].f <= bracket[1].f {
            low_idx = 0;
            high_idx = 1;
        } else {
            low_idx = 1;
            high_idx = 0;
        }
    }
    // return stuff
    (
        bracket[low_idx].f,
        bracket[low_idx].g.clone(),
        bracket[low_idx].t,
        ls_func_evals,
    )
}

