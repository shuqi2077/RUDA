// SPDX-License-Identifier: Apache-2.0
use std::{
    cmp::Ordering, collections::BinaryHeap
};
use crate::IntegrationError;
use crate::error::{
    checked, validate_tolerance
};
#[derive(Clone, Copy, Debug)]
pub struct QuadratureOptions {
    pub absolute_tolerance: f64, pub relative_tolerance: f64,
    pub max_intervals: usize, pub max_evaluations: usize,
}
impl Default for QuadratureOptions {
    fn default()->Self{
        Self{
            absolute_tolerance: 1e-10, relative_tolerance: 1e-8, max_intervals: 4096, max_evaluations: 200_000
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuadratureStatus {
    Converged, MaxIntervals, MaxEvaluations, RoundoffLimit
}
#[derive(Clone, Debug)]
pub struct QuadratureReport {
    pub integral: f64, pub estimated_absolute_error: f64, pub evaluations: usize,
    pub intervals: usize, pub status: QuadratureStatus,
}
impl QuadratureReport {
    pub fn converged(&self)->bool{
        self.status==QuadratureStatus::Converged
    }
}
#[derive(Clone, Debug)]
struct Interval{
    a: f64, b: f64, value: f64, error: f64, id: usize
}
impl PartialEq for Interval{
    fn eq(&self, other: &Self)->bool{
        self.error.to_bits()==other.error.to_bits()&&self.id==other.id
    }
}
impl Eq for Interval{
}
impl PartialOrd for Interval{
    fn partial_cmp(&self, other: &Self)->Option<Ordering>{
        Some(self.cmp(other))
    }
}
impl Ord for Interval{
    fn cmp(&self, other: &Self)->Ordering{
        self.error.total_cmp(&other.error).then_with(||self.id.cmp(&other.id))
    }
}
// Standard mathematical Gauss-Kronrod 15/7 abscissae and weights. The algorithm
// here is independently implemented; it is not a copy of GSL/QUADPACK source.
const X: [f64; 8]=[0.9914553711208126, 0.9491079123427585, 0.8648644233597691,
0.7415311855993945, 0.5860872354676911, 0.4058451513773972, 0.2077849550078985, 0.0];
const WK: [f64; 8]=[0.022935322010529225, 0.06309209262997855, 0.10479001032225018,
0.14065325971552592, 0.1690047266392679, 0.19035057806478542, 0.20443294007529889, 0.20948214108472782];
const WG: [f64; 4]=[0.1294849661688697, 0.27970539148927667, 0.38183005050511894, 0.4179591836734694];
fn evaluate(f: &mut impl FnMut(f64)->Result<f64, IntegrationError>, x: f64)->Result<f64, IntegrationError>{
    let value=f(x)?;
    if !value.is_finite(){
        Err(IntegrationError::NonFiniteFunction{
            at: x, index: 0
        })
    } else{
        Ok(value)
    }
}
fn rule(f: &mut impl FnMut(f64)->Result<f64, IntegrationError>, a: f64, b: f64, id: usize)->Result<Interval, IntegrationError>{
    let center=a+(b-a)*0.5;
    let half=(b-a)*0.5;
    if half==0.0 {
        return Err(IntegrationError::Arithmetic("interval half-width underflows; rescale the integration variable"));
    }
    let fc=evaluate(f, center)?;
    let mut left=[0.0; 7];
    let mut right=[0.0; 7];
    // Weighted values are formed separately to avoid overflow in f(left)+f(right).
    let mut k=WK[7]*fc;
    let mut g=WG[3]*fc;
    let mut abs=WK[7]*fc.abs();
    for i in 0..7{
        left[i]=evaluate(f, center-half*X[i])?;
        right[i]=evaluate(f, center+half*X[i])?;
        k=checked(k+WK[i]*left[i]+WK[i]*right[i], "quadrature weighted sum")?;
        abs=checked(abs+WK[i]*left[i].abs()+WK[i]*right[i].abs(), "quadrature absolute sum")?;
        if i%2==1{
            g=checked(g+WG[i/2]*left[i]+WG[i/2]*right[i], "Gauss weighted sum")?;
        }
    }
    let mean=0.5*k;
    let mut asc=WK[7]*(fc-mean).abs();
    for i in 0..7{
        asc=checked(asc+WK[i]*(left[i]-mean).abs()+WK[i]*(right[i]-mean).abs(), "quadrature deviation")?;
    }
    let value=checked(half*k, "quadrature integral")?;
    let abs=checked(half*abs, "quadrature absolute integral")?;
    let asc=checked(half*asc, "quadrature deviation integral")?;
    let mut error=checked((half*(k-g)).abs(), "quadrature error estimate")?;
    if asc>0.0&&error>0.0{
        error=asc*(200.0*(error/asc)).powf(1.5).min(1.0);
    }
    error=error.max(50.0*f64::EPSILON*abs);
    Ok(Interval{
        a, b, value, error, id
    })
}
fn totals(heap: &BinaryHeap<Interval>)->Result<(f64, f64), IntegrationError>{
    // Recompute rather than repeatedly subtract nearly equal leaf estimates.
    let(mut sum, mut correction, mut error)=(0.0f64, 0.0f64, 0.0f64);
    for leaf in heap{
        let t=checked(sum+leaf.value, "quadrature total")?;
        correction+=if sum.abs()>=leaf.value.abs(){
            (sum-t)+leaf.value
        } else{
            (leaf.value-t)+sum
        };
        sum=t;
        error=checked(error+leaf.error, "quadrature total error")?;
    }
    Ok((checked(sum+correction, "quadrature compensated total")?, error))
}
/// Adaptive finite-interval integral. The callback is evaluated 15 times initially
/// and 30 times per bisection. Largest estimated-error interval is refined first.
/// Reversed bounds negate the result; identical bounds return zero with no calls.
/// On resource/roundoff exhaustion the report preserves the best partition and
/// explicitly marks non-convergence. Singular/infinite/highly oscillatory problems
/// require a more specialized method; the estimator can miss narrow features.
pub fn integrate(mut f: impl FnMut(f64)->Result<f64, IntegrationError>, a: f64, b: f64,
options: QuadratureOptions)->Result<QuadratureReport, IntegrationError>{
    validate_tolerance(options.absolute_tolerance, options.relative_tolerance)?;
    if options.max_intervals==0||options.max_evaluations<15{
        return Err(IntegrationError::InvalidOption("need at least one interval and 15 evaluations"));
    }
    if !a.is_finite()||!b.is_finite()||!(b-a).is_finite(){
        return Err(IntegrationError::NonFiniteInput("finite representable interval required"));
    }
    if a==b{
        return Ok(QuadratureReport{
            integral: 0.0, estimated_absolute_error: 0.0, evaluations: 0, intervals: 0, status: QuadratureStatus::Converged
        });
    }
    let (a, b, sign)=if a<b{
        (a, b, 1.0)
    } else{
        (b, a, -1.0)
    };
    let mut heap=BinaryHeap::new();
    heap.try_reserve(1).map_err(|_|IntegrationError::Allocation)?;
    heap.push(rule(&mut f, a, b, 0)?);
    let(mut evaluations, mut next_id)=(15usize, 1usize);
    loop{
        let(total, error)=totals(&heap)?;
        let tolerance=checked(options.absolute_tolerance.max(options.relative_tolerance*total.abs()), "quadrature tolerance")?;
        let status=if error<=tolerance{
            Some(QuadratureStatus::Converged)
        }
        else if heap.len()>=options.max_intervals{
            Some(QuadratureStatus::MaxIntervals)
        }
        else if options.max_evaluations.saturating_sub(evaluations)<30{
            Some(QuadratureStatus::MaxEvaluations)
        } else{
            None
        };
        if let Some(status)=status{
            return Ok(QuadratureReport{
                integral: sign*total, estimated_absolute_error: error, evaluations, intervals: heap.len(), status
            });
        }
        let largest=heap.pop().ok_or(IntegrationError::Arithmetic("empty quadrature partition"))?;
        let mid=largest.a+(largest.b-largest.a)*0.5;
        if mid==largest.a||mid==largest.b{
            heap.push(largest);
            return Ok(QuadratureReport{
                integral: sign*total, estimated_absolute_error: error, evaluations, intervals: heap.len(), status: QuadratureStatus::RoundoffLimit
            });
        }
        heap.try_reserve(2).map_err(|_|IntegrationError::Allocation)?;
        let l=rule(&mut f, largest.a, mid, next_id)?;
        let r=rule(&mut f, mid, largest.b, next_id+1)?;
        next_id=next_id.checked_add(2).ok_or(IntegrationError::Arithmetic("interval id overflow"))?;
        heap.push(l);
        heap.push(r);
        evaluations+=30;
    }
}
