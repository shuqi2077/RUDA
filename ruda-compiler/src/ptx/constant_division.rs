//! Exact strength reduction for unsigned compile-time divisors.
//! All products are integer products. No numerator-range, alignment, or
//! fast-math promise is needed; mutable source/output registers may alias.
use super::{emit::Emitter, types::Scalar};

#[derive(Clone, Copy, Debug)]
pub(super) enum UnsignedDivision {
    Shift(u32),
    Compare,
    MultiplyShift { multiplier: u64, shift: u32 },
    Correct { multiplier: u64 },
}

/// Select using compile-time u128 arithmetic, including for U64.
/// If m = ceil(2^(bits+s)/d) and its rounding error is < 2^s, then
/// floor(n*m / 2^(bits+s)) = floor(n/d) for every bits-wide numerator.
/// Otherwise the rounded-down reciprocal needs at most one correction.
pub(super) fn unsigned_division(bits: u32, divisor: u64) -> Option<UnsignedDivision> {
    debug_assert!((1..=64).contains(&bits));
    let radix = 1u128 << bits;
    let d = u128::from(divisor) & (radix - 1);
    if d == 0 {
        return None;
    }
    if d.is_power_of_two() {
        return Some(UnsignedDivision::Shift(d.trailing_zeros()));
    }
    if d >= radix / 2 {
        return Some(UnsignedDivision::Compare);
    }
    let shift = d.ilog2();
    let scaled = radix << shift;
    let rounded = scaled / d + 1;
    if d - scaled % d < 1u128 << shift {
        Some(UnsignedDivision::MultiplyShift {
            multiplier: rounded as u64,
            shift,
        })
    } else {
        Some(UnsignedDivision::Correct { multiplier: (radix / d) as u64 })
    }
}

impl Emitter {
    pub(super) fn unsigned_constant_division(
        &mut self, ty: Scalar, destination: &str, source: &str, divisor: u64, remainder: bool,
    ) -> bool {
        let (bits, divisor) = match ty {
            Scalar::U32 => (32, u64::from(divisor as u32)),
            Scalar::U64 => (64, divisor),
            _ => return false,
        };
        let Some(plan) = unsigned_division(bits, divisor) else { return false };
        let suffix = ty.suffix();
        match plan {
            UnsignedDivision::Shift(shift) => {
                if remainder {
                    self.line(format!("and.{} {destination}, {source}, {};", ty.bits(), divisor - 1));
                } else {
                    self.line(format!("shr.{suffix} {destination}, {source}, {shift};"));
                }
            }
            UnsignedDivision::Compare => {
                // The full-width quotient can only be zero or one.
                let predicate = self.reg(Scalar::Pred);
                self.line(format!("setp.ge.{suffix} {predicate}, {source}, {divisor};"));
                if remainder {
                    let correction = self.reg(ty);
                    self.line(format!("selp.{suffix} {correction}, {divisor}, 0, {predicate};"));
                    self.line(format!("sub.{suffix} {destination}, {source}, {correction};"));
                } else {
                    self.line(format!("selp.{suffix} {destination}, 1, 0, {predicate};"));
                }
            }
            UnsignedDivision::MultiplyShift { multiplier, shift } => {
                let quotient = if remainder { self.reg(ty) } else { destination.to_owned() };
                self.line(format!("mul.hi.{suffix} {quotient}, {source}, {multiplier};"));
                self.line(format!("shr.{suffix} {quotient}, {quotient}, {shift};"));
                if remainder {
                    let negative = if bits == 32 {
                        u64::from((divisor as u32).wrapping_neg())
                    } else {
                        divisor.wrapping_neg()
                    };
                    // Low-word multiply-add is exact modulo 2^bits.
                    self.line(format!("mad.lo.{suffix} {destination}, {quotient}, {negative}, {source};"));
                }
            }
            UnsignedDivision::Correct { multiplier } => {
                let quotient = self.reg(ty);
                let residual = self.reg(ty);
                let correction = self.reg(ty);
                let predicate = self.reg(Scalar::Pred);
                let negative = if bits == 32 {
                    u64::from((divisor as u32).wrapping_neg())
                } else {
                    divisor.wrapping_neg()
                };
                self.line(format!("mul.hi.{suffix} {quotient}, {source}, {multiplier};"));
                self.line(format!("mad.lo.{suffix} {residual}, {quotient}, {negative}, {source};"));
                self.line(format!("setp.ge.{suffix} {predicate}, {residual}, {divisor};"));
                if remainder {
                    self.line(format!("selp.{suffix} {correction}, {divisor}, 0, {predicate};"));
                    self.line(format!("sub.{suffix} {destination}, {residual}, {correction};"));
                } else {
                    self.line(format!("selp.{suffix} {correction}, 1, 0, {predicate};"));
                    self.line(format!("add.{suffix} {destination}, {quotient}, {correction};"));
                }
            }
        }
        true
    }
}
