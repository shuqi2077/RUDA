use super::*;
use super::super::constant_division::{UnsignedDivision, unsigned_division};

fn evaluate(bits: u32, n: u64, d: u64) -> (u64, u64) {
    let n = u128::from(n);
    let d = u128::from(d);
    let q = match unsigned_division(bits, d as u64).unwrap() {
        UnsignedDivision::Shift(shift) => n >> shift,
        UnsignedDivision::Compare => u128::from(n >= d),
        UnsignedDivision::MultiplyShift { multiplier, shift } =>
            ((n * u128::from(multiplier)) >> bits) >> shift,
        UnsignedDivision::Correct { multiplier } => {
            let q = (n * u128::from(multiplier)) >> bits;
            q + u128::from(n - q * d >= d)
        }
    };
    (q as u64, (n - q * d) as u64)
}

#[test]
fn unsigned_division_plans_are_exact() {
    // Exhaust every reduced-width numerator/divisor, then probe full-width
    // extremes and quotient boundaries without converting through a float.
    for d in 1..256 {
        for n in 0..256 {
            assert_eq!(evaluate(8, n, d), (n / d, n % d));
        }
    }
    for bits in [32, 64] {
        let max = if bits == 32 { u64::from(u32::MAX) } else { u64::MAX };
        let mut divisors = (1..=1024).collect::<Vec<_>>();
        for shift in 1..bits {
            let power = 1u64 << shift;
            divisors.extend([power - 1, power, power + 1]);
        }
        divisors.extend([max - 1, max]);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..1024 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            divisors.push((state & max).max(1));
        }
        for d in divisors {
            let mut numerators = vec![0, 1, d - 1, d, max - 1, max];
            for q in [1, 2, 3, max / d, (max / d).saturating_sub(1)] {
                if let Some(product) = q.checked_mul(d).filter(|p| *p <= max) {
                    numerators.push(product);
                    numerators.push(product.saturating_sub(1));
                    numerators.push(product.saturating_add(1).min(max));
                }
            }
            for _ in 0..32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                numerators.push(state & max);
            }
            for n in numerators {
                assert_eq!(evaluate(bits, n, d), (n / d, n % d), "u{bits}: {n}/{d}");
            }
        }
        assert!(unsigned_division(bits, 0).is_none());
    }
}

#[test]
fn constant_division_uses_short_integer_paths() {
    for (bits, kind) in [(32, UIntKind::U32), (64, UIntKind::U64)] {
        let ty = Type::from(kind);
        for d in [3, 5, 7, 10, 31, 96, 136, 192, (1u64 << (bits - 1)) + 1] {
            for (name, operation) in [
                ("div", Arithmetic::Div as fn(BinaryOperator) -> Arithmetic),
                ("mod", Arithmetic::Modulo), ("rem", Arithmetic::Remainder),
            ] {
                for address in [UIntKind::U32, UIntKind::U64] {
                    for inplace in [false, true] {
                        let mut k = kernel(&format!("short_{name}_{bits}_{d}_a{}_i{}",
                            if address == UIntKind::U32 {32} else {64}, u8::from(inplace)));
                        let input = buffer(&mut k, ty, Visibility::Read);
                        let output = buffer(&mut k, ty, Visibility::ReadWrite);
                        let index = Variable::builtin(Builtin::UnitPos, address.into());
                        let src = local(0, ty);
                        let dst = if inplace { src } else { local(1, ty) };
                        load(&mut k.body, input, index, src);
                        k.body.register(Instruction::new(operation(BinaryOperator {
                            lhs: src, rhs: ty.constant(ConstantValue::UInt(d)),
                        }), dst));
                        store(&mut k.body, output, index, dst);
                        let result = compile(k, ExecutionMode::Unchecked, address).unwrap();
                        let s = &result.source;
                        assert!(!s.contains("div.u"));
                        assert!(!s.contains("rem.u"));
                        assert!(!s.contains("cvt.f"));
                        match unsigned_division(bits, d).unwrap() {
                            UnsignedDivision::MultiplyShift { .. } => {
                                assert!(!s.contains("setp.ge."));
                                assert!(s.contains(&format!("shr.u{bits}")));
                                if name != "div" { assert!(s.contains(&format!("mad.lo.u{bits}"))); }
                            }
                            UnsignedDivision::Compare => assert!(!s.contains("mul.hi.")),
                            UnsignedDivision::Correct { .. } => assert!(s.contains(&format!("mad.lo.u{bits}"))),
                            UnsignedDivision::Shift(_) => unreachable!(),
                        }
                        export(&result);
                    }
                }
            }
        }
    }
}

#[test]
fn dynamic_unsigned_division_remains_dynamic() {
    for kind in [UIntKind::U32, UIntKind::U64] {
        let ty = Type::from(kind);
        let mut k = add_kernel("dynamic_unsigned_division", ty, UIntKind::U32);
        k.body.instructions[2] = Instruction::new(Arithmetic::Div(BinaryOperator {
            lhs: local(0, ty), rhs: local(1, ty),
        }), local(2, ty));
        let source = compile(k, ExecutionMode::Unchecked, UIntKind::U32).unwrap().source;
        assert!(source.contains(if kind == UIntKind::U32 { "div.u32" } else { "div.u64" }));
    }
}
