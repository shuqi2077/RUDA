use super::*;

#[test]
fn host_integer_reductions_cover_numeric_bitwise_and_wrapping_semantics() {
    let payload = i64::encode(&[2, 6, 3, 4]);
    for (operation, expected) in [
        (ReductionOperation::Sum, vec![5, 10]),
        (ReductionOperation::Product, vec![6, 24]),
        (ReductionOperation::Minimum, vec![2, 4]),
        (ReductionOperation::Maximum, vec![3, 6]),
        (ReductionOperation::BitAnd, vec![2, 4]),
        (ReductionOperation::BitOr, vec![3, 6]),
        (ReductionOperation::BitXor, vec![1, 2]),
    ] {
        let reduced = reduce_host_values::<i64>(&payload, 2, 2, operation).unwrap();
        assert_eq!(i64::decode(&reduced).unwrap(), expected);
    }

    let overflow = u8::encode(&[250, 10]);
    let reduced = reduce_host_values::<u8>(&overflow, 1, 2, ReductionOperation::Sum).unwrap();
    assert_eq!(u8::decode(&reduced).unwrap(), vec![4]);
}

#[test]
fn host_boolean_reductions_match_gloo_logical_semantics() {
    let payload = bool::encode(&[false, true, true, true]);
    for (operation, expected) in [
        (ReductionOperation::Sum, vec![true, true]),
        (ReductionOperation::Product, vec![false, true]),
        (ReductionOperation::Minimum, vec![false, true]),
        (ReductionOperation::Maximum, vec![true, true]),
        (ReductionOperation::BitAnd, vec![false, true]),
        (ReductionOperation::BitOr, vec![true, true]),
        (ReductionOperation::BitXor, vec![true, false]),
    ] {
        let reduced = reduce_host_values::<bool>(&payload, 2, 2, operation).unwrap();
        assert_eq!(bool::decode(&reduced).unwrap(), expected);
    }
    assert!(bool::decode(&[2]).is_err());
}

#[test]
fn host_f64_reductions_reject_bitwise_operations() {
    let payload = f64::encode(&[2.0, 6.0, 3.0, 4.0]);
    for (operation, expected) in [
        (ReductionOperation::Sum, vec![5.0, 10.0]),
        (ReductionOperation::Product, vec![6.0, 24.0]),
        (ReductionOperation::Minimum, vec![2.0, 4.0]),
        (ReductionOperation::Maximum, vec![3.0, 6.0]),
    ] {
        let reduced = reduce_host_values::<f64>(&payload, 2, 2, operation).unwrap();
        assert_eq!(f64::decode(&reduced).unwrap(), expected);
    }
    assert!(
        validate_host_reduction_operation(ElementType::F64, ReductionOperation::BitAnd).is_err()
    );
}

#[test]
fn host_complex_reductions_cover_sum_product_and_reject_unordered_operations() {
    let values32 = [
        Complex32 {
            real: 1.0,
            imaginary: 2.0,
        },
        Complex32 {
            real: 3.0,
            imaginary: -1.0,
        },
        Complex32 {
            real: 4.0,
            imaginary: -3.0,
        },
        Complex32 {
            real: -2.0,
            imaginary: 5.0,
        },
    ];
    let encoded32 = Complex32::encode(&values32);
    assert_eq!(Complex32::decode(&encoded32).unwrap(), values32);
    let sum32 = reduce_host_values::<Complex32>(&encoded32, 2, 2, ReductionOperation::Sum).unwrap();
    assert_eq!(
        Complex32::decode(&sum32).unwrap(),
        [
            Complex32 {
                real: 5.0,
                imaginary: -1.0,
            },
            Complex32 {
                real: 1.0,
                imaginary: 4.0,
            },
        ]
    );
    let product32 =
        reduce_host_values::<Complex32>(&encoded32, 2, 2, ReductionOperation::Product).unwrap();
    assert_eq!(
        Complex32::decode(&product32).unwrap(),
        [
            Complex32 {
                real: 10.0,
                imaginary: 5.0,
            },
            Complex32 {
                real: -1.0,
                imaginary: 17.0,
            },
        ]
    );

    let values64 = [
        Complex64 {
            real: 1.25,
            imaginary: -2.5,
        },
        Complex64 {
            real: 3.75,
            imaginary: 4.5,
        },
    ];
    let encoded64 = Complex64::encode(&values64);
    assert_eq!(Complex64::decode(&encoded64).unwrap(), values64);
    let sum64 = reduce_host_values::<Complex64>(&encoded64, 1, 2, ReductionOperation::Sum).unwrap();
    assert_eq!(
        Complex64::decode(&sum64).unwrap(),
        [Complex64 {
            real: 5.0,
            imaginary: 2.0,
        }]
    );

    for element_type in [ElementType::Complex64, ElementType::Complex128] {
        for operation in [
            ReductionOperation::Minimum,
            ReductionOperation::Maximum,
            ReductionOperation::BitAnd,
            ReductionOperation::BitOr,
            ReductionOperation::BitXor,
        ] {
            assert!(validate_host_reduction_operation(element_type, operation).is_err());
        }
    }
}

#[test]
fn f64_reductions_preserve_nan_zero_and_rank_order() {
    let nan = f64::from_bits(0x7ff8_0000_0000_0123);
    let payload = f64::encode(&[
        nan,
        -0.0,
        f64::INFINITY,
        5.0,
        2.0,
        0.0,
        f64::NEG_INFINITY,
        nan,
    ]);
    for (operation, third) in [
        (ReductionOperation::Minimum, f64::NEG_INFINITY),
        (ReductionOperation::Maximum, f64::INFINITY),
    ] {
        let output = reduce_host_values::<f64>(&payload, 4, 2, operation).unwrap();
        assert_eq!(output, f64::encode(&[nan, -0.0, third, 5.0]));
    }
    let payload = f64::encode(&[1.0e16, -1.0e16, 1.0]);
    let output = reduce_host_values::<f64>(&payload, 1, 3, ReductionOperation::Sum).unwrap();
    assert_eq!(output, f64::encode(&[1.0]));
}
