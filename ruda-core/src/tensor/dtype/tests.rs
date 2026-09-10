use super::*;

#[test]
fn finfo_f32() {
    let info = FloatDType::F32.finfo();
    assert_eq!(info.epsilon, f32::EPSILON as f64);
    assert_eq!(info.max, f32::MAX as f64);
    assert_eq!(info.min, f32::MIN as f64);
    assert_eq!(info.min_positive, f32::MIN_POSITIVE as f64);
}

#[test]
fn finfo_f64() {
    let info = FloatDType::F64.finfo();
    assert_eq!(info.epsilon, f64::EPSILON);
    assert_eq!(info.max, f64::MAX);
    assert_eq!(info.min, f64::MIN);
    assert_eq!(info.min_positive, f64::MIN_POSITIVE);
}

#[test]
fn finfo_f16() {
    let info = FloatDType::F16.finfo();
    assert_eq!(info.epsilon, f16::EPSILON.to_f64_const());
    assert!(info.epsilon > 0.0);
    assert!(info.min_positive > 0.0);
    // f16 epsilon is much larger than f32
    assert!(info.epsilon > FloatDType::F32.finfo().epsilon);
}

#[test]
fn finfo_bf16() {
    let info = FloatDType::BF16.finfo();
    assert_eq!(info.epsilon, bf16::EPSILON.to_f64_const());
    assert!(info.epsilon > 0.0);
    assert!(info.min_positive > 0.0);
    // bf16 epsilon is larger than f32 (fewer mantissa bits)
    assert!(info.epsilon > FloatDType::F32.finfo().epsilon);
}

#[test]
fn finfo_flex32_uses_f16_limits() {
    let flex = FloatDType::Flex32.finfo();
    let f16_info = FloatDType::F16.finfo();
    assert_eq!(flex.epsilon, f16_info.epsilon);
    assert_eq!(flex.min_positive, f16_info.min_positive);
}

#[test]
fn dtype_finfo_delegates_to_float_dtype() {
    assert_eq!(DType::F32.finfo(), Some(FloatDType::F32.finfo()));
    assert_eq!(DType::F64.finfo(), Some(FloatDType::F64.finfo()));
    assert_eq!(DType::F16.finfo(), Some(FloatDType::F16.finfo()));
    assert_eq!(DType::BF16.finfo(), Some(FloatDType::BF16.finfo()));
    assert_eq!(DType::Flex32.finfo(), Some(FloatDType::Flex32.finfo()));
}

#[test]
fn dtype_finfo_returns_none_for_non_float() {
    assert!(DType::I32.finfo().is_none());
    assert!(DType::U8.finfo().is_none());
    assert!(DType::Bool(BoolStore::Native).finfo().is_none());
}

#[test]
fn finfo_invariants() {
    for dtype in [
        FloatDType::F64,
        FloatDType::F32,
        FloatDType::F16,
        FloatDType::BF16,
        FloatDType::Flex32,
    ] {
        let info = dtype.finfo();
        assert!(info.epsilon > 0.0, "{dtype:?}: epsilon must be positive");
        assert!(
            info.min_positive > 0.0,
            "{dtype:?}: min_positive must be positive"
        );
        assert!(info.max > 0.0, "{dtype:?}: max must be positive");
        assert!(info.min < 0.0, "{dtype:?}: min must be negative");
        assert!(
            info.max > info.min_positive,
            "{dtype:?}: max > min_positive"
        );
    }
}
