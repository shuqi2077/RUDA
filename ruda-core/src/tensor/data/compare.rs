use super::{TensorData, Tolerance};
use crate::tensor::{BoolStore, DType};
use crate::tensor::element::{Element, ElementOrdered};
use alloc::format;
use alloc::string::String;
use half::{bf16, f16};
use num_traits::{Float, ToPrimitive};

impl TensorData {
    /// Asserts the data is equal to another data.
    ///
    /// # Arguments
    ///
    /// * `other` - The other data.
    /// * `strict` - If true, the data types must the be same.
    ///   Otherwise, the comparison is done in the current data type.
    ///
    /// # Panics
    ///
    /// Panics if the data is not equal.
    #[track_caller]
    pub fn assert_eq(&self, other: &Self, strict: bool) {
        if strict {
            assert_eq!(
                self.dtype, other.dtype,
                "Data types differ ({:?} != {:?})",
                self.dtype, other.dtype
            );
        }

        match self.dtype {
            DType::F64 => self.assert_eq_elem::<f64>(other),
            DType::F32 | DType::Flex32 => self.assert_eq_elem::<f32>(other),
            DType::F16 => self.assert_eq_elem::<f16>(other),
            DType::BF16 => self.assert_eq_elem::<bf16>(other),
            DType::I64 => self.assert_eq_elem::<i64>(other),
            DType::I32 => self.assert_eq_elem::<i32>(other),
            DType::I16 => self.assert_eq_elem::<i16>(other),
            DType::I8 => self.assert_eq_elem::<i8>(other),
            DType::U64 => self.assert_eq_elem::<u64>(other),
            DType::U32 => self.assert_eq_elem::<u32>(other),
            DType::U16 => self.assert_eq_elem::<u16>(other),
            DType::U8 => self.assert_eq_elem::<u8>(other),
            DType::Bool(BoolStore::Native) => self.assert_eq_elem::<bool>(other),
            DType::Bool(BoolStore::U8) => self.assert_eq_elem::<u8>(other),
            DType::Bool(BoolStore::U32) => self.assert_eq_elem::<u32>(other),
            DType::QFloat(q) => {
                // Strict or not, it doesn't make sense to compare quantized data to not quantized data for equality
                let q_other = if let DType::QFloat(q_other) = other.dtype {
                    q_other
                } else {
                    panic!("Quantized data differs from other not quantized data")
                };

                // Data equality mostly depends on input quantization type, but we also check level
                if q.value == q_other.value && q.level == q_other.level {
                    self.assert_eq_elem::<i8>(other)
                } else {
                    panic!("Quantization schemes differ ({q:?} != {q_other:?})")
                }
            }
        }
    }

    #[track_caller]
    fn assert_eq_elem<E: Element>(&self, other: &Self) {
        let mut message = String::new();
        if self.shape != other.shape {
            message += format!(
                "\n  => Shape is different: {:?} != {:?}",
                self.shape, other.shape
            )
            .as_str();
        }

        let mut num_diff = 0;
        let max_num_diff = 5;
        for (i, (a, b)) in self.iter::<E>().zip(other.iter::<E>()).enumerate() {
            if !a.eq(&b) {
                // Only print the first 5 different values.
                if num_diff < max_num_diff {
                    message += format!("\n  => Position {i}: {a} != {b}").as_str();
                }
                num_diff += 1;
            }
        }

        if num_diff >= max_num_diff {
            message += format!("\n{} more errors...", num_diff - max_num_diff).as_str();
        }

        if !message.is_empty() {
            panic!("Tensors are not eq:{message}");
        }
    }

    /// Asserts the data is approximately equal to another data.
    ///
    /// # Arguments
    ///
    /// * `other` - The other data.
    /// * `tolerance` - The tolerance of the comparison.
    ///
    /// # Panics
    ///
    /// Panics if the data is not approximately equal.
    #[track_caller]
    pub fn assert_approx_eq<F: Float + Element>(&self, other: &Self, tolerance: Tolerance<F>) {
        let mut message = String::new();
        if self.shape != other.shape {
            message += format!(
                "\n  => Shape is different: {:?} != {:?}",
                self.shape, other.shape
            )
            .as_str();
        }

        let iter = self.iter::<F>().zip(other.iter::<F>());

        let mut num_diff = 0;
        let max_num_diff = 5;

        for (i, (a, b)) in iter.enumerate() {
            //if they are both nan, then they are equally nan
            let both_nan = a.is_nan() && b.is_nan();
            //this works for both infinities
            let both_inf =
                a.is_infinite() && b.is_infinite() && ((a > F::zero()) == (b > F::zero()));

            if both_nan || both_inf {
                continue;
            }

            if !tolerance.approx_eq(F::from(a).unwrap(), F::from(b).unwrap()) {
                // Only print the first 5 different values.
                if num_diff < max_num_diff {
                    let diff_abs = ToPrimitive::to_f64(&(a - b).abs()).unwrap();
                    let max = F::max(a.abs(), b.abs());
                    let diff_rel = diff_abs / ToPrimitive::to_f64(&max).unwrap();

                    let tol_rel = ToPrimitive::to_f64(&tolerance.relative).unwrap();
                    let tol_abs = ToPrimitive::to_f64(&tolerance.absolute).unwrap();

                    message += format!(
                        "\n  => Position {i}: {a} != {b}\n     diff (rel = {diff_rel:+.2e}, abs = {diff_abs:+.2e}), tol (rel = {tol_rel:+.2e}, abs = {tol_abs:+.2e})"
                    )
                    .as_str();
                }
                num_diff += 1;
            }
        }

        if num_diff >= max_num_diff {
            message += format!("\n{} more errors...", num_diff - 5).as_str();
        }

        if !message.is_empty() {
            panic!("Tensors are not approx eq:{message}");
        }
    }

    /// Asserts each value is within a given range.
    ///
    /// # Arguments
    ///
    /// * `range` - The range.
    ///
    /// # Panics
    ///
    /// If any value is not within the half-open range bounded inclusively below
    /// and exclusively above (`start..end`).
    pub fn assert_within_range<E: ElementOrdered>(&self, range: core::ops::Range<E>) {
        for elem in self.iter::<E>() {
            if elem.cmp(&range.start).is_lt() || elem.cmp(&range.end).is_ge() {
                panic!("Element ({elem:?}) is not within range {range:?}");
            }
        }
    }

    /// Asserts each value is within a given inclusive range.
    ///
    /// # Arguments
    ///
    /// * `range` - The range.
    ///
    /// # Panics
    ///
    /// If any value is not within the half-open range bounded inclusively (`start..=end`).
    pub fn assert_within_range_inclusive<E: ElementOrdered>(
        &self,
        range: core::ops::RangeInclusive<E>,
    ) {
        let start = range.start();
        let end = range.end();

        for elem in self.iter::<E>() {
            if elem.cmp(start).is_lt() || elem.cmp(end).is_gt() {
                panic!("Element ({elem:?}) is not within range {range:?}");
            }
        }
    }
}

#[cfg(test)]
#[path = "compare_tests.rs"]
mod tests;
