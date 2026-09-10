use num_traits::{Float, ToPrimitive};

/// The tolerance used to compare to floating point numbers.
///
/// Generally, two numbers `x` and `y` are approximately equal if
///
/// ```text
/// |x - y| < max(R * (|x + y|), A)
/// ```
///
/// where `R` is the relative tolerance and `A` is the absolute tolerance.
///
///
/// The most common way to initialize this struct is to use `Tolerance::<F>::default()`.
/// In that case, the relative and absolute tolerances are computed using an heuristic based
/// on the EPSILON and MIN_POSITIVE values of the given floating point type `F`.
///
/// Another common initialization is `Tolerance::<F>::rel_abs(1e-4, 1e-5).set_half_precision_relative(1e-2)`.
/// This will use a sane default to manage values too close to 0.0 and
/// use different relative tolerances depending on the floating point precision.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance<F> {
    pub(super) relative: F,
    pub(super) absolute: F,
}

impl<F: Float> Default for Tolerance<F> {
    fn default() -> Self {
        Self::balanced()
    }
}

impl<F: Float> Tolerance<F> {
    /// Create a tolerance with strict precision setting.
    pub fn strict() -> Self {
        Self {
            relative: F::from(0.00).unwrap(),
            absolute: F::from(64).unwrap() * F::min_positive_value(),
        }
    }
    /// Create a tolerance with balanced precision setting.
    pub fn balanced() -> Self {
        Self {
            relative: F::from(0.005).unwrap(), // 0.5%
            absolute: F::from(1e-5).unwrap(),
        }
    }

    /// Create a tolerance with permissive precision setting.
    pub fn permissive() -> Self {
        Self {
            relative: F::from(0.01).unwrap(), // 1.0%
            absolute: F::from(0.01).unwrap(),
        }
    }
    /// When comparing two numbers, this uses both the relative and absolute differences.
    ///
    /// That is, `x` and `y` are approximately equal if
    ///
    /// ```text
    /// |x - y| < max(R * (|x + y|), A)
    /// ```
    ///
    /// where `R` is the `relative` tolerance and `A` is the `absolute` tolerance.
    pub fn rel_abs<FF: ToPrimitive>(relative: FF, absolute: FF) -> Self {
        let relative = Self::check_relative(relative);
        let absolute = Self::check_absolute(absolute);

        Self { relative, absolute }
    }

    /// When comparing two numbers, this uses only the relative difference.
    ///
    /// That is, `x` and `y` are approximately equal if
    ///
    /// ```text
    /// |x - y| < R * max(|x|, |y|)
    /// ```
    ///
    /// where `R` is the relative `tolerance`.
    pub fn relative<FF: ToPrimitive>(tolerance: FF) -> Self {
        let relative = Self::check_relative(tolerance);

        Self {
            relative,
            absolute: F::from(0.0).unwrap(),
        }
    }

    /// When comparing two numbers, this uses only the absolute difference.
    ///
    /// That is, `x` and `y` are approximately equal if
    ///
    /// ```text
    /// |x - y| < A
    /// ```
    ///
    /// where `A` is the absolute `tolerance`.
    pub fn absolute<FF: ToPrimitive>(tolerance: FF) -> Self {
        let absolute = Self::check_absolute(tolerance);

        Self {
            relative: F::from(0.0).unwrap(),
            absolute,
        }
    }

    /// Change the relative tolerance to the given one.
    pub fn set_relative<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        self.relative = Self::check_relative(tolerance);
        self
    }

    /// Change the relative tolerance to the given one only if `F` is half precision.
    pub fn set_half_precision_relative<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 2 {
            self.relative = Self::check_relative(tolerance);
        }
        self
    }

    /// Change the relative tolerance to the given one only if `F` is single precision.
    pub fn set_single_precision_relative<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 4 {
            self.relative = Self::check_relative(tolerance);
        }
        self
    }

    /// Change the relative tolerance to the given one only if `F` is double precision.
    pub fn set_double_precision_relative<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 8 {
            self.relative = Self::check_relative(tolerance);
        }
        self
    }

    /// Change the absolute tolerance to the given one.
    pub fn set_absolute<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        self.absolute = Self::check_absolute(tolerance);
        self
    }

    /// Change the absolute tolerance to the given one only if `F` is half precision.
    pub fn set_half_precision_absolute<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 2 {
            self.absolute = Self::check_absolute(tolerance);
        }
        self
    }

    /// Change the absolute tolerance to the given one only if `F` is single precision.
    pub fn set_single_precision_absolute<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 4 {
            self.absolute = Self::check_absolute(tolerance);
        }
        self
    }

    /// Change the absolute tolerance to the given one only if `F` is double precision.
    pub fn set_double_precision_absolute<FF: ToPrimitive>(mut self, tolerance: FF) -> Self {
        if core::mem::size_of::<F>() == 8 {
            self.absolute = Self::check_absolute(tolerance);
        }
        self
    }

    /// Checks if `x` and `y` are approximately equal given the tolerance.
    pub fn approx_eq(&self, x: F, y: F) -> bool {
        // See the accepted answer here
        // https://stackoverflow.com/questions/4915462/how-should-i-do-floating-point-comparison

        // This also handles the case where both a and b are infinity so that we don't need
        // to manage it in the rest of the function.
        if x == y {
            return true;
        }

        let diff = (x - y).abs();
        let max = F::max(x.abs(), y.abs());

        diff < self.absolute.max(self.relative * max)
    }

    fn check_relative<FF: ToPrimitive>(tolerance: FF) -> F {
        let tolerance = F::from(tolerance).unwrap();
        assert!(tolerance <= F::one());
        tolerance
    }

    fn check_absolute<FF: ToPrimitive>(tolerance: FF) -> F {
        let tolerance = F::from(tolerance).unwrap();
        assert!(tolerance >= F::zero());
        tolerance
    }
}
