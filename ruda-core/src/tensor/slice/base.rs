use super::ranges::{convert_signed_index, handle_signed_inclusive_end};
use alloc::vec::Vec;
use core::ops::Range;

/// A slice specification for a single tensor dimension.
///
/// This struct represents a range with an optional step, used for advanced indexing
/// operations on tensors. It is typically created using the [`s!`] macro rather than
/// constructed directly.
///
/// # Fields
///
/// * `start` - The starting index (inclusive). Negative values count from the end.
/// * `end` - The ending index (exclusive). `None` means to the end of the dimension.
/// * `step` - The stride between elements. Must be non-zero.
///
/// # Index Interpretation
///
/// - **Positive indices**: Count from the beginning (0-based)
/// - **Negative indices**: Count from the end (-1 is the last element)
/// - **Bounds checking**: Indices are clamped to valid ranges
///
/// # Step Behavior
///
/// - **Positive step**: Traverse forward through the range
/// - **Negative step**: Traverse backward through the range
/// - **Step size**: Determines how many elements to skip
///
/// # Examples
///
/// While you typically use the [`s!`] macro, you can also construct slices directly:
///
/// ```rust,ignore
/// use ruda_tensor::Slice;
///
/// // Equivalent to s![2..8]
/// let slice1 = Slice::new(2, Some(8), 1);
///
/// // Equivalent to s![0..10;2]
/// let slice2 = Slice::new(0, Some(10), 2);
///
/// // Equivalent to s![..;-1] (reverse)
/// let slice3 = Slice::new(0, None, -1);
/// ```
///
/// See also the [`s!`] macro for the preferred way to create slices.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Slice {
    /// Slice start index.
    pub start: isize,
    /// Slice end index (exclusive).
    pub end: Option<isize>,
    /// Step between elements (default: 1).
    pub step: isize,
}

impl Default for Slice {
    fn default() -> Self {
        Self::full()
    }
}

impl Slice {
    /// Creates a new slice with start, end, and step
    pub const fn new(start: isize, end: Option<isize>, step: isize) -> Self {
        assert!(step != 0, "Step cannot be zero");
        Self { start, end, step }
    }

    /// Creates a slice that represents the full range.
    pub const fn full() -> Self {
        Self::new(0, None, 1)
    }

    /// Creates a slice that represents a single index
    pub fn index(idx: isize) -> Self {
        Self {
            start: idx,
            end: handle_signed_inclusive_end(idx),
            step: 1,
        }
    }

    /// Converts the slice to a vector.
    pub fn into_vec(self) -> Vec<isize> {
        assert!(
            self.end.is_some(),
            "Slice must have an end to convert to a vector: {self:?}"
        );
        self.into_iter().collect()
    }

    /// Clips the slice to a maximum size.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// assert_eq!(
    ///     Slice::new(0, None, 1).bound_to(10),
    ///     Slice::new(0, Some(10), 1));
    /// assert_eq!(
    ///     Slice::new(0, Some(5), 1).bound_to(10),
    ///     Slice::new(0, Some(5), 1));
    /// assert_eq!(
    ///     Slice::new(0, None, -1).bound_to(10),
    ///     Slice::new(0, Some(-11), -1));
    /// assert_eq!(
    ///     Slice::new(0, Some(-5), -1).bound_to(10),
    ///     Slice::new(0, Some(-5), -1));
    /// ```
    pub fn bound_to(self, size: usize) -> Self {
        let mut bounds = size as isize;

        if let Some(end) = self.end {
            if end > 0 {
                bounds = end.min(bounds);
            } else {
                bounds = end.max(-(bounds + 1));
            }
        } else if self.is_reversed() {
            bounds = -(bounds + 1);
        }

        Self {
            end: Some(bounds),
            ..self
        }
    }

    /// Creates a slice with a custom step
    pub fn with_step(start: isize, end: Option<isize>, step: isize) -> Self {
        assert!(step != 0, "Step cannot be zero");
        Self { start, end, step }
    }

    /// Creates a slice from a range with a specified step
    pub fn from_range_stepped<R: Into<Slice>>(range: R, step: isize) -> Self {
        assert!(step != 0, "Step cannot be zero");
        let mut slice = range.into();
        slice.step = step;
        slice
    }

    /// Returns the step of the slice
    pub fn step(&self) -> isize {
        self.step
    }

    /// Returns the range for this slice given a dimension size
    pub fn range(&self, size: usize) -> Range<usize> {
        self.to_range(size)
    }

    /// Convert this slice to a range for a dimension of the given size.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the dimension to slice.
    ///
    /// # Returns
    ///
    /// A `Range<usize>` representing the slice bounds.
    pub fn to_range(&self, size: usize) -> Range<usize> {
        // Always return a valid range with start <= end
        // The step information will be handled separately
        let start = convert_signed_index(self.start, size);
        let end = match self.end {
            Some(end) => convert_signed_index(end, size),
            None => size,
        };
        start..end
    }

    /// Converts the slice into a range and step tuple
    pub fn to_range_and_step(&self, size: usize) -> (Range<usize>, isize) {
        let range = self.to_range(size);
        (range, self.step)
    }

    /// Returns true if the step is negative
    pub fn is_reversed(&self) -> bool {
        self.step < 0
    }

    /// Calculates the output size for this slice operation
    pub fn output_size(&self, dim_size: usize) -> usize {
        let range = self.to_range(dim_size);
        // Handle empty slices (start >= end)
        if range.start >= range.end {
            return 0;
        }
        let len = range.end - range.start;
        if self.step.unsigned_abs() == 1 {
            len
        } else {
            len.div_ceil(self.step.unsigned_abs())
        }
    }
}
