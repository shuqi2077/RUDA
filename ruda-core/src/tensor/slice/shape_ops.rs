use super::{Slice, SliceArg};
use crate::tensor::{MetadataError, Shape};
use alloc::vec::Vec;
use core::ops::Range;

/// Slice-related ops on [`Shape`]
pub trait SliceOps: Sized {
    /// Convert shape dimensions to full covering ranges (0..dim) for each dimension.
    fn into_ranges(self) -> Vec<Range<usize>>;
    /// Converts slice arguments into an array of slice specifications for the shape.
    ///
    /// This method returns an array of `Slice` objects that can be used for slicing operations.
    /// The slices are clamped to the shape's dimensions. Similar to `into_ranges()`, but
    /// allows custom slice specifications instead of full ranges.
    /// For creating complex slice specifications, use the [`s!`] macro.
    ///
    /// # Arguments
    ///
    /// * `slices` - An array of slice specifications, where each element can be:
    ///   - A range (e.g., `2..5`)
    ///   - An index
    ///   - A `Slice` object
    ///   - The output of the [`s!`] macro for advanced slicing
    ///
    /// # Behavior
    ///
    /// - Supports partial and full slicing in any number of dimensions.
    /// - Missing ranges are treated as full slices if D > D2.
    /// - Handles negative indices by wrapping around from the end of the dimension.
    /// - Clamps ranges to the shape's dimensions if they exceed the bounds.
    ///
    /// # Returns
    ///
    /// An array of `Slice` objects corresponding to the provided slice specifications,
    /// clamped to the shape's actual dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruda_core::{s, tensor::{Shape, Slice, SliceOps}};
    ///
    /// fn example() {
    ///     // 1D slicing
    ///     let slices = Shape::new([4]).into_slices(1..4);
    ///     assert_eq!(slices[0].to_range(4), 1..3);
    ///
    ///     // 2D slicing
    ///     let slices = Shape::new([3, 4]).into_slices(s![1..4, 0..2]);
    ///     assert_eq!(slices[0].to_range(3), 1..3);
    ///     assert_eq!(slices[1].to_range(4), 0..2);
    ///
    ///     // Using negative indices
    ///     let slices = Shape::new([3]).into_slices(..-2);
    ///     assert_eq!(slices[0].to_range(3), 0..1);
    ///
    ///     // Using the slice macro to select different ranges
    ///     let slices = Shape::new([2, 3, 4]).into_slices(s![.., 1..-1]);
    ///     assert_eq!(slices[0].to_range(2), 0..2);
    ///     assert_eq!(slices[1].to_range(3), 1..2);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`s!`] - The recommended macro for creating slice specifications
    /// - [`Shape::into_ranges`] - Convert to full covering ranges
    ///
    /// [`s!`]: crate::s!
    fn into_slices<S>(self, slices: S) -> Vec<Slice>
    where
        S: SliceArg;
    /// Compute the output shape from the given slices.
    fn slice(self, slices: &[Slice]) -> Result<Self, MetadataError>;
}

impl SliceOps for Shape {
    fn into_ranges(self) -> Vec<Range<usize>> {
        self.iter().map(|&d| 0..d).collect()
    }

    fn into_slices<S>(self, slices: S) -> Vec<Slice>
    where
        S: SliceArg,
    {
        slices.into_slices(&self)
    }

    fn slice(mut self, slices: &[Slice]) -> Result<Self, MetadataError> {
        if slices.len() > self.rank() {
            return Err(MetadataError::RankMismatch {
                left: self.rank(),
                right: slices.len(),
            });
        }

        slices
            .iter()
            .zip(self.iter_mut())
            .for_each(|(slice, dim_size)| *dim_size = slice.output_size(*dim_size));

        Ok(self)
    }
}

#[cfg(test)]
#[allow(clippy::identity_op, reason = "useful for clarity")]
#[path = "shape_tests.rs"]
mod tests;
