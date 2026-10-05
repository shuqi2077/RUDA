//! Checked leading-axis layouts shared by tensor collective transports.
use super::shape::{MetadataError, Shape};

/// Logical input and rank-ordered output of an equal-size collective.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveShape {
    /// Each rank's original tensor shape.
    pub input: Shape,
    /// Gathered shape or this rank's contiguous reduced shard.
    pub output: Shape,
    /// Logical input element count, independent of view strides.
    pub input_elements: usize,
    /// Logical output element count.
    pub output_elements: usize,
}
impl CollectiveShape {
    /// Concatenate each rank's tensor along axis zero, in rank order.
    pub fn all_gather(input: Shape, world_size: usize) -> Result<Self, MetadataError> {
        validate(&input, world_size)?;
        let mut output = input.clone();
        output[0] = output[0]
            .checked_mul(world_size)
            .ok_or_else(|| invalid("all-gather leading axis overflow"))?;
        Self::new(input, output)
    }
    /// Partition a reduced tensor into equal contiguous axis-zero shards.
    /// The leading axis, not just the total volume, must divide by world size.
    pub fn reduce_scatter(input: Shape, world_size: usize) -> Result<Self, MetadataError> {
        validate(&input, world_size)?;
        if !input[0].is_multiple_of(world_size) {
            return Err(invalid(
                "reduce-scatter leading axis must be divisible by world size",
            ));
        }
        let mut output = input.clone();
        output[0] /= world_size;
        Self::new(input, output)
    }
    fn new(input: Shape, output: Shape) -> Result<Self, MetadataError> {
        Ok(Self {
            input_elements: elements(&input)?,
            output_elements: elements(&output)?,
            input,
            output,
        })
    }
}
fn validate(shape: &Shape, world_size: usize) -> Result<(), MetadataError> {
    if world_size == 0 {
        return Err(invalid("collective world size must be positive"));
    }
    if shape.rank() == 0 {
        return Err(invalid("collective tensor requires a leading axis"));
    }
    Ok(())
}
fn elements(shape: &Shape) -> Result<usize, MetadataError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| invalid("collective tensor element count overflow"))
}
fn invalid(reason: &str) -> MetadataError {
    MetadataError::Invalid {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_ordered_leading_axis_counts_and_empty_shapes() {
        let plan = CollectiveShape::all_gather(Shape::new([2, 3, 5]), 4).unwrap();
        assert_eq!(plan.output, Shape::new([8, 3, 5]));
        assert_eq!((plan.input_elements, plan.output_elements), (30, 120));
        let inverse = CollectiveShape::reduce_scatter(plan.output, 4).unwrap();
        assert_eq!(inverse.output, Shape::new([2, 3, 5]));
        assert_eq!((inverse.input_elements, inverse.output_elements), (120, 30));
        for input in [Shape::new([0, 5]), Shape::new([4, 0])] {
            let plan = CollectiveShape::all_gather(input.clone(), 2).unwrap();
            assert_eq!((plan.input_elements, plan.output_elements), (0, 0));
            assert_eq!(
                CollectiveShape::reduce_scatter(plan.output, 2)
                    .unwrap()
                    .output,
                input
            );
        }
        assert!(CollectiveShape::reduce_scatter(Shape::new([3, 2]), 2).is_err());
        assert!(CollectiveShape::all_gather(Shape::new([1]), 0).is_err());
        assert!(CollectiveShape::all_gather(Shape::new([]), 1).is_err());
        assert!(CollectiveShape::all_gather(Shape::new([usize::MAX, 0]), 2).is_err());
        assert!(CollectiveShape::all_gather(Shape::new([usize::MAX / 2, 3]), 1).is_err());
    }
}
