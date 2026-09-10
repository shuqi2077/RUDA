use crate::tensor::Shape;
use super::*;
use alloc::string::ToString;
use alloc::{vec, vec::Vec};

#[test]
fn test_slice_to_str() {
    assert_eq!(Slice::new(0, None, 1).to_string(), "..");

    assert_eq!(Slice::new(0, Some(1), 1).to_string(), "0");

    assert_eq!(Slice::new(0, Some(10), 1).to_string(), "..10");
    assert_eq!(Slice::new(1, Some(10), 1).to_string(), "1..10");

    assert_eq!(Slice::new(-3, Some(10), -2).to_string(), "-3..10;-2");
}

#[test]
fn test_slice_from_str() {
    assert_eq!("1".parse::<Slice>(), Ok(Slice::new(1, Some(2), 1)));
    assert_eq!("..".parse::<Slice>(), Ok(Slice::new(0, None, 1)));
    assert_eq!("..3".parse::<Slice>(), Ok(Slice::new(0, Some(3), 1)));
    assert_eq!("..=3".parse::<Slice>(), Ok(Slice::new(0, Some(4), 1)));

    assert_eq!("-12..3".parse::<Slice>(), Ok(Slice::new(-12, Some(3), 1)));
    assert_eq!("..;-1".parse::<Slice>(), Ok(Slice::new(0, None, -1)));

    assert_eq!("..=3;-2".parse::<Slice>(), Ok(Slice::new(0, Some(4), -2)));

    assert_eq!(
        "..;0".parse::<Slice>(),
        Err(crate::tensor::errors::ExpressionError::invalid_expression(
            "Step cannot be zero",
            "..;0"
        ))
    );

    assert_eq!(
        "".parse::<Slice>(),
        Err(crate::tensor::errors::ExpressionError::parse_error("Empty expression", ""))
    );
    assert_eq!(
        "a".parse::<Slice>(),
        Err(crate::tensor::errors::ExpressionError::parse_error(
            "Invalid integer: 'a': invalid digit found in string",
            "a"
        ))
    );
    assert_eq!(
        "..a".parse::<Slice>(),
        Err(crate::tensor::errors::ExpressionError::parse_error(
            "Invalid integer: 'a': invalid digit found in string",
            "..a"
        ))
    );
    assert_eq!(
        "a:b:c".parse::<Slice>(),
        Err(crate::tensor::errors::ExpressionError::parse_error(
            "Invalid integer: 'a:b:c': invalid digit found in string",
            "a:b:c"
        ))
    );
}

#[test]
fn test_slice_output_size() {
    // Test the output_size method directly
    assert_eq!(Slice::new(0, Some(10), 1).output_size(10), 10);
    assert_eq!(Slice::new(0, Some(10), 2).output_size(10), 5);
    assert_eq!(Slice::new(0, Some(10), 3).output_size(10), 4); // ceil(10/3)
    assert_eq!(Slice::new(0, Some(10), -1).output_size(10), 10);
    assert_eq!(Slice::new(0, Some(10), -2).output_size(10), 5);
    assert_eq!(Slice::new(2, Some(8), -3).output_size(10), 2); // ceil(6/3)
    assert_eq!(Slice::new(5, Some(5), 1).output_size(10), 0); // empty range
}

#[test]
fn test_bound_to() {
    assert_eq!(
        Slice::new(0, None, 1).bound_to(10),
        Slice::new(0, Some(10), 1)
    );
    assert_eq!(
        Slice::new(0, Some(5), 1).bound_to(10),
        Slice::new(0, Some(5), 1)
    );

    assert_eq!(
        Slice::new(0, None, -1).bound_to(10),
        Slice::new(0, Some(-11), -1)
    );
    assert_eq!(
        Slice::new(0, Some(-5), -1).bound_to(10),
        Slice::new(0, Some(-5), -1)
    );
}

#[test]
fn test_slice_iter() {
    assert_eq!(
        Slice::new(2, Some(3), 1).into_iter().collect::<Vec<_>>(),
        vec![2]
    );
    assert_eq!(
        Slice::new(3, Some(-1), -1).into_iter().collect::<Vec<_>>(),
        vec![3, 2, 1, 0]
    );

    assert_eq!(Slice::new(3, Some(-1), -1).into_vec(), vec![3, 2, 1, 0]);

    assert_eq!(
        Slice::new(3, None, 2)
            .into_iter()
            .take(3)
            .collect::<Vec<_>>(),
        vec![3, 5, 7]
    );
    assert_eq!(
        Slice::new(3, None, 2)
            .bound_to(8)
            .into_iter()
            .collect::<Vec<_>>(),
        vec![3, 5, 7]
    );
}

#[test]
#[should_panic(
    expected = "Slice must have an end to convert to a vector: Slice { start: 0, end: None, step: 1 }"
)]
fn test_unbound_slice_into_vec() {
    Slice::new(0, None, 1).into_vec();
}

#[test]
fn into_slices_should_return_for_all_shape_dims() {
    let slice = s![1];
    let shape = Shape::new([2, 3, 1]);

    let slices = slice.into_slices(&shape);

    assert_eq!(slices.len(), shape.len());

    assert_eq!(slices[0], Slice::new(1, Some(2), 1));
    assert_eq!(slices[1], Slice::new(0, Some(3), 1));
    assert_eq!(slices[2], Slice::new(0, Some(1), 1));

    let slice = s![1, 0..2];
    let slices = slice.into_slices(&shape);

    assert_eq!(slices.len(), shape.len());

    assert_eq!(slices[0], Slice::new(1, Some(2), 1));
    assert_eq!(slices[1], Slice::new(0, Some(2), 1));
    assert_eq!(slices[2], Slice::new(0, Some(1), 1));

    let slice = s![..];
    let slices = slice.into_slices(&shape);

    assert_eq!(slices.len(), shape.len());

    assert_eq!(slices[0], Slice::new(0, Some(2), 1));
    assert_eq!(slices[1], Slice::new(0, Some(3), 1));
    assert_eq!(slices[2], Slice::new(0, Some(1), 1));
}

#[test]
fn into_slices_all_dimensions() {
    let slice = s![1, ..2, ..];
    let shape = Shape::new([2, 3, 1]);

    let slices = slice.into_slices(&shape);

    assert_eq!(slices.len(), shape.len());

    assert_eq!(slices[0], Slice::new(1, Some(2), 1));
    assert_eq!(slices[1], Slice::new(0, Some(2), 1));
    assert_eq!(slices[2], Slice::new(0, Some(1), 1));
}

#[test]
fn into_slices_supports_empty_dimensions() {
    let slice = s![.., 1, ..];
    let shape = Shape::new([0, 3, 1]);

    let slices = slice.into_slices(&shape);

    assert_eq!(slices.len(), shape.len());

    assert_eq!(slices[0], Slice::new(0, Some(0), 1));
    assert_eq!(slices[1], Slice::new(1, Some(2), 1));
    assert_eq!(slices[2], Slice::new(0, Some(1), 1));
}

#[test]
#[should_panic = "Too many slices provided for shape"]
fn into_slices_should_match_shape_rank() {
    let slice = s![.., 1, ..];
    let shape = Shape::new([3, 1]);

    let _ = slice.into_slices(&shape);
}

#[test]
fn should_support_const_and_full() {
    static SLICES: [Slice; 2] = [Slice::full(), Slice::new(2, None, 1)];
    assert_eq!(SLICES[0], Slice::new(0, None, 1));
    assert_eq!(SLICES[1], Slice::new(2, None, 1));
}

#[test]
fn should_support_default() {
    assert_eq!(Slice::default(), Slice::new(0, None, 1));
}

#[test]
fn should_support_copy() {
    let mut slice = Slice::new(1, Some(3), 2);
    let slice_copy = slice;

    slice.end = Some(4);

    assert_eq!(slice, Slice::new(1, Some(4), 2));
    assert_eq!(slice_copy, Slice::new(1, Some(3), 2));
}
