use crate::tensor::Shape;

/// Calculate the number of unfolding windows that can be extracted from a dimension of given size.
pub fn calculate_unfold_windows(dim_size: usize, window_size: usize, step_size: usize) -> usize {
    assert!(step_size > 0);
    let x = dim_size + step_size;
    if x < window_size {
        0
    } else {
        (x - window_size) / step_size
    }
}

/// Calculate the output shape for an unfold operation.
///
/// The operation yields a view with all complete windows of size `size` in dimension `dim`;
/// where windows are advanced by `step` at each index.
///
/// The number of windows is `max(0, (shape[dim] - size).ceil_div(step))`.
///
/// # Arguments
///
/// * `shape` - The input shape to unfold; of shape ``[pre=..., dim shape, post=...]``
/// * `dim` - the dimension to unfold.
/// * `size` - the size of each unfolded window.
/// * `step` - the step between each window.
///
/// # Returns
///
/// A shape with ``[pre=..., windows, post=..., size]``.
pub fn calculate_unfold_shape<S: Into<Shape>>(
    shape: S,
    dim: usize,
    size: usize,
    step: usize,
) -> Shape {
    let mut shape = shape.into();
    let d_shape = shape[dim];
    let windows = calculate_unfold_windows(d_shape, size, step);
    shape[dim] = windows;
    shape.push(size);

    shape
}
