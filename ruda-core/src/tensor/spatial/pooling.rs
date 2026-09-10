/// Calculate the expected output size when doing a pooling operation.
///
/// # Arguments
///
/// * `kernel_size` - Size of the pooling kernel
/// * `stride` - Stride of the pooling operation
/// * `padding` - Padding applied to input
/// * `dilation` - Dilation of the pooling kernel
/// * `size_in` - Input size (height or width)
/// * `ceil_mode` - If true, use ceiling instead of floor for output size calculation.
///   This allows the last pooling window to go out-of-bounds if needed.
pub fn calculate_pool_output_size(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    size_in: usize,
    ceil_mode: bool,
) -> usize {
    let numerator = size_in + 2 * padding - dilation * (kernel_size - 1) - 1;
    if ceil_mode {
        // Ceiling division: (a + b - 1) / b
        numerator.div_ceil(stride) + 1
    } else {
        // Floor division (default)
        numerator / stride + 1
    }
}

