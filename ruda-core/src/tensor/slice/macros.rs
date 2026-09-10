/// Slice argument constructor for tensor indexing.
///
/// The `s![]` macro is used to create multi-dimensional slice specifications for tensors.
/// It converts various range syntax forms into a `&[Slice]` that can be used with
/// `tensor.slice()` and `tensor.slice_assign()` operations.
///
/// # Syntax Overview
///
/// ## Basic Forms
///
/// * **`s![index]`** - Index a single element (produces a subview with that axis removed)
/// * **`s![range]`** - Slice a range of elements
/// * **`s![range;step]`** - Slice a range with a custom step
/// * **`s![dim1, dim2, ...]`** - Multiple dimensions, each can be any of the above forms
///
/// ## Range Types
///
/// All standard Rust range types are supported:
/// * **`a..b`** - From `a` (inclusive) to `b` (exclusive)
/// * **`a..=b`** - From `a` to `b` (both inclusive)
/// * **`a..`** - From `a` to the end
/// * **`..b`** - From the beginning to `b` (exclusive)
/// * **`..=b`** - From the beginning to `b` (inclusive)
/// * **`..`** - The full range (all elements)
///
/// ## Negative Indices
///
/// Negative indices count from the end of the axis:
/// * **`-1`** refers to the last element
/// * **`-2`** refers to the second-to-last element
/// * And so on...
///
/// This works in all range forms: `s![-3..-1]`, `s![-2..]`, `s![..-1]`
///
/// ## Step Syntax
///
/// Steps control the stride between selected elements:
/// * **`;step`** after a range specifies the step
/// * **Positive steps** select every nth element going forward
/// * **Negative steps** select every nth element going backward
/// * Default step is `1` when not specified
/// * Step cannot be `0`
///
/// ### Negative Step Behavior
///
/// With negative steps, the range bounds still specify *which* elements to include,
/// but the traversal order is reversed:
///
/// * `s![0..5;-1]` selects indices `[4, 3, 2, 1, 0]` (not `[0, 1, 2, 3, 4]`)
/// * `s![2..8;-2]` selects indices `[7, 5, 3]` (starting from 7, going backward by 2)
/// * `s![..;-1]` reverses the entire axis
///
/// This matches the semantics of NumPy and the ndarray crate.
///
/// # Examples
///
/// ## Basic Slicing
///
/// ```rust,ignore
/// use ruda_tensor::api::{Tensor, s};
///
/// # fn example<B: Backend>(tensor: Tensor<B, 3>) {
/// // Select rows 0-5 (exclusive)
/// let subset = tensor.slice(s![0..5, .., ..]);
///
/// // Select the last row
/// let last_row = tensor.slice(s![-1, .., ..]);
///
/// // Select columns 2, 3, 4
/// let cols = tensor.slice(s![.., 2..5, ..]);
///
/// // Select a single element at position [1, 2, 3]
/// let element = tensor.slice(s![1, 2, 3]);
/// # }
/// ```
///
/// ## Slicing with Steps
///
/// ```rust,ignore
/// use ruda_tensor::api::{Tensor, s};
///
/// # fn example<B: Backend>(tensor: Tensor<B, 2>) {
/// // Select every 2nd row
/// let even_rows = tensor.slice(s![0..10;2, ..]);
///
/// // Select every 3rd column
/// let cols = tensor.slice(s![.., 0..9;3]);
///
/// // Select every 2nd element in reverse order
/// let reversed_even = tensor.slice(s![10..0;-2, ..]);
/// # }
/// ```
///
/// ## Reversing Dimensions
///
/// ```rust,ignore
/// use ruda_tensor::api::{Tensor, s};
///
/// # fn example<B: Backend>(tensor: Tensor<B, 2>) {
/// // Reverse the first dimension
/// let reversed = tensor.slice(s![..;-1, ..]);
///
/// // Reverse both dimensions
/// let fully_reversed = tensor.slice(s![..;-1, ..;-1]);
///
/// // Reverse a specific range
/// let range_reversed = tensor.slice(s![2..8;-1, ..]);
/// # }
/// ```
///
/// ## Complex Multi-dimensional Slicing
///
/// ```rust,ignore
/// use ruda_tensor::api::{Tensor, s};
///
/// # fn example<B: Backend>(tensor: Tensor<B, 4>) {
/// // Mix of different slice types
/// let complex = tensor.slice(s![
///     0..10;2,    // Every 2nd element from 0 to 10
///     ..,         // All elements in dimension 1
///     5..15;-3,   // Every 3rd element from 14 down to 5
///     -1          // Last element in dimension 3
/// ]);
///
/// // Using inclusive ranges
/// let inclusive = tensor.slice(s![2..=5, 1..=3, .., ..]);
///
/// // Negative indices with steps
/// let from_end = tensor.slice(s![-5..-1;2, .., .., ..]);
/// # }
/// ```
///
/// ## Slice Assignment
///
/// ```rust,ignore
/// use ruda_tensor::api::{Tensor, s};
///
/// # fn example<B: Backend>(tensor: Tensor<B, 2>, values: Tensor<B, 2>) {
/// // Assign to every 2nd row
/// let tensor = tensor.slice_assign(s![0..10;2, ..], values);
///
/// // Assign to a reversed slice
/// let tensor = tensor.slice_assign(s![..;-1, 0..5], values);
/// # }
/// ```
#[macro_export]
macro_rules! s {
    // Empty - should not happen
    [] => {
        compile_error!("Empty slice specification")
    };

    // Single expression with step
    [$range:expr; $step:expr] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::tensor::Slice::from_range_stepped($range, $step)
            }
        }
    };

    // Single expression without step (no comma after)
    [$range:expr] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::tensor::Slice::from($range)
            }
        }
    };

    // Two or more expressions with first having step
    [$range:expr; $step:expr, $($rest:tt)*] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::s!(@internal [$crate::tensor::Slice::from_range_stepped($range, $step)] $($rest)*)
            }
        }
    };

    // Two or more expressions with first not having step
    [$range:expr, $($rest:tt)*] => {
        {
            #[allow(clippy::reversed_empty_ranges)]
            {
                $crate::s!(@internal [$crate::tensor::Slice::from($range)] $($rest)*)
            }
        }
    };

    // Internal: finished parsing
    (@internal [$($acc:expr),*]) => {
        [$($acc),*]
    };

    // Internal: parse range with step followed by comma
    (@internal [$($acc:expr),*] $range:expr; $step:expr, $($rest:tt)*) => {
        $crate::s!(@internal [$($acc,)* $crate::tensor::Slice::from_range_stepped($range, $step as isize)] $($rest)*)
    };

    // Internal: parse range with step at end
    (@internal [$($acc:expr),*] $range:expr; $step:expr) => {
        $crate::s!(@internal [$($acc,)* $crate::tensor::Slice::from_range_stepped($range, $step as isize)])
    };

    // Internal: parse range without step followed by comma
    (@internal [$($acc:expr),*] $range:expr, $($rest:tt)*) => {
        $crate::s!(@internal [$($acc,)* $crate::tensor::Slice::from($range)] $($rest)*)
    };

    // Internal: parse range without step at end
    (@internal [$($acc:expr),*] $range:expr) => {
        $crate::s!(@internal [$($acc,)* $crate::tensor::Slice::from($range)])
    };
}
