use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda]
pub(crate) fn scaled_index_division(
    index: usize,
    numerator_extent: usize,
    denominator_extent: usize,
    #[comptime] max_value: usize,
) -> (usize, usize) {
    let whole = index * (numerator_extent / denominator_extent);
    let fraction = numerator_extent % denominator_extent;
    if index == 0 || fraction == 0 {
        (whole, 0)
    } else if fraction <= max_value / index {
        let product = index * fraction;
        (whole + product / denominator_extent, product % denominator_extent)
    } else {
        let mut quotient = 0usize;
        let mut remainder = 0usize;
        let mut shift = comptime![max_value.ilog2() as usize + 1];
        while shift > 0 {
            shift -= 1;
            if remainder >= denominator_extent - remainder {
                remainder -= denominator_extent - remainder;
                quotient = quotient * 2 + 1;
            } else {
                remainder *= 2;
                quotient *= 2;
            }
            if ((index >> shift) & 1) != 0 {
                if remainder >= denominator_extent - fraction {
                    remainder -= denominator_extent - fraction;
                    quotient += 1;
                } else {
                    remainder += fraction;
                }
            }
        }
        (whole + quotient, remainder)
    }
}
