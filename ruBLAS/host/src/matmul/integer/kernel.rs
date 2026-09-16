/// Accumulate a row-major i64 matrix product into zero-initialized output.
///
/// Traversing RHS rows in the innermost loop reads and writes contiguous
/// memory instead of gathering one RHS column at a time. Each output still
/// receives products in ascending contraction-axis order, with wrapping
/// multiplication and addition in i64 throughout.
pub(super) fn matmul_i64(
    lhs: &[i64],
    rhs: &[i64],
    output: &mut [i64],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!(lhs.len(), m * k);
    debug_assert_eq!(rhs.len(), k * n);
    debug_assert_eq!(output.len(), m * n);
    if n == 0 {
        return;
    }

    for (i, out_row) in output.chunks_exact_mut(n).enumerate() {
        let lhs_row = &lhs[i * k..(i + 1) * k];
        for (&a, rhs_row) in lhs_row.iter().zip(rhs.chunks_exact(n)) {
            for (out, &b) in out_row.iter_mut().zip(rhs_row) {
                *out = out.wrapping_add(a.wrapping_mul(b));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use self::alloc::{vec, vec::Vec};
    use super::matmul_i64;

    #[test]
    fn matches_column_dot_reference_with_wrapping_and_tails() {
        for (m, k, n) in [(1, 1, 1), (3, 17, 1), (2, 5, 7), (5, 31, 17), (2, 7, 129)] {
            let lhs: Vec<i64> = (0..m * k)
                .map(|i| (i as i64).wrapping_mul(i64::MAX / 3).wrapping_sub(5))
                .collect();
            let rhs: Vec<i64> = (0..k * n)
                .map(|i| (i as i64).wrapping_mul(i64::MIN / 7).wrapping_add(11))
                .collect();
            let mut output = vec![0; m * n];
            matmul_i64(&lhs, &rhs, &mut output, m, k, n);
            for i in 0..m {
                for j in 0..n {
                    let expected = (0..k).fold(0i64, |sum, contraction| {
                        sum.wrapping_add(
                            lhs[i * k + contraction].wrapping_mul(rhs[contraction * n + j]),
                        )
                    });
                    assert_eq!(output[i * n + j], expected, "shape [{m}, {k}, {n}], ({i}, {j})");
                }
            }
        }
    }

    #[test]
    fn handles_empty_axes() {
        for (m, k, n) in [(0, 3, 5), (2, 3, 0), (2, 0, 5), (0, 0, 0)] {
            let mut output = vec![0; m * n];
            matmul_i64(&vec![0; m * k], &vec![0; k * n], &mut output, m, k, n);
            assert_eq!(output, vec![0; m * n]);
        }
    }
}
