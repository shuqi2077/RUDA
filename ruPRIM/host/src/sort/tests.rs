    use super::*;

    // Exercise both below and above the parallel threshold so serial and
    // rayon fast paths in sort_along_dim agree on row-wise sort results.
    fn check_sort_last_dim(rows: usize, cols: usize) {
        let n = rows * cols;
        // Deterministic non-monotonic input with repeats (mirrors bench fill % 1000).
        let src: Vec<f32> = (0..n)
            .map(|i| ((i * 1664525 + 1013904223) % 1000) as f32)
            .collect();

        let mut data = src.clone();
        let shape = Shape::new([rows, cols]);
        sort_along_dim(&mut data, &shape, 1, false, f32::total_cmp);

        for r in 0..rows {
            let row = &data[r * cols..(r + 1) * cols];
            for w in row.windows(2) {
                assert!(w[0] <= w[1], "row {r} not sorted: {:?}", row);
            }
            let mut expected: Vec<f32> = src[r * cols..(r + 1) * cols].to_vec();
            expected.sort_unstable_by(f32::total_cmp);
            assert_eq!(row, expected.as_slice());
        }
    }

    #[test]
    fn sort_along_last_dim_small_serial() {
        // 64*64 = 4K elements, well under PARALLEL_THRESHOLD.
        check_sort_last_dim(64, 64);
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn sort_along_last_dim_large_parallel() {
        // Just above PARALLEL_THRESHOLD so the rayon path is exercised
        // without paying for a much larger input under debug test builds.
        let cols = 1024;
        let rows = (PARALLEL_THRESHOLD / cols) + 1;
        check_sort_last_dim(rows, cols);
    }

    #[test]
    fn sort_along_last_dim_descending() {
        let mut data: Vec<f32> = (0..4096).map(|i| (i % 17) as f32).collect();
        let shape = Shape::new([128, 32]);
        sort_along_dim(&mut data, &shape, 1, true, f32::total_cmp);
        for r in 0..128 {
            let row = &data[r * 32..(r + 1) * 32];
            for w in row.windows(2) {
                assert!(w[0] >= w[1]);
            }
        }
    }

    fn check_sort_with_indices_last_dim(rows: usize, cols: usize, descending: bool) {
        let src: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut values = src.clone();
        let mut indices = vec![0isize; rows * cols];
        let shape = Shape::new([rows, cols]);
        sort_along_dim_with_indices(
            &mut values,
            &mut indices,
            &shape,
            1,
            descending,
            f32::total_cmp,
        );
        for r in 0..rows {
            let vs = &values[r * cols..(r + 1) * cols];
            let idx_row = &indices[r * cols..(r + 1) * cols];
            let orig = &src[r * cols..(r + 1) * cols];
            let want_order = if descending {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Greater
            };
            for w in vs.windows(2) {
                assert_ne!(f32::total_cmp(&w[0], &w[1]), want_order);
            }
            // Indices must reconstruct the sorted values from the original row
            // and must be a valid permutation of 0..cols (each index appears once).
            let mut seen = vec![false; cols];
            for (i, &orig_idx) in idx_row.iter().enumerate() {
                let j = orig_idx as usize;
                assert_eq!(vs[i], orig[j]);
                assert!(!seen[j], "row {r}: index {j} repeated");
                seen[j] = true;
            }
        }
    }

    #[test]
    fn sort_with_indices_last_dim_ascending() {
        // 512*512 = 256K, exactly PARALLEL_THRESHOLD (hits parallel branch).
        check_sort_with_indices_last_dim(512, 512, false);
    }

    #[test]
    fn sort_with_indices_last_dim_descending() {
        check_sort_with_indices_last_dim(512, 512, true);
    }

    fn check_argsort_last_dim(rows: usize, cols: usize, descending: bool) {
        let src: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919) % 997) as f32)
            .collect();
        let mut indices = vec![0isize; rows * cols];
        let shape = Shape::new([rows, cols]);
        argsort_along_dim(&src, &mut indices, &shape, 1, descending, f32::total_cmp);
        for r in 0..rows {
            let idx_row = &indices[r * cols..(r + 1) * cols];
            let orig = &src[r * cols..(r + 1) * cols];
            let sorted: Vec<f32> = idx_row.iter().map(|&i| orig[i as usize]).collect();
            for w in sorted.windows(2) {
                if descending {
                    assert!(w[0] >= w[1]);
                } else {
                    assert!(w[0] <= w[1]);
                }
            }
            // idx_row must be a permutation of 0..cols; with heavy duplicates
            // in src this catches parallel bugs that emit the same index twice.
            let mut seen = vec![false; cols];
            for &i in idx_row {
                let j = i as usize;
                assert!(!seen[j], "row {r}: index {j} repeated");
                seen[j] = true;
            }
        }
    }

    #[test]
    fn argsort_last_dim_ascending() {
        // 200*1500 = 300K, above PARALLEL_THRESHOLD.
        check_argsort_last_dim(200, 1500, false);
    }

    #[test]
    fn argsort_last_dim_descending() {
        check_argsort_last_dim(200, 1500, true);
    }
