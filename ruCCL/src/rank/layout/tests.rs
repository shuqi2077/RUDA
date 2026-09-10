use super::*;

#[test]
fn balanced_ranges_preserve_rank_order_and_empty_partitions() {
    assert_eq!(balanced_ranges(10, 3), vec![0..4, 4..7, 7..10]);
    assert_eq!(balanced_ranges(2, 4), vec![0..1, 1..2, 2..2, 2..2]);
    assert_eq!(balanced_ranges(0, 3), vec![0..0, 0..0, 0..0]);
    for length in 0..32 {
        for partitions in 1..16 {
            let ranges = balanced_ranges(length, partitions);
            assert_eq!(ranges.len(), partitions);
            assert_eq!(ranges[0].start, 0);
            assert_eq!(ranges.last().unwrap().end, length);
            assert_eq!(ranges.iter().map(Range::len).sum::<usize>(), length);
            for pair in ranges.windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
                assert!(pair[0].len() >= pair[1].len());
                assert!(pair[0].len() - pair[1].len() <= 1);
            }
        }
    }
}

#[test]
#[should_panic]
fn zero_partitions_keeps_original_precondition() {
    balanced_ranges(0, 0);
}

#[test]
fn prefix_and_hierarchy_arithmetic_keep_overflow_and_saturation() {
    assert_eq!(prefix_offsets(&[2, 0, 3, 1]).unwrap(), vec![0, 2, 2, 5]);
    assert!(prefix_offsets(&[]).unwrap().is_empty());
    assert!(matches!(
        prefix_offsets(&[usize::MAX, 1]),
        Err(RankError::Overflow("pairwise count prefix"))
    ));
    assert_eq!(hierarchy_steps(&[vec![0, 1, 2], vec![3]], 4), 8);
    assert_eq!(hierarchy_steps(&[], 0), 0);
    assert_eq!(hierarchy_steps(&[vec![0, 1]], usize::MAX), u32::MAX);
}
