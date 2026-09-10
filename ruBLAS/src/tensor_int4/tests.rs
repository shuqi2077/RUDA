use super::*;

#[test]
fn awq_layout_validates_packing_groups_and_index_range() {
    let layout = AwqGemmLayout::new(128, 24, 32).unwrap();
    assert_eq!(layout.groups(), 4);
    for (k, n, group) in [(0, 8, 1), (8, 0, 1), (8, 8, 0), (7, 8, 2), (8, 7, 2)] {
        assert!(AwqGemmLayout::new(k, n, group).is_err());
    }
    assert!(AwqGemmLayout::new(u32::MAX as usize, 8, 1).is_err());
}


