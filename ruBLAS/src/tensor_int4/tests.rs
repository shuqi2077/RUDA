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



#[test]
fn packed_storage_accounts_for_scales_and_zeros_without_dense_shadow() {
    let layout=AwqGemmLayout::new(128,256,64).unwrap();
    assert_eq!(layout.packed_payload_bytes(DType::F16,false).unwrap(),17664);
    assert_eq!(layout.packed_payload_bytes(DType::BF16,true).unwrap(),18176);
    assert_eq!(layout.packed_payload_bytes(DType::F32,false).unwrap(),18688);
    assert!(layout.packed_payload_bytes(DType::I32,false).is_err());
}
