use super::*;

#[test]
fn ring_tags_preserve_existing_bit_fields_and_masks() {
    assert_eq!(ring_data_tag(0, 0, 0, 0), 0x4758_0000_0000_0000);
    assert_eq!(ring_data_tag(1, 2, 3, 4), 0x4758_0000_0120_3004);
    assert_eq!(
        ring_data_tag(1, 2, 3, 4),
        ring_data_tag(1 + (1 << 24), 2 + (1 << 4), 3 + (1 << 8), 4 + (1 << 12)),
    );
}

#[test]
fn reduction_agreement_tags_keep_operation_codes() {
    for operation in 0..=6 {
        let options = reduction_exchange_options(2, 17, operation);
        assert_eq!(options.root_rank, 2);
        assert_eq!(options.element_count, 17);
        assert_eq!(options.flags, 0);
        assert_eq!(options.tag, u64::from(operation) + 1);
    }
}
