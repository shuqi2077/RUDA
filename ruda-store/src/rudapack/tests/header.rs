use crate::rudapack::base::*;

#[test]
fn test_header_serialization() {
    let header = RudapackHeader::new(12345);

    // Check fields
    assert_eq!(header.magic, MAGIC_NUMBER);
    assert_eq!(header.version, FORMAT_VERSION);
    assert_eq!(header.metadata_size, 12345);

    // Serialize to bytes
    let bytes = header.into_bytes();
    assert_eq!(bytes.len(), HEADER_SIZE);

    // Deserialize back
    let header2 = RudapackHeader::from_bytes(&bytes).unwrap();
    assert_eq!(header2.magic, header.magic);
    assert_eq!(header2.version, header.version);
    assert_eq!(header2.metadata_size, header.metadata_size);
}

#[test]
fn test_header_invalid_magic() {
    let mut bytes = [0u8; HEADER_SIZE];
    // Write wrong magic number
    bytes[0..4].copy_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    let result = RudapackHeader::from_bytes(&bytes);
    match result {
        Err(RudapackError::InvalidMagicNumber) => {}
        _ => panic!("Expected InvalidMagicNumber error"),
    }
}

#[test]
fn test_header_insufficient_bytes() {
    let bytes = [0u8; 5]; // Too short

    let result = RudapackHeader::from_bytes(&bytes);
    match result {
        Err(RudapackError::InvalidHeader) => {}
        _ => panic!("Expected InvalidHeader error"),
    }
}

#[test]
fn test_version_compatibility() {
    // Create a header with current version
    let header = RudapackHeader::new(100);
    let bytes = header.into_bytes();

    // Should succeed with current version
    let result = RudapackHeader::from_bytes(&bytes);
    assert!(result.is_ok());

    // Test with future version (should fail in real implementation)
    // For now, we just verify the version field is correctly set
    let header = result.unwrap();
    assert_eq!(header.version, FORMAT_VERSION);
}
