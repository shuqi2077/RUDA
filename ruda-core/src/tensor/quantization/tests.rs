use super::packing::unpack_q_to_i8s;

use super::*;
use alloc::vec;

#[test]
fn should_pack_i8s_to_u32() {
    let packed = pack_i8s_to_u32s(vec![-128, 2, -3, 127]);

    assert_eq!(packed, vec![2147287680]);
}

#[test]
fn should_pack_i8s_to_u32_padded() {
    let packed = pack_i8s_to_u32s(vec![-128, 2, -3, 127, 55]);
    let packed_padded = pack_i8s_to_u32s(vec![-128, 2, -3, 127, 55, 0, 0, 0]);

    assert_eq!(packed, vec![2147287680, 55]);
    assert_eq!(packed, packed_padded);
}

#[test]
fn should_unpack_u32s_to_i8s() {
    let unpacked = unpack_q_to_i8s(&[2147287680u32], 4, &QuantValue::Q8S);

    assert_eq!(unpacked, vec![-128, 2, -3, 127]);
}

#[test]
fn should_unpack_u32s_to_i8s_padded() {
    let unpacked = unpack_q_to_i8s(&[55u32], 1, &QuantValue::Q8S);

    assert_eq!(unpacked, vec![55]);
}

#[test]
fn should_unpack_u32s_to_i8s_arange() {
    let unpacked = unpack_q_to_i8s(
        &[
            0u32, 286331136, 286331153, 572657937, 572662306, 857874978, 858993459, 858993459,
            1145324612, 1145324612, 1431655748, 1431655765, 1717982549, 1717986918, 2003199590,
            2004318071,
        ],
        128,
        &QuantValue::Q4S,
    );

    assert_eq!(
        unpacked,
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
        ]
    );
}

#[cfg(feature = "tensor-data")]
#[test]
fn should_pack_unpack_quantization_parameters_per_tensor_symmetric() {
    // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    let scale = 0.03937008;
    let values = vec![0i8, 25, 51, 76, 102, 127];

    let q_bytes = QuantizedBytes::new(
        values.clone(),
        QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native),
        &[scale],
    );

    let (q_values, qparams) = q_bytes.into_vec_i8();

    assert_eq!(qparams.scales, vec![scale]);

    assert_eq!(q_values, values);
}
