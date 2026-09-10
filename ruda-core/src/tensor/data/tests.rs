use crate::tensor::distribution::Distribution;
use crate::tensor::element::{Element, ElementConversion};
use alloc::vec::Vec;
use half::{bf16, f16};

use super::*;
use alloc::vec;
use crate::shape;
use rand::{
    SeedableRng,
    rngs::{StdRng, SysRng},
};

#[test]
fn should_have_rank() {
    let shape = [3, 5, 6];
    let data = TensorData::random::<f32, _, _>(
        shape,
        Distribution::Default,
        &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
    );

    assert_eq!(data.rank(), 3);
}

#[test]
fn into_vec_should_yield_same_value_as_iter() {
    let shape = [3, 5, 6];
    let data = TensorData::random::<f32, _, _>(
        shape,
        Distribution::Default,
        &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
    );

    let expected = data.iter::<f32>().collect::<Vec<f32>>();
    let actual = data.into_vec::<f32>().unwrap();

    assert_eq!(expected, actual);
}

#[test]
#[should_panic]
fn into_vec_should_assert_wrong_dtype() {
    let shape = [3, 5, 6];
    let data = TensorData::random::<f32, _, _>(
        shape,
        Distribution::Default,
        &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
    );

    data.into_vec::<i32>().unwrap();
}

#[test]
fn should_have_right_num_elements() {
    let shape = [3, 5, 6];
    let num_elements: usize = shape.iter().product();
    let data = TensorData::random::<f32, _, _>(
        shape,
        Distribution::Default,
        &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
    );

    assert_eq!(num_elements, data.bytes.len() / 4); // f32 stored as u8s
    assert_eq!(num_elements, data.as_slice::<f32>().unwrap().len());
}

#[test]
fn should_have_right_shape() {
    let data = TensorData::from([[3.0, 5.0, 6.0]]);
    assert_eq!(data.shape, shape![1, 3]);

    let data = TensorData::from([[4.0, 5.0, 8.0], [3.0, 5.0, 6.0]]);
    assert_eq!(data.shape, shape![2, 3]);

    let data = TensorData::from([3.0, 5.0, 6.0]);
    assert_eq!(data.shape, shape![3]);
}

#[test]
fn should_convert_bytes_correctly() {
    let mut vector: Vec<f32> = Vec::with_capacity(5);
    vector.push(2.0);
    vector.push(3.0);
    let data1 = TensorData::new(vector, vec![2]);

    let factor = core::mem::size_of::<f32>() / core::mem::size_of::<u8>();
    assert_eq!(data1.bytes.len(), 2 * factor);
    assert_eq!(data1.bytes.capacity(), 5 * factor);
}

#[test]
fn should_convert_bytes_correctly_inplace() {
    fn test_precision<E: Element>() {
        let data = TensorData::new((0..32).collect(), [32]);
        for (i, val) in data
            .clone()
            .convert::<E>()
            .into_vec::<E>()
            .unwrap()
            .into_iter()
            .enumerate()
        {
            assert_eq!(i as u32, val.elem::<u32>())
        }
    }
    test_precision::<f32>();
    test_precision::<f16>();
    test_precision::<i64>();
    test_precision::<i32>();
}

macro_rules! test_dtypes {
($test_name:ident, $($dtype:ty),*) => {
    $(
        paste::paste! {
            #[test]
            fn [<$test_name _ $dtype:snake>]() {
                let full_dtype = TensorData::full_dtype([2, 16], 4, <$dtype>::dtype());
                let full = TensorData::full::<$dtype, _>([2, 16], 4.elem());
                assert_eq!(full_dtype, full);
            }
        }
    )*
};
}

test_dtypes!(
    should_create_with_dtype,
    bool,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f16,
    bf16,
    f32,
    f64
);

#[test]
fn should_serialize_deserialize_tensor_data() {
    let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    assert_eq!(
        data.as_bytes(),
        [
            0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192,
            64
        ]
    );
    let serialized = serde_json::to_string(&data).unwrap();
    let deserialized: TensorData = serde_json::from_str(&serialized).unwrap();
    assert_eq!(data, deserialized);
}

#[test]
fn should_deserialize_tensor_data_with_shape_inner() {
    // TensorData `shape` was previously a Vec<usize>.
    let serialized = r#"{
        "bytes": [0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64],
        "shape": [2, 3],
        "dtype": "F32"
    }"#;

    let data: TensorData = serde_json::from_str(serialized).unwrap();
    assert_eq!(data.shape, shape![2, 3]);
    assert_eq!(
        data.as_slice::<f32>().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn should_serialize_shape_as_flat_array() {
    // Ensure the new Shape serializes identically to how Vec<usize> used to,
    // i.e. as a flat JSON array, not as an object like `{"dims": [2, 3]}`.
    let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let serialized = serde_json::to_string(&data).unwrap();
    let json: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    assert_eq!(json["shape"], serde_json::json!([2, 3]));
}
