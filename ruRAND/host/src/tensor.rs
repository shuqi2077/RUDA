use ruda_core::tensor::{FloatDType, IntDType, Shape, data::TensorData, distribution::Distribution, host::HostTensor};
use half::{bf16, f16};

pub fn float_random(
    shape: Shape,
    distribution: Distribution,
    dtype: FloatDType,
) -> HostTensor {
    let mut seed = crate::state::SEED.lock().unwrap();
    let mut rng = seed.take().unwrap_or_else(crate::state::get_seeded_rng);
    let data = match dtype {
        FloatDType::F64 => TensorData::random::<f64, _, _>(shape, distribution, &mut rng),
        FloatDType::F32 | FloatDType::Flex32 => {
            TensorData::random::<f32, _, _>(shape, distribution, &mut rng)
        }
        FloatDType::F16 => TensorData::random::<f16, _, _>(shape, distribution, &mut rng),
        FloatDType::BF16 => TensorData::random::<bf16, _, _>(shape, distribution, &mut rng),
    };
    *seed = Some(rng);
    HostTensor::from_data(data)
}

pub fn int_random(
    shape: Shape,
    distribution: Distribution,
    dtype: IntDType,
) -> HostTensor {
    let mut seed = crate::state::SEED.lock().unwrap();
    let mut rng = seed.take().unwrap_or_else(crate::state::get_seeded_rng);
    let data = match dtype {
        IntDType::I64 => TensorData::random::<i64, _, _>(shape, distribution, &mut rng),
        IntDType::I32 => TensorData::random::<i32, _, _>(shape, distribution, &mut rng),
        IntDType::I16 => TensorData::random::<i16, _, _>(shape, distribution, &mut rng),
        IntDType::I8 => TensorData::random::<i8, _, _>(shape, distribution, &mut rng),
        IntDType::U64 => TensorData::random::<u64, _, _>(shape, distribution, &mut rng),
        IntDType::U32 => TensorData::random::<u32, _, _>(shape, distribution, &mut rng),
        IntDType::U16 => TensorData::random::<u16, _, _>(shape, distribution, &mut rng),
        IntDType::U8 => TensorData::random::<u8, _, _>(shape, distribution, &mut rng),
    };
    *seed = Some(rng);
    HostTensor::from_data(data)
}

