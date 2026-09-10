use crate::dsl::Runtime;
use ruda_core::tensor::DType;
use ruda_core::ir::features::TypeUsage;

pub fn supports_dtype<R: Runtime>(device: &R::Device, dtype: DType) -> bool {
    let client = R::client(device);

    let type_usage = client.properties().type_usage(dtype.into());
    // Same as `TypeUsage::all_scalar()`, but we make the usage explicit here
    type_usage.is_superset(
        TypeUsage::Buffer
            | TypeUsage::Conversion
            | TypeUsage::Arithmetic
            | TypeUsage::DotProduct,
    )
}
