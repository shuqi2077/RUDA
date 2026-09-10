use ruda_core::tensor::DType;
use ruda_kernel::dsl::prelude::InputScalar;

use super::{MaskFillStrategy, mask_where::MaskWhereStrategy};
use ruda_kernel::dsl::Runtime;
use ruda_kernel::tensor::RudaTensor;

/// Execute the mask fill kernel.
pub fn mask_fill_auto<R: Runtime>(
    tensor: RudaTensor<R>,
    mask: RudaTensor<R>,
    value: InputScalar,
    dtype_bool: DType,
) -> RudaTensor<R> {
    let strategy = if tensor.can_mut() && tensor.is_nonoverlapping() {
        MaskFillStrategy::Inplace
    } else {
        MaskFillStrategy::Readonly
    };

    super::mask_fill(tensor, mask, value, strategy, dtype_bool)
}

/// Execute the mask where kernel.
pub fn mask_where_auto<R: Runtime>(
    tensor: RudaTensor<R>,
    mask: RudaTensor<R>,
    value: RudaTensor<R>,
    dtype_bool: DType,
) -> RudaTensor<R> {
    let strategy = if tensor.can_mut_broadcast(&value) {
        MaskWhereStrategy::InplaceLhs
    } else if value.can_mut_broadcast(&tensor) {
        MaskWhereStrategy::InplaceRhs
    } else {
        MaskWhereStrategy::Readonly
    };

    super::mask_where(tensor, mask, value, strategy, dtype_bool)
}
