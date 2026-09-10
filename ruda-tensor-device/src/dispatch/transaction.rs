use ruda_tensor::{
    backend::ExecutionError,
    ops::{TransactionOps, TransactionPrimitive, TransactionPrimitiveData},
};

use crate::{DeviceBackend, DeviceRuntime, FloatElement, IntElement, element::BoolElement};

impl<R, F, I, BT> TransactionOps<Self> for DeviceBackend<R, F, I, BT>
where
    R: DeviceRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    async fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> Result<TransactionPrimitiveData, ExecutionError> {
        ruda_kernel::tensor::transaction::execute(
            ruda_kernel::tensor::transaction::ReadbackBatch {
                read_floats: transaction.read_floats,
                read_qfloats: transaction.read_qfloats,
                read_ints: transaction.read_ints,
                read_bools: transaction.read_bools,
            },
        ).await
    }
}
