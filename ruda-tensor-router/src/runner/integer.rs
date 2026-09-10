use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn run_integer(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        op: &IntOperationIr,
    ) {
        match op {
            IntOperationIr::IntoFloat(desc) => {
                let tensor = handles.get_int_tensor::<B>(&desc.input);

                let output = B::int_into_float(tensor, desc.out.dtype.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            IntOperationIr::Matmul(desc) => {
                binary_int_ops!(handles, desc, B::int_matmul)
            }
            IntOperationIr::BitwiseAnd(desc) => {
                binary_int_ops!(handles, desc, B::bitwise_and)
            }
            IntOperationIr::BitwiseAndScalar(desc) => {
                scalar_int_ops!(handles, desc, B::bitwise_and_scalar)
            }
            IntOperationIr::BitwiseOr(desc) => {
                binary_int_ops!(handles, desc, B::bitwise_or)
            }
            IntOperationIr::BitwiseOrScalar(desc) => {
                scalar_int_ops!(handles, desc, B::bitwise_or_scalar)
            }
            IntOperationIr::BitwiseXor(desc) => {
                binary_int_ops!(handles, desc, B::bitwise_xor)
            }
            IntOperationIr::BitwiseXorScalar(desc) => {
                scalar_int_ops!(handles, desc, B::bitwise_xor_scalar)
            }
            IntOperationIr::BitwiseNot(desc) => {
                unary_int_ops!(handles, desc, B::bitwise_not)
            }
            IntOperationIr::BitwiseLeftShift(desc) => {
                binary_int_ops!(handles, desc, B::bitwise_left_shift)
            }
            IntOperationIr::BitwiseRightShift(desc) => {
                binary_int_ops!(handles, desc, B::bitwise_right_shift)
            }
            IntOperationIr::BitwiseLeftShiftScalar(desc) => {
                scalar_int_ops!(handles, desc, B::bitwise_left_shift_scalar)
            }
            IntOperationIr::BitwiseRightShiftScalar(desc) => {
                scalar_int_ops!(handles, desc, B::bitwise_right_shift_scalar)
            }
        }
    }
}
