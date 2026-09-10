use super::*;

impl<B: BackendIr> Runner<B> {
    pub(super) fn run_boolean(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        op: &BoolOperationIr,
    ) {
        match op {
            BoolOperationIr::IntoFloat(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.input);

                let output = B::bool_into_float(tensor, desc.out.dtype.into());
                handles.register_float_tensor::<B>(&desc.out.id, output);
            }
            BoolOperationIr::IntoInt(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.input);

                let output = B::bool_into_int(tensor, desc.out.dtype.into());
                handles.register_int_tensor::<B>(&desc.out.id, output);
            }
            BoolOperationIr::Not(desc) => {
                let tensor = handles.get_bool_tensor::<B>(&desc.input);

                let output = B::bool_not(tensor);
                handles.register_bool_tensor::<B>(&desc.out.id, output);
            }
            BoolOperationIr::And(desc) => {
                binary_bool_ops!(handles, desc, B::bool_and)
            }
            BoolOperationIr::Or(desc) => {
                binary_bool_ops!(handles, desc, B::bool_or)
            }
        }
    }
}
