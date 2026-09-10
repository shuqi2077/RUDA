use super::*;
use alloc::vec;
use ruda_core::future::block_on;
use ruda_tensor::graph::{BinaryOpIr, LinearOpIr, ShapeOpIr, UnaryOpIr};
use ruda_tensor_host::Host;

fn output(runner: &Runner<Host>, shape: impl Into<Shape>, dtype: DType) -> TensorIr {
    TensorIr {
        id: runner.create_empty_handle(),
        shape: shape.into(),
        dtype,
        status: TensorStatus::NotInit,
    }
}

fn readable(mut tensor: TensorIr) -> TensorIr {
    tensor.status = TensorStatus::ReadOnly;
    tensor
}

#[test]
fn split_runner_dispatches_float_numeric_layout_and_elementwise_operations() {
    let runner = Runner::<Host>::new(Default::default());
    let lhs = runner.register_tensor_data_desc(TensorData::new(vec![0.0f32, 1.0, 4.0, 9.0], [4]));
    let rhs = runner.register_tensor_data_desc(TensorData::new(vec![1.0f32, 3.0, 5.0, 7.0], [4]));
    let sum = output(&runner, [4], DType::F32);
    runner.register_op(OperationIr::NumericFloat(
        DType::F32,
        NumericOperationIr::Add(BinaryOpIr {
            lhs,
            rhs,
            out: sum.clone(),
        }),
    ));
    let reshaped = output(&runner, [2, 2], DType::F32);
    runner.register_op(OperationIr::BaseFloat(BaseOperationIr::Reshape(
        ShapeOpIr {
            input: readable(sum),
            out: reshaped.clone(),
        },
    )));
    let square_root = output(&runner, [2, 2], DType::F32);
    runner.register_op(OperationIr::Float(
        DType::F32,
        FloatOperationIr::Sqrt(UnaryOpIr {
            input: readable(reshaped),
            out: square_root.clone(),
        }),
    ));
    let values = block_on(runner.read_tensor_async(readable(square_root))).unwrap();
    assert_eq!(values.shape, Shape::from([2, 2]));
    assert_eq!(values.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn split_runner_dispatches_integer_and_boolean_families_without_dtype_changes() {
    let runner = Runner::<Host>::new(Default::default());
    let lhs = runner.register_tensor_data_desc(TensorData::new(vec![1i32, 2], [2]));
    let rhs = runner.register_tensor_data_desc(TensorData::new(vec![3i32, 4], [2]));
    let sum = output(&runner, [2], DType::I32);
    runner.register_op(OperationIr::NumericInt(
        DType::I32,
        NumericOperationIr::Add(BinaryOpIr {
            lhs,
            rhs,
            out: sum.clone(),
        }),
    ));
    let reshaped = output(&runner, [1, 2], DType::I32);
    runner.register_op(OperationIr::BaseInt(BaseOperationIr::Reshape(ShapeOpIr {
        input: readable(sum),
        out: reshaped.clone(),
    })));
    let inverted = output(&runner, [1, 2], DType::I32);
    runner.register_op(OperationIr::Int(IntOperationIr::BitwiseNot(UnaryOpIr {
        input: readable(reshaped),
        out: inverted.clone(),
    })));
    assert_eq!(
        block_on(runner.read_tensor_async(readable(inverted)))
            .unwrap()
            .to_vec::<i32>()
            .unwrap(),
        vec![!4, !6]
    );
    let input = runner.register_tensor_data_desc(TensorData::new(vec![true, false], [2]));
    let dtype = input.dtype;
    let reshaped = output(&runner, [1, 2], dtype);
    runner.register_op(OperationIr::BaseBool(BaseOperationIr::Reshape(ShapeOpIr {
        input,
        out: reshaped.clone(),
    })));
    let inverted = output(&runner, [1, 2], dtype);
    runner.register_op(OperationIr::Bool(BoolOperationIr::Not(UnaryOpIr {
        input: readable(reshaped),
        out: inverted.clone(),
    })));
    let result = block_on(runner.read_tensor_async(readable(inverted))).unwrap();
    assert_eq!(result.dtype, dtype);
    assert_eq!(result.to_vec::<bool>().unwrap(), vec![false, true]);
}

#[test]
fn split_runner_module_linear_preserves_optional_bias_and_handle_registration() {
    let runner = Runner::<Host>::new(Default::default());
    let x = runner.register_tensor_data_desc(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]));
    let weight =
        runner.register_tensor_data_desc(TensorData::new(vec![2.0f32, 0.0, 0.0, 3.0], [2, 2]));
    let bias = runner.register_tensor_data_desc(TensorData::new(vec![0.5f32, -0.5], [2]));
    let out = output(&runner, [2, 2], DType::F32);
    runner.register_op(OperationIr::Module(ModuleOperationIr::Linear(LinearOpIr {
        x,
        weight,
        bias: Some(bias),
        out: out.clone(),
    })));
    let values = block_on(runner.read_tensor_async(readable(out.clone())))
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(values, vec![2.5, 5.5, 6.5, 11.5]);
    runner.register_op(OperationIr::Drop(out.clone()));
    assert!(
        runner
            .context
            .lock()
            .unwrap()
            .handles
            .remove_handle(out.id)
            .is_none()
    );
}

#[test]
fn split_runner_clones_share_handles_and_read_future_owns_its_result() {
    let runner = Runner::<Host>::new(Default::default());
    let first_id = runner.create_empty_handle();
    let cloned = runner.clone();
    let second_id = cloned.create_empty_handle();
    assert_ne!(first_id, second_id);
    let tensor = runner.register_tensor_data(TensorData::new(vec![7.0f32, 11.0], [2]));
    let alias = tensor.clone();
    assert_eq!(tensor.id, alias.id);
    drop(tensor);
    drop(runner);
    let future = cloned.read_tensor_async(alias.into_ir());
    drop(cloned);
    assert_eq!(
        block_on(future).unwrap().to_vec::<f32>().unwrap(),
        vec![7.0, 11.0]
    );
}
