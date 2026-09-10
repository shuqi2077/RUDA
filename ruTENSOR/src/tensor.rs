use crate::{BinaryOp, ComputeType, Error, Mode, OperandDescriptor, OperationDescriptor, Plan, ReductionOp, Result};
use crate::einsum::{infer_output, output_dtype};
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};

/// Sum the product of A and B over modes absent from the output.
pub fn contract<R: Runtime>(
    a: &RudaTensor<R>, modes_a: &[Mode], b: &RudaTensor<R>, modes_b: &[Mode], output_modes: &[Mode],
) -> Result<RudaTensor<R>> {
    let operands = vec![OperandDescriptor::from_tensor(a, modes_a)?, OperandDescriptor::from_tensor(b, modes_b)?];
    let output = infer_output(&operands, output_modes, output_dtype(&operands))?;
    let compute = ComputeType::for_operands(&operands);
    Plan::new(OperationDescriptor::sum_product(operands, None, output, output_modes, compute)?)?
        .execute(&[a, b], &[1.0])
}

/// Reduce modes absent from `output_modes`, in the specified output order.
pub fn reduce<R: Runtime>(
    input: &RudaTensor<R>, input_modes: &[Mode], output_modes: &[Mode], reduction: ReductionOp,
) -> Result<RudaTensor<R>> {
    let operand = OperandDescriptor::from_tensor(input, input_modes)?;
    let output = infer_output(std::slice::from_ref(&operand), output_modes, input.dtype)?;
    let compute = ComputeType::for_operands(std::slice::from_ref(&operand));
    Plan::new(OperationDescriptor::reduction(operand, None, output, output_modes, reduction, compute)?)?
        .execute(&[input], &[1.0])
}

/// Materialize a permutation in a new packed row-major tensor; do not return an aliased view.
pub fn permute<R: Runtime>(input: &RudaTensor<R>, axes: &[usize]) -> Result<RudaTensor<R>> {
    let rank = input.meta.rank();
    if axes.len() != rank || axes.iter().any(|&axis| axis >= rank)
        || axes.iter().enumerate().any(|(i, axis)| axes[..i].contains(axis)) {
        return Err(Error::InvalidOperation("axes must be a permutation of 0..rank".into()));
    }
    let modes = (0..rank).map(|axis| Mode::try_from(axis).map_err(|_| Error::Overflow)).collect::<Result<Vec<_>>>()?;
    let output_modes: Vec<_> = axes.iter().map(|&axis| modes[axis]).collect();
    let operand = OperandDescriptor::from_tensor(input, &modes)?;
    let output = infer_output(std::slice::from_ref(&operand), &output_modes, input.dtype)?;
    let compute = ComputeType::for_operands(std::slice::from_ref(&operand));
    Plan::new(OperationDescriptor::permutation(operand, output, &output_modes, compute)?)?
        .execute(&[input], &[1.0])
}

/// Named-axis broadcasting followed by a binary operation.
pub fn elementwise_binary<R: Runtime>(
    a: &RudaTensor<R>, modes_a: &[Mode], b: &RudaTensor<R>, modes_b: &[Mode],
    output_modes: &[Mode], combine: BinaryOp,
) -> Result<RudaTensor<R>> {
    let a_desc = OperandDescriptor::from_tensor(a, modes_a)?;
    let b_desc = OperandDescriptor::from_tensor(b, modes_b)?;
    let operands = [a_desc, b_desc];
    let output = infer_output(&operands, output_modes, output_dtype(&operands))?;
    let compute = ComputeType::for_operands(&operands);
    let [a_desc, b_desc] = operands;
    Plan::new(OperationDescriptor::elementwise_binary(a_desc, b_desc, output, output_modes, combine, compute)?)?
        .execute(&[a, b], &[1.0, 1.0])
}

/// Combine three broadcast operands as `(A op_ab B) op_abc C`.
pub fn elementwise_trinary<R: Runtime>(
    inputs: [(&RudaTensor<R>, &[Mode]); 3], output_modes: &[Mode],
    combine_ab: BinaryOp, combine_abc: BinaryOp,
) -> Result<RudaTensor<R>> {
    let operands = [OperandDescriptor::from_tensor(inputs[0].0, inputs[0].1)?,
        OperandDescriptor::from_tensor(inputs[1].0, inputs[1].1)?,
        OperandDescriptor::from_tensor(inputs[2].0, inputs[2].1)?];
    let output = infer_output(&operands, output_modes, output_dtype(&operands))?;
    let compute = ComputeType::for_operands(&operands);
    Plan::new(OperationDescriptor::elementwise_trinary(operands, output, output_modes, combine_ab, combine_abc, compute)?)?
        .execute(&[inputs[0].0, inputs[1].0, inputs[2].0], &[1.0, 1.0, 1.0])
}
