use std::collections::BTreeMap;
use crate::{ComputeType, Error, Mode, OperandDescriptor, OperationDescriptor, Plan, Result, TensorDescriptor};
use crate::plan::merge_extent;
use ruda_core::tensor::DType;
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};

#[derive(Clone, Debug)]
enum Token { Label(Mode), Ellipsis }

fn tokens(text: &str) -> Result<Vec<Token>> {
    let mut result = Vec::new();
    let bytes = text.as_bytes();
    let mut index = 0;
    let mut ellipsis = false;
    while index < bytes.len() {
        if bytes[index].is_ascii_alphabetic() {
            result.push(Token::Label(bytes[index] as Mode));
            index += 1;
        } else if bytes[index..].starts_with(b"...") && !ellipsis {
            result.push(Token::Ellipsis);
            ellipsis = true;
            index += 3;
        } else {
            return Err(Error::InvalidExpression("use ASCII letters and at most one ellipsis per operand".into()));
        }
    }
    Ok(result)
}

fn expand(tokens: &[Token], ellipsis_modes: &[Mode]) -> Vec<Mode> {
    let mut modes = Vec::new();
    for token in tokens {
        match token {
            Token::Label(mode) => modes.push(*mode),
            Token::Ellipsis => modes.extend_from_slice(ellipsis_modes),
        }
    }
    modes
}

pub(crate) fn output_dtype(inputs: &[OperandDescriptor]) -> DType {
    let first = inputs[0].tensor().dtype();
    if inputs.iter().all(|input| input.tensor().dtype() == first) { return first; }
    if inputs.iter().any(|input| input.tensor().dtype() == DType::F64) { DType::F64 } else { DType::F32 }
}

pub(crate) fn infer_output(inputs: &[OperandDescriptor], modes: &[Mode], dtype: DType) -> Result<TensorDescriptor> {
    let mut extents = BTreeMap::new();
    for input in inputs {
        for (&mode, &extent) in input.modes().iter().zip(input.tensor().extents()) {
            merge_extent(&mut extents, mode, extent)?;
        }
    }
    let shape = modes.iter().map(|mode| extents.get(mode).copied().ok_or_else(||
        Error::InvalidOperation(format!("output mode {mode} is absent from all inputs"))))
        .collect::<Result<Vec<_>>>()?;
    TensorDescriptor::contiguous(&shape, dtype)
}

/// A reusable einsum expression with fixed input shapes, strides and storage types.
#[derive(Clone, Debug)]
pub struct EinsumPlan {
    expression: String,
    plan: Plan,
}

impl EinsumPlan {
    pub fn new(expression: &str, inputs: &[TensorDescriptor]) -> Result<Self> {
        Self::build(expression, inputs, None)
    }

    pub fn with_options(
        expression: &str, inputs: &[TensorDescriptor], output_dtype: DType, compute: ComputeType,
    ) -> Result<Self> {
        Self::build(expression, inputs, Some((output_dtype, compute)))
    }

    fn build(expression: &str, inputs: &[TensorDescriptor], options: Option<(DType, ComputeType)>) -> Result<Self> {
        if inputs.is_empty() {
            return Err(Error::InvalidExpression("at least one tensor is required".into()));
        }
        let expression: String = expression.chars().filter(|c| !c.is_ascii_whitespace()).collect();
        let mut sides = expression.split("->");
        let lhs = sides.next().unwrap_or_default();
        let rhs = sides.next();
        if sides.next().is_some() {
            return Err(Error::InvalidExpression("more than one output arrow".into()));
        }
        let input_text: Vec<_> = lhs.split(',').collect();
        if input_text.len() != inputs.len() {
            return Err(Error::InputCount { expected: input_text.len(), actual: inputs.len() });
        }
        let token_lists: Vec<_> = input_text.iter().map(|text| tokens(text)).collect::<Result<_>>()?;
        let mut widths = Vec::new();
        let mut counts = BTreeMap::<Mode, usize>::new();
        for (tokens, descriptor) in token_lists.iter().zip(inputs) {
            let explicit = tokens.iter().filter(|token| matches!(token, Token::Label(_))).count();
            let has_ellipsis = tokens.iter().any(|token| matches!(token, Token::Ellipsis));
            if explicit > descriptor.rank() || (!has_ellipsis && explicit != descriptor.rank()) {
                return Err(Error::InvalidExpression("operand labels do not match its tensor rank".into()));
            }
            widths.push(descriptor.rank() - explicit);
            for token in tokens {
                if let Token::Label(mode) = token { *counts.entry(*mode).or_default() += 1; }
            }
        }
        let width = widths.iter().copied().max().unwrap_or(0);
        let ellipsis_modes = (0..width).map(|axis| {
            let axis = i32::try_from(axis).map_err(|_| Error::Overflow)?;
            i32::MIN.checked_add(axis).ok_or(Error::Overflow)
        }).collect::<Result<Vec<_>>>()?;
        let mut operands = Vec::new();
        for ((tokens, descriptor), local_width) in token_lists.iter().zip(inputs).zip(widths) {
            let modes = expand(tokens, &ellipsis_modes[width - local_width..]);
            operands.push(OperandDescriptor::new(descriptor.clone(), &modes)?);
        }
        let output_modes = if let Some(rhs) = rhs {
            expand(&tokens(rhs)?, &ellipsis_modes)
        } else {
            let mut modes = ellipsis_modes;
            modes.extend(counts.iter().filter_map(|(&mode, &count)| (count == 1).then_some(mode)));
            modes
        };
        let (dtype, compute) = options.unwrap_or_else(||
            (output_dtype(&operands), ComputeType::for_operands(&operands)));
        let output = infer_output(&operands, &output_modes, dtype)?;
        let operation = OperationDescriptor::sum_product(operands, None, output, &output_modes, compute)?;
        Ok(Self { expression, plan: Plan::new(operation)? })
    }

    pub fn expression(&self) -> &str { &self.expression }
    pub fn plan(&self) -> &Plan { &self.plan }
    pub fn output(&self) -> &TensorDescriptor { self.plan.operation().output() }

    pub fn execute<R: Runtime>(&self, inputs: &[&RudaTensor<R>]) -> Result<RudaTensor<R>> {
        self.plan.execute(inputs, &[1.0])
    }

    pub fn execute_into<R: Runtime>(&self, inputs: &[&RudaTensor<R>], output: RudaTensor<R>) -> Result<RudaTensor<R>> {
        self.plan.execute_into(inputs, output, &[1.0])
    }
}

/// Evaluate an einsum expression on device tensors without reading values back to the host.
///
/// Ellipses are right-aligned for broadcasting. Repeated input labels select diagonals.
/// Explicit output labels select and order free axes; all other axes are summed.
pub fn einsum<R: Runtime>(expression: &str, inputs: &[&RudaTensor<R>]) -> Result<RudaTensor<R>> {
    let descriptors = inputs.iter().map(|input| TensorDescriptor::from_tensor(*input)).collect::<Result<Vec<_>>>()?;
    EinsumPlan::new(expression, &descriptors)?.execute(inputs)
}

pub fn einsum_with_options<R: Runtime>(
    expression: &str, inputs: &[&RudaTensor<R>], output_dtype: DType, compute: ComputeType,
) -> Result<RudaTensor<R>> {
    let descriptors = inputs.iter().map(|input| TensorDescriptor::from_tensor(*input)).collect::<Result<Vec<_>>>()?;
    EinsumPlan::with_options(expression, &descriptors, output_dtype, compute)?.execute(inputs)
}
