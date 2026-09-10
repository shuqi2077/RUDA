use crate::{BinaryOp, ComputeType, Error, ReductionOp, Result, UnaryOp};
use crate::operation::Kind;
use ruda_core::tensor::{DType, TensorMetadata};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_contiguous_dtype};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Config {
    pub kind: Kind,
    pub unary: Vec<UnaryOp>,
    pub terms: usize,
    pub addend: bool,
}

#[ruda]
fn unary<F: Float>(x: F, #[comptime] op: UnaryOp) -> F {
    match op {
        UnaryOp::Identity | UnaryOp::Conjugate => x,
        UnaryOp::Negate => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Relu => select(x != x, x, select(x > F::new(0.0), x, F::new(0.0))),
        UnaryOp::Reciprocal => F::new(1.0) / x,
    }
}

#[ruda]
fn binary<F: Float>(a: F, b: F, #[comptime] op: BinaryOp) -> F {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Mul => a * b,
        BinaryOp::Min => {
            let value = select(a < b, a, b);
            // Select -0 for min(+0, -0) independently of operand order.
            let zero = select(F::new(1.0) / a < F::new(0.0), a, b);
            let value = select(a == F::new(0.0) && b == F::new(0.0), zero, value);
            select(a != a, a, select(b != b, b, value))
        }
        BinaryOp::Max => {
            let value = select(a > b, a, b);
            let zero = select(F::new(1.0) / a > F::new(0.0), a, b);
            let value = select(a == F::new(0.0) && b == F::new(0.0), zero, value);
            select(a != a, a, select(b != b, b, value))
        }
    }
}

#[ruda]
fn identity<F: Float>(#[comptime] op: ReductionOp) -> F {
    match op {
        ReductionOp::Sum => F::new(0.0),
        ReductionOp::Product => F::new(1.0),
        ReductionOp::Min => F::new(f32::INFINITY),
        ReductionOp::Max => F::new(f32::NEG_INFINITY),
    }
}

#[ruda]
fn fold<F: Float>(a: F, b: F, #[comptime] op: ReductionOp) -> F {
    match op {
        ReductionOp::Sum => a + b,
        ReductionOp::Product => a * b,
        ReductionOp::Min => binary::<F>(a, b, BinaryOp::Min),
        ReductionOp::Max => binary::<F>(a, b, BinaryOp::Max),
    }
}

#[ruda]
fn read<F: Float>(
    input: &Tensor<F>, strides: &Sequence<usize>,
    output_shape: &Sequence<FastDivmod<usize>>,
    reduction_shape: &Sequence<FastDivmod<usize>>,
    output_index: usize, reduction_index: usize,
    #[comptime] operand: usize, #[comptime] op: UnaryOp,
) -> F {
    let output_rank = comptime![output_shape.len()];
    let rank = comptime![output_rank + reduction_shape.len()];
    let mut position = output_index;
    let mut offset = 0usize;
    #[unroll]
    for axis in 0..output_rank {
        let axis = comptime![output_rank - axis - 1];
        let (rest, coordinate) = output_shape[axis].div_mod(position);
        position = rest;
        offset += coordinate * strides[operand * rank + axis];
    }
    position = reduction_index;
    #[unroll]
    for axis in 0..reduction_shape.len() {
        let axis = comptime![reduction_shape.len() - axis - 1];
        let (rest, coordinate) = reduction_shape[axis].div_mod(position);
        position = rest;
        offset += coordinate * strides[operand * rank + output_rank + axis];
    }
    unary::<F>(input[offset], op)
}

#[ruda]
fn output_offset(
    output_index: usize, shape: &Sequence<FastDivmod<usize>>, strides: &Sequence<usize>,
) -> usize {
    let mut position = output_index;
    let mut offset = 0usize;
    #[unroll]
    for axis in 0..shape.len() {
        let axis = comptime![shape.len() - axis - 1];
        let (rest, coordinate) = shape[axis].div_mod(position);
        position = rest;
        offset += coordinate * strides[axis];
    }
    offset
}

#[ruda]
fn product_term<F: Float>(
    inputs: &Sequence<Tensor<F>>, strides: &Sequence<usize>,
    output_shape: &Sequence<FastDivmod<usize>>, reduction_shape: &Sequence<FastDivmod<usize>>,
    output_index: usize, reduction_index: usize, #[comptime] config: Config,
) -> F {
    let mut value = read::<F>(inputs.index(0usize), strides, output_shape, reduction_shape,
        output_index, reduction_index, 0usize, comptime![config.unary[0]]);
    #[unroll]
    for operand in 1..config.terms {
        value *= read::<F>(inputs.index(operand), strides, output_shape, reduction_shape,
            output_index, reduction_index, operand, comptime![config.unary[operand]]);
    }
    value
}

#[ruda]
fn epilogue<F: Float>(
    value: F, inputs: &Sequence<Tensor<F>>, strides: &Sequence<usize>,
    output_shape: &Sequence<FastDivmod<usize>>, reduction_shape: &Sequence<FastDivmod<usize>>,
    scalars: &Sequence<InputScalar>, output_index: usize, #[comptime] config: Config,
) -> F {
    let mut result = scalars[0].get::<F>() * value;
    if comptime![config.addend] {
        let beta = scalars[1].get::<F>();
        // A zero beta does not read C, including when C contains NaNs.
        if beta != F::new(0.0) {
            let addend = read::<F>(inputs.index(config.terms), strides, output_shape,
                reduction_shape, output_index, 0, config.terms,
                comptime![config.unary[config.terms]]);
            result += beta * addend;
        }
    }
    result
}

#[ruda(launch, address_type = "dynamic")]
fn pointwise<F: Float, O: Float>(
    inputs: Sequence<Tensor<F>>, output: &mut Tensor<O>,
    strides: Sequence<usize>, output_strides: Sequence<usize>,
    output_shape: Sequence<FastDivmod<usize>>, reduction_shape: Sequence<FastDivmod<usize>>,
    scalars: Sequence<InputScalar>, count: usize,
    #[comptime] config: Config, #[define(O)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    if index >= count { terminate!(); }
    let mut value = F::new(0.0);
    match config.kind {
        Kind::Elementwise(op_ab, op_abc) => {
            let a = scalars[0].get::<F>() * read::<F>(inputs.index(0usize), &strides,
                &output_shape, &reduction_shape, index, 0, 0usize, comptime![config.unary[0]]);
            let b = scalars[1].get::<F>() * read::<F>(inputs.index(1usize), &strides,
                &output_shape, &reduction_shape, index, 0, 1usize, comptime![config.unary[1]]);
            value = binary::<F>(a, b, op_ab);
            if comptime![config.terms == 3] {
                let c = scalars[2].get::<F>() * read::<F>(inputs.index(2usize), &strides,
                    &output_shape, &reduction_shape, index, 0, 2usize, comptime![config.unary[2]]);
                value = binary::<F>(value, c, op_abc);
            }
        }
        _ => {
            value = product_term::<F>(&inputs, &strides, &output_shape, &reduction_shape,
                index, 0, config.clone());
            value = epilogue::<F>(value, &inputs, &strides, &output_shape, &reduction_shape,
                &scalars, index, config);
        }
    }
    output[output_offset(index, &output_shape, &output_strides)] = O::cast_from(value);
}

#[ruda(launch, address_type = "dynamic")]
fn aggregate<F: Float, O: Float>(
    inputs: Sequence<Tensor<F>>, output: &mut Tensor<O>,
    strides: Sequence<usize>, output_strides: Sequence<usize>,
    output_shape: Sequence<FastDivmod<usize>>, reduction_shape: Sequence<FastDivmod<usize>>,
    scalars: Sequence<InputScalar>, count: usize, reduction_count: usize,
    #[comptime] threads: usize, #[comptime] reduction: ReductionOp,
    #[comptime] config: Config, #[define(O)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS / threads;
    if index >= count { terminate!(); }
    let lane = UNIT_POS as usize;
    let mut value = identity::<F>(reduction);
    let mut position = lane;
    while position < reduction_count {
        let next = product_term::<F>(&inputs, &strides, &output_shape, &reduction_shape,
            index, position, config.clone());
        value = fold::<F>(value, next, reduction);
        if reduction_count - position <= threads { break; }
        position += threads;
    }
    let mut shared = SharedMemory::<F>::new(threads);
    shared[lane] = value;
    let mut step = RUDA_DIM as usize / 2;
    while step > 0 {
        sync_ruda();
        if lane < step {
            shared[lane] = fold::<F>(shared[lane], shared[lane + step], reduction);
        }
        step /= 2;
    }
    if lane == 0 {
        value = epilogue::<F>(shared[0], &inputs, &strides, &output_shape, &reduction_shape,
            &scalars, index, config);
        output[output_offset(index, &output_shape, &output_strides)] = O::cast_from(value);
    }
}

#[ruda(launch, address_type = "dynamic")]
fn convert<I: Float, O: Float>(
    input: &Tensor<I>, output: &mut Tensor<O>,
    shape: Sequence<FastDivmod<usize>>, count: usize,
    #[define(I, O)] _dtypes: [StorageType; 2],
) {
    let index = ABSOLUTE_POS;
    if index >= count { terminate!(); }
    let mut position = index;
    let mut offset = 0usize;
    #[unroll]
    for axis in 0..shape.len() {
        let axis = comptime![shape.len() - axis - 1];
        let (rest, coordinate) = shape[axis].div_mod(position);
        position = rest;
        offset += coordinate * input.stride(axis);
    }
    output[index] = O::cast_from(input[offset]);
}

fn grid<R: Runtime>(client: &ComputeClient<R>, units: usize, dim: RudaDim) -> Result<(RudaCount, AddressType)> {
    let threads = dim.num_elems() as usize;
    let groups = units.div_ceil(threads);
    let limits = client.properties().hardware.max_ruda_count;
    if limits.0 == 0 || limits.1 == 0 || limits.2 == 0 {
        return Err(Error::UnsupportedDevice("empty dispatch grid".into()));
    }
    let x = groups.min(limits.0 as usize).max(1);
    let remaining = groups.div_ceil(x);
    let y = remaining.min(limits.1 as usize).max(1);
    let z = remaining.div_ceil(y).max(1);
    if z > limits.2 as usize {
        return Err(Error::UnsupportedDevice("operation exceeds the device dispatch grid".into()));
    }
    let launched = x.checked_mul(y).and_then(|n| n.checked_mul(z))
        .and_then(|n| n.checked_mul(threads)).ok_or(Error::Overflow)?;
    Ok((RudaCount::Static(x as u32, y as u32, z as u32), AddressType::from_len(launched)))
}

pub(crate) fn convert_operand<R: Runtime>(input: &RudaTensor<R>, dtype: DType) -> Result<RudaTensor<R>> {
    if input.dtype == dtype || input.meta.shape().contains(&0) { return Ok(input.clone()); }
    let count = input.meta.num_elements();
    count.checked_mul(dtype.size()).ok_or(Error::Overflow)?;
    let output = empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), input.shape(), dtype);
    let dim = RudaDim::new(input.client.properties(), count);
    let (ruda_count, address) = grid(&input.client, count, dim)?;
    convert::launch(&input.client, ruda_count, dim,
        address.max(input.required_address_type()).max(output.required_address_type()),
        input.clone().into_tensor_arg(), output.clone().into_tensor_arg(),
        input.meta.shape().iter().copied().collect::<SequenceArg<R, FastDivmod<usize>>>(),
        count, [input.dtype.into(), dtype.into()]);
    Ok(output)
}

pub(crate) struct Launch<'a, R: Runtime> {
    pub inputs: &'a [RudaTensor<R>],
    pub output: &'a RudaTensor<R>,
    pub strides: Vec<usize>,
    pub reduction_extents: &'a [usize],
    pub scalars: &'a [f64],
    pub count: usize,
    pub reduction_count: usize,
    pub threads: usize,
    pub compute: ComputeType,
    pub config: Config,
}

impl<R: Runtime> Launch<'_, R> {
    pub fn submit(self) -> Result<()> {
        match self.compute {
            ComputeType::F32 => self.submit_typed::<f32>(),
            ComputeType::F64 => self.submit_typed::<f64>(),
        }
    }

    fn submit_typed<F: Float>(self) -> Result<()> {
        let client = &self.output.client;
        let inputs = self.inputs.iter().map(|x| x.clone().into_tensor_arg())
            .collect::<SequenceArg<R, Tensor<F>>>();
        let strides = self.strides.into_iter().collect::<SequenceArg<R, usize>>();
        let output_strides = self.output.meta.strides().iter().copied().collect::<SequenceArg<R, usize>>();
        let output_shape = self.output.meta.shape().iter().map(|&d| d.max(1))
            .collect::<SequenceArg<R, FastDivmod<usize>>>();
        let reduction_shape = self.reduction_extents.iter().map(|&d| d.max(1))
            .collect::<SequenceArg<R, FastDivmod<usize>>>();
        let scalars = self.scalars.iter().map(|&x| InputScalar::new(x, self.compute.dtype()))
            .collect::<SequenceArg<R, InputScalar>>();
        let address = self.inputs.iter().fold(self.output.required_address_type(),
            |address, input| address.max(input.required_address_type()))
            .max(AddressType::from_len(self.count.max(self.reduction_count)))
            .max(AddressType::from_len(self.count * self.threads));
        if self.reduction_extents.is_empty() {
            let dim = RudaDim::new(client.properties(), self.count);
            let (ruda_count, grid_address) = grid(client, self.count, dim)?;
            pointwise::launch::<F, R>(client, ruda_count, dim,
                address.max(grid_address), inputs, self.output.clone().into_tensor_arg(), strides, output_strides,
                output_shape, reduction_shape, scalars, self.count, self.config, self.output.dtype.into());
        } else {
            let dim = RudaDim::new_1d(self.threads as u32);
            let reduction = match self.config.kind {
                Kind::Reduction(op) => op,
                _ => ReductionOp::Sum,
            };
            let (ruda_count, grid_address) = grid(client, self.count * self.threads, dim)?;
            aggregate::launch::<F, R>(client, ruda_count, dim,
                address.max(grid_address), inputs, self.output.clone().into_tensor_arg(), strides, output_strides,
                output_shape, reduction_shape, scalars, self.count, self.reduction_count,
                self.threads, reduction, self.config, self.output.dtype.into());
        }
        Ok(())
    }
}
