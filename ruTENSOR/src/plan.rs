use std::collections::BTreeMap;
use crate::{Error, Mode, OperationDescriptor, Result};
use crate::descriptor::product;
use crate::kernel::{Config, Launch, convert_operand};
use crate::operation::Kind;
use ruda_core::{device::Device, tensor::{Metadata, Shape}};
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};

/// A reusable index map and execution description, independent of buffer addresses.
#[derive(Clone, Debug)]
pub struct Plan {
    operation: OperationDescriptor,
    modes: Vec<Mode>,
    reduction_extents: Vec<usize>,
    reduction_count: usize,
    /// Logical mode index of every physical axis of each operand.
    axis_modes: Vec<Vec<usize>>,
}

pub(crate) fn merge_extent(extents: &mut BTreeMap<Mode, usize>, mode: Mode, value: usize) -> Result<()> {
    if let Some(&previous) = extents.get(&mode) {
        if previous != value && previous != 1 && value != 1 {
            return Err(Error::IncompatibleExtent { mode, left: previous, right: value });
        }
        // Broadcasting 0 with 1 produces an empty dimension, not a size-one dimension.
        extents.insert(mode, if previous == 1 { value } else { previous });
    } else {
        extents.insert(mode, value);
    }
    Ok(())
}

impl Plan {
    /// Resolve free, broadcast, diagonal and reduction axes without executing a kernel.
    pub fn new(operation: OperationDescriptor) -> Result<Self> {
        let mut extents = BTreeMap::new();
        for input in &operation.inputs {
            for (&mode, &extent) in input.modes.iter().zip(&input.tensor.extents) {
                merge_extent(&mut extents, mode, extent)?;
            }
        }
        for (&mode, &extent) in operation.output_modes.iter().zip(&operation.output.extents) {
            // Explicit output descriptors may introduce broadcast-only axes.
            merge_extent(&mut extents, mode, extent)?;
            if extents[&mode] != extent {
                return Err(Error::InvalidOperation(format!("output extent for mode {mode} would discard values")));
            }
        }
        let mut reduction_modes = Vec::new();
        for input in &operation.inputs[..operation.terms] {
            for &mode in &input.modes {
                if !operation.output_modes.contains(&mode) && !reduction_modes.contains(&mode) {
                    reduction_modes.push(mode);
                }
            }
        }
        if operation.addend && operation.inputs[operation.terms].modes.iter()
            .any(|mode| !operation.output_modes.contains(mode)) {
            return Err(Error::InvalidOperation("C may only contain output modes".into()));
        }
        if !reduction_modes.is_empty() && matches!(operation.kind, Kind::Permutation | Kind::Elementwise(_, _)) {
            return Err(Error::InvalidOperation("elementwise and permutation operations cannot discard modes".into()));
        }
        if operation.kind == Kind::Permutation {
            let input = &operation.inputs[0];
            if input.modes.len() != operation.output_modes.len()
                || input.modes.iter().any(|mode| !operation.output_modes.contains(mode))
                || input.modes.iter().enumerate().any(|(i, mode)| input.modes[..i].contains(mode)) {
                return Err(Error::InvalidOperation("a permutation must reorder each input mode exactly once".into()));
            }
            for (i, mode) in input.modes.iter().enumerate() {
                if input.tensor.extents[i] != extents[mode] {
                    return Err(Error::InvalidOperation("a permutation cannot expand an input axis".into()));
                }
            }
        }
        let reduction_extents: Vec<_> = reduction_modes.iter().map(|m| extents[m]).collect();
        let reduction_count = product(&reduction_extents)?;
        let mut modes = operation.output_modes.clone();
        modes.extend(reduction_modes);
        let axis_modes = operation.inputs.iter().map(|input| input.modes.iter()
            .map(|m| modes.iter().position(|x| x == m).expect("validated mode"))
            .collect()).collect();
        Ok(Self { operation, modes, reduction_extents, reduction_count, axis_modes })
    }

    pub fn operation(&self) -> &OperationDescriptor { &self.operation }
    pub fn reduction_extents(&self) -> &[usize] { &self.reduction_extents }
    pub fn reduction_elements(&self) -> usize { self.reduction_count }

    /// Allocate D and enqueue the operation. All inputs remain unchanged.
    pub fn execute<R: Runtime>(&self, inputs: &[&RudaTensor<R>], scalars: &[f64]) -> Result<RudaTensor<R>> {
        self.validate_inputs(inputs, scalars)?;
        let reference = inputs[0];
        let descriptor = &self.operation.output;
        let handle = reference.client.empty(descriptor.storage_bytes().max(descriptor.dtype.size()));
        let output = RudaTensor::new(reference.client.clone(), handle,
            Metadata::new(Shape::from(descriptor.extents.clone()), descriptor.strides.clone()),
            reference.device.clone(), descriptor.dtype);
        self.submit(inputs, output, scalars)
    }

    /// Write into an exclusively owned D buffer and return it. D cannot alias any input.
    pub fn execute_into<R: Runtime>(
        &self, inputs: &[&RudaTensor<R>], output: RudaTensor<R>, scalars: &[f64],
    ) -> Result<RudaTensor<R>> {
        self.validate_inputs(inputs, scalars)?;
        if !self.operation.output.matches(&output) { return Err(Error::OutputMismatch); }
        self.operation.output.check_buffer(&output)?;
        if inputs[0].device.to_id() != output.device.to_id() { return Err(Error::DeviceMismatch); }
        if !output.can_mut() { return Err(Error::SharedOutput); }
        self.submit(inputs, output, scalars)
    }

    fn validate_inputs<R: Runtime>(&self, inputs: &[&RudaTensor<R>], scalars: &[f64]) -> Result<()> {
        if inputs.len() != self.operation.inputs.len() {
            return Err(Error::InputCount { expected: self.operation.inputs.len(), actual: inputs.len() });
        }
        if scalars.len() != self.operation.scalar_count() {
            return Err(Error::ScalarCount { expected: self.operation.scalar_count(), actual: scalars.len() });
        }
        for (index, (input, descriptor)) in inputs.iter().zip(&self.operation.inputs).enumerate() {
            if !descriptor.tensor.matches(input) { return Err(Error::TensorMismatch { input: index }); }
            descriptor.tensor.check_buffer(input)?;
            if inputs[0].device.to_id() != input.device.to_id() { return Err(Error::DeviceMismatch); }
        }
        Ok(())
    }

    fn submit<R: Runtime>(
        &self, inputs: &[&RudaTensor<R>], output: RudaTensor<R>, scalars: &[f64],
    ) -> Result<RudaTensor<R>> {
        let count = self.operation.output.num_elements();
        if count == 0 { return Ok(output); }
        let hardware = &output.client.properties().hardware;
        let mut maximum = (hardware.max_ruda_dim.0 as usize)
            .min(hardware.max_units_per_ruda as usize).min(128);
        if !self.reduction_extents.is_empty() {
            maximum = maximum.min(hardware.max_shared_memory_size / self.operation.compute.dtype().size());
        }
        if maximum == 0 {
            return Err(Error::UnsupportedDevice("no workgroup threads available".into()));
        }
        let desired = maximum.min(self.reduction_count.max(1));
        let threads = 1usize << desired.ilog2();
        count.checked_mul(threads).ok_or(Error::Overflow)?;
        let compute = self.operation.compute;
        let converted = inputs.iter().map(|input| convert_operand(input, compute.dtype()))
            .collect::<Result<Vec<_>>>()?;
        let stride_count = self.modes.len().checked_mul(inputs.len()).ok_or(Error::Overflow)?;
        let mut strides = vec![0usize; stride_count];
        for (operand, tensor) in converted.iter().enumerate() {
            for (axis, &mode) in self.axis_modes[operand].iter().enumerate() {
                if tensor.meta.shape()[axis] != 1 {
                    let position = operand * self.modes.len() + mode;
                    strides[position] = strides[position].checked_add(tensor.meta.strides()[axis])
                        .ok_or(Error::Overflow)?;
                }
            }
        }
        Launch {
            inputs: &converted, output: &output, strides,
            reduction_extents: &self.reduction_extents, scalars, count,
            reduction_count: self.reduction_count, threads, compute,
            config: Config {
                kind: self.operation.kind,
                unary: self.operation.inputs.iter().map(|x| x.unary).collect(),
                terms: self.operation.terms, addend: self.operation.addend,
            },
        }.submit()?;
        Ok(output)
    }
}
