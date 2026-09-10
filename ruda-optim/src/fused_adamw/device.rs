// SPDX-License-Identifier: Apache-2.0
use super::{AdamWOptions, FusedAdamWError, StepControl, config::checked_elements, kernel};
use ruda_core::{device::Device, tensor::DType};
use ruda_kernel::{
    dsl::{Runtime, calculate_ruda_count_elemwise, prelude::RudaDim},
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype},
};

/// Device-resident FP32 moments. Construct restored state with [`Self::from_parts`].
///
/// The completed-step counter belongs to this parameter/bucket, not a global
/// training iteration. All components must be checkpointed together.
pub struct AdamWState<R: Runtime> {
    step: u64,
    first: RudaTensor<R>,
    second: RudaTensor<R>,
    maximum: Option<RudaTensor<R>>,
}
impl<R: Runtime> Clone for AdamWState<R> {
    fn clone(&self) -> Self {
        Self { step: self.step, first: self.first.clone(), second: self.second.clone(), maximum: self.maximum.clone() }
    }
}
impl<R: Runtime> AdamWState<R> {
    /// Reassemble a checkpoint. Tensor metadata and queue are checked on the next
    /// call to [`adamw_step`]; data finiteness is deliberately not read back here.
    pub fn from_parts(
        step: u64, first: RudaTensor<R>, second: RudaTensor<R>, maximum: Option<RudaTensor<R>>,
    ) -> Result<Self, FusedAdamWError> {
        if step == 0 { return Err(FusedAdamWError::InvalidState("restored step must be positive")); }
        Ok(Self { step, first, second, maximum })
    }
    /// Number of scheduled, nonempty, non-skipped updates.
    pub fn step(&self) -> u64 { self.step }
    /// Uncorrected first moment, stored in FP32.
    pub fn first_moment(&self) -> &RudaTensor<R> { &self.first }
    /// Uncorrected second moment, stored in FP32.
    pub fn second_moment(&self) -> &RudaTensor<R> { &self.second }
    /// Running maximum of the uncorrected second moment, for AMSGrad only.
    pub fn max_second_moment(&self) -> Option<&RudaTensor<R>> { self.maximum.as_ref() }
    /// Consume the state for explicit checkpoint integration. No transfer occurs.
    pub fn into_parts(self) -> (u64, RudaTensor<R>, RudaTensor<R>, Option<RudaTensor<R>>) {
        (self.step, self.first, self.second, self.maximum)
    }
}

/// Out-of-place result. Inputs and any external aliases remain unchanged.
pub struct AdamWUpdate<R: Runtime> {
    /// FP32 master parameters, without a hidden cast to model storage precision.
    pub parameters: RudaTensor<R>,
    /// An empty/skipped first call leaves this `None`.
    pub state: Option<AdamWState<R>>,
    /// `true` means one update kernel was submitted, not device completion.
    pub updated: bool,
}

/// Update one tensor or a pre-flattened homogeneous parameter bucket in one kernel.
///
/// Contract:
/// - Parameters/moments: dense contiguous FP32; gradient: F32/F16/BF16, same shape.
/// - All operands use the same device and submission queue. Views with padding or
///   transposes must be made contiguous explicitly; no hidden materialization.
/// - State is initialized in the first update kernel, without zero-fill kernels.
/// - Empty tensors and `skip_update` do not allocate, submit or advance the counter.
/// - Finite gradients are the caller's responsibility. `skip_update` can be set by
///   an external overflow check; this function does not synchronize/read a flag.
/// - Calls are asynchronous. Synchronize the runtime before declaring completion,
///   saving a checkpoint or reporting benchmark time. Device failure is **not** a
///   transactional commit: discard the returned pending state on runtime failure.
/// - No differentiable-optimizer/autograd graph or model adaptor is installed.
///
/// The one-launch claim excludes explicit input preparation/model casts and applies
/// only to the nonempty, non-skipped contiguous path. Inputs are read-only, so even
/// aliased inputs are safe. Returned parameter/state buffers are newly allocated.
pub fn adamw_step<R: Runtime>(
    parameters: &RudaTensor<R>,
    gradients: &RudaTensor<R>,
    state: Option<&AdamWState<R>>,
    options: &AdamWOptions,
    control: StepControl,
) -> Result<AdamWUpdate<R>, FusedAdamWError> {
    adamw_step_scaled(parameters, gradients, state, options, control, 1.0)
}

// Clip factor is deliberately applied AFTER FP32 unscale, not folded into the
// reciprocal loss scale: folding changes overflow detection and rounding.
pub(super) fn adamw_step_scaled<R: Runtime>(
    parameters: &RudaTensor<R>,
    gradients: &RudaTensor<R>,
    state: Option<&AdamWState<R>>,
    options: &AdamWOptions,
    control: StepControl,
    clip_multiplier: f32,
) -> Result<AdamWUpdate<R>, FusedAdamWError> {
    if !clip_multiplier.is_finite() || !(0.0..=1.0).contains(&clip_multiplier) {
        return Err(FusedAdamWError::InvalidOption("clip_multiplier"));
    }
    let size = validate_step(parameters, gradients, state, options, control)?;
    if size == 0 || control.skip_update {
        return Ok(AdamWUpdate { parameters: parameters.clone(), state: state.cloned(), updated: false });
    }
    let coefficients = options.prepare_step(state.map_or(0, |s| s.step), control)?;
    let client = parameters.client.clone();
    let allocate = || empty_device_contiguous_dtype(
        client.clone(), parameters.device.clone(), parameters.meta.shape().clone(), DType::F32,
    );
    let new_parameters = allocate();
    let new_first = allocate();
    let new_second = allocate();
    let new_maximum = options.amsgrad.then(allocate);

    // Always bind valid allocations. For the first step the input moment arguments
    // are not read, and when AMSGrad is off its output argument is never written.
    // These branches are compile-time choices, not tests on uninitialized data.
    let old_first = state.map_or(parameters, |s| &s.first);
    let old_second = state.map_or(parameters, |s| &s.second);
    let old_maximum = state.and_then(|s| s.maximum.as_ref()).unwrap_or(parameters);
    let maximum_output = new_maximum.as_ref().unwrap_or(&new_second);
    let dim = RudaDim::new(client.properties(), size);
    kernel::adamw::launch::<R>(
        &client, calculate_ruda_count_elemwise(&client, size, dim), dim,
        parameters.clone().into_array_arg(), gradients.clone().into_array_arg(),
        old_first.clone().into_array_arg(), old_second.clone().into_array_arg(), old_maximum.clone().into_array_arg(),
        new_parameters.clone().into_array_arg(), new_first.clone().into_array_arg(),
        new_second.clone().into_array_arg(), maximum_output.clone().into_array_arg(),
        options.learning_rate, options.beta1, options.beta2, options.epsilon,
        coefficients.decay_multiplier, coefficients.inverse_bias1, coefficients.inverse_bias2,
        coefficients.inverse_gradient_scale, clip_multiplier,
        state.is_some(), options.amsgrad, options.maximize, clip_multiplier != 1.0,
        // Scope the cache identity to this kernel's algorithm, not only its name.
        concat!(include_str!("kernel.rs"), include_str!("config.rs")).to_owned(),
        gradients.dtype.into(),
    );
    Ok(AdamWUpdate {
        parameters: new_parameters,
        state: Some(AdamWState { step: coefficients.step, first: new_first, second: new_second, maximum: new_maximum }),
        updated: true,
    })
}

// Shared preflight: the guarded group validates EVERY entry before any kernel
// launch, and checks step overflow before calculating gradient statistics.
pub(super) fn validate_step<R: Runtime>(
    parameters: &RudaTensor<R>, gradients: &RudaTensor<R>,
    state: Option<&AdamWState<R>>, options: &AdamWOptions, control: StepControl,
) -> Result<usize, FusedAdamWError> {
    options.validate()?;
    control.validate()?;
    let size = validate_tensor(parameters, false)?;
    validate_related(parameters, gradients, true, "gradient")?;
    if let Some(state) = state {
        if state.step == 0 || state.maximum.is_some() != options.amsgrad {
            return Err(FusedAdamWError::InvalidState("step/AMSGrad mode"));
        }
        validate_related(parameters, &state.first, false, "first moment")?;
        validate_related(parameters, &state.second, false, "second moment")?;
        if let Some(maximum) = &state.maximum {
            validate_related(parameters, maximum, false, "max second moment")?;
        }
    }
    if size != 0 && !control.skip_update {
        state.map_or(0, |s| s.step).checked_add(1).ok_or(FusedAdamWError::StepOverflow)?;
    }
    Ok(size)
}

fn validate_related<R: Runtime>(
    parameters: &RudaTensor<R>, tensor: &RudaTensor<R>, gradient: bool, name: &'static str,
) -> Result<(), FusedAdamWError> {
    validate_tensor(tensor, gradient)?;
    if tensor.meta.shape() != parameters.meta.shape() {
        return Err(FusedAdamWError::ShapeMismatch(name));
    }
    if tensor.device.to_id() != parameters.device.to_id()
        || !tensor.client.same_execution_queue(&parameters.client)
    {
        return Err(FusedAdamWError::DifferentExecutionQueue(name));
    }
    Ok(())
}

pub(super) fn validate_tensor<R: Runtime>(tensor: &RudaTensor<R>, gradient: bool) -> Result<usize, FusedAdamWError> {
    let dtype_ok = if gradient {
        matches!(tensor.dtype, DType::F32 | DType::F16 | DType::BF16)
    } else { tensor.dtype == DType::F32 };
    if !dtype_ok || tensor.qparams.is_some() {
        return Err(FusedAdamWError::InvalidTensor("parameters/moments require FP32; gradients require F32/F16/BF16"));
    }
    let size = checked_elements(tensor.meta.shape())?;
    if tensor.meta.shape().len() != tensor.meta.strides().len() {
        return Err(FusedAdamWError::InvalidTensor("shape/stride rank mismatch"));
    }
    if size != 0 && !tensor.is_contiguous() {
        return Err(FusedAdamWError::InvalidTensor("explicit contiguous inputs required"));
    }
    let bytes_per_element = tensor.dtype.size() as u64;
    let start = tensor.handle.offset_start.unwrap_or(0);
    let end = tensor.handle.offset_end.unwrap_or(0);
    let usable = tensor.handle.size().checked_sub(start).and_then(|n| n.checked_sub(end))
        .ok_or(FusedAdamWError::InvalidTensor("invalid storage offsets"))?;
    if start % bytes_per_element != 0 || usable < size as u64 * bytes_per_element {
        return Err(FusedAdamWError::InvalidTensor("misaligned or undersized tensor storage"));
    }
    Ok(size)
}
