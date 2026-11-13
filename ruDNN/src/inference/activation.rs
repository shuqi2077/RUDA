//! Native gated activations, independent of any vendor inference runtime.
use super::{InferenceError, alloc, check, float};
use ruda_core::tensor::Shape;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Identity,
    Silu,
    Gelu,
    GeluTanh,
    Relu,
    ReluSquared,
}
impl Activation {
    pub(crate) fn code(self) -> u32 {
        match self {
            Self::Identity => 0,
            Self::Silu => 1,
            Self::Gelu => 2,
            Self::GeluTanh => 3,
            Self::Relu => 4,
            Self::ReluSquared => 5,
        }
    }
}

/// Gate/up halves are concatenated along the last axis. Other axes are
/// arbitrary packed rows; tails need not be vector-width aligned.
pub fn gated<R: Runtime>(
    input: &RudaTensor<R>,
    activation: Activation,
) -> Result<RudaTensor<R>, InferenceError> {
    check(input, input)?;
    float(input)?;
    let mut shape = input.meta.shape().to_vec();
    let last = shape
        .last_mut()
        .ok_or(InferenceError("gated activation needs a final axis"))?;
    if *last == 0 || *last % 2 != 0 {
        return Err(InferenceError(
            "gate and up halves must have equal nonempty width",
        ));
    }
    *last /= 2;
    let width = *last;
    let n = input.meta.num_elements() / 2;
    let out = alloc(input, Shape::from(shape), input.dtype);
    if n != 0 {
        let dim = RudaDim::new(input.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, n, dim);
        gate::launch::<R>(
            &input.client,
            count,
            dim,
            input.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            width as u32,
            activation.code(),
            input.dtype.into(),
        );
    }
    Ok(out)
}
/// Preallocated gated activation for graph replay. The output may use another
/// floating dtype, while the multiply is always performed before the cast.
pub unsafe fn gated_into<R: Runtime>(
    input: &RudaTensor<R>,
    activation: Activation,
    out: &RudaTensor<R>,
) -> Result<(), InferenceError> {
    for t in [input, out] {
        check(t, input)?;
        float(t)?;
    }
    let mut shape = input.meta.shape().to_vec();
    let last = shape
        .last_mut()
        .ok_or(InferenceError("gated activation needs a final axis"))?;
    if *last == 0 || *last % 2 != 0 {
        return Err(InferenceError("gate/up width must be nonempty and equal"));
    }
    *last /= 2;
    let width = *last;
    if out.meta.shape()[..] != shape {
        return Err(InferenceError("preallocated gated output shape mismatch"));
    }
    let n = out.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(input.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, n, dim);
        gate_cast::launch::<R>(
            &input.client,
            count,
            dim,
            input.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            width as u32,
            activation.code(),
            input.dtype.into(),
            out.dtype.into(),
        );
    }
    Ok(())
}
pub struct PreparedGated<R: Runtime> {
    pack: super::PreparedPack<R>,
    pub output: RudaTensor<R>,
    activation: Activation,
}
impl<R: Runtime> PreparedGated<R> {
    pub fn prepare(
        input: &RudaTensor<R>,
        activation: Activation,
        dtype: ruda_core::tensor::DType,
    ) -> Result<Self, InferenceError> {
        let pack = super::PreparedPack::prepare(input)?;
        float(&pack.output)?;
        let mut shape = pack.output.meta.shape().to_vec();
        let last = shape
            .last_mut()
            .ok_or(InferenceError("gated activation requires an axis"))?;
        if *last == 0 || *last % 2 != 0 {
            return Err(InferenceError("gated feature width must be positive/even"));
        }
        *last /= 2;
        let output = alloc(&pack.output, Shape::from(shape), dtype);
        float(&output)?;
        Ok(Self {
            pack,
            output,
            activation,
        })
    }
    pub unsafe fn run(&self) -> Result<&RudaTensor<R>, InferenceError> {
        unsafe {
            self.pack.run();
            gated_into(&self.pack.output, self.activation, &self.output)?;
        }
        Ok(&self.output)
    }
}
#[ruda(launch)]
pub fn gate_cast<F: Float, O: Float>(
    x: &Array<F>,
    out: &mut Array<O>,
    n: u32,
    width: u32,
    #[comptime] activation: u32,
    #[define(F)] _f: StorageType,
    #[define(O)] _o: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / width as usize;
        let col = i % width as usize;
        let a = f32::cast_from(x[row * width as usize * 2 + col]);
        let b = f32::cast_from(x[row * width as usize * 2 + width as usize + col]);
        out[i] = O::cast_from(apply(a, activation) * b);
    }
}
pub fn silu_and_mul<R: Runtime>(x: &RudaTensor<R>) -> Result<RudaTensor<R>, InferenceError> {
    gated(x, Activation::Silu)
}
pub fn gelu_and_mul<R: Runtime>(x: &RudaTensor<R>) -> Result<RudaTensor<R>, InferenceError> {
    gated(x, Activation::Gelu)
}
pub fn gelu_tanh_and_mul<R: Runtime>(x: &RudaTensor<R>) -> Result<RudaTensor<R>, InferenceError> {
    gated(x, Activation::GeluTanh)
}

#[ruda]
pub(crate) fn apply(x: f32, #[comptime] kind: u32) -> f32 {
    let mut y = x;
    if comptime!(kind == 1) {
        y = x / (1.0 + (-x).exp());
    }
    if comptime!(kind == 2) {
        // A&S 7.1.26, absolute erf error < 1.5e-7. FP32 throughout.
        let z = x * core::f32::consts::FRAC_1_SQRT_2;
        let a = z.abs();
        let t = 1.0 / (1.0 + 0.3275911 * a);
        let polynomial =
            (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t;
        let mut erf = 1.0 - polynomial * (-a * a).exp();
        if z < 0.0 {
            erf = -erf;
        }
        y = 0.5 * x * (1.0 + erf);
    }
    if comptime!(kind == 3) {
        y = 0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh());
    }
    if comptime!(kind == 4) {
        y = f32::max(x, 0.0);
    }
    if comptime!(kind == 5) {
        let z = f32::max(x, 0.0);
        y = z * z;
    }
    y
}
#[ruda(launch)]
pub fn gate<F: Float>(
    input: &Array<F>,
    out: &mut Array<F>,
    n: u32,
    width: u32,
    #[comptime] kind: u32,
    #[define(F)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / width as usize;
        let col = i % width as usize;
        let x = f32::cast_from(input[row * width as usize * 2 + col]);
        let up = f32::cast_from(input[row * width as usize * 2 + width as usize + col]);
        out[i] = F::cast_from(apply(x, kind) * up);
    }
}
