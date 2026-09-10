use super::*;
use ruda_tensor::api::{activation::sigmoid, module::conv1d, ops::ConvOptions};
use rudnn::gated_delta::{GatedDeltaInput, chunk_gated_delta_rule, gated_delta_rule};

pub(super) struct Delta<B: Backend> {
    pub qkv: Linear<B>,
    pub z: Linear<B>,
    pub a: Linear<B>,
    pub b: Linear<B>,
    pub out: Linear<B>,
    pub conv: Tensor<B, 3>,
    pub dt_bias: Tensor<B, 1>,
    pub a_log: Tensor<B, 1>,
    pub norm: Tensor<B, 1>,
}

fn causal_convolution<B: Backend>(
    history: Tensor<B, 3>, weight: Tensor<B, 3>, sequence: usize,
) -> Tensor<B, 3> {
    let [batch, channels, _] = history.dims();
    let dtype = history.dtype();
    let accumulator = match dtype { DType::F16 | DType::BF16 => DType::F32, _ => dtype };
    // Round the FP32-accumulated convolution to the input dtype before SiLU.
    silu(conv1d(history.cast(accumulator), weight.cast(accumulator), None,
        ConvOptions::new([1], [0], [1], channels)).cast(dtype).cast(DType::F32))
        .cast(dtype)
        .slice([0..batch, 0..channels, 1..sequence + 1])
        .swap_dims(1, 2)
}

impl<R, F, I, BT> Delta<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime,
    R::Server: ComputeServer,
    R::Device: DeviceOps,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    pub(super) fn gate_projection(
        linear: &Linear<DeviceBackend<R, F, I, BT>>,
        input: Tensor<DeviceBackend<R, F, I, BT>, 3>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let dtype = input.dtype();
        let output = rublas::tensor_matmul::matmul(
            input.into_primitive().tensor(),
            linear.weight.val().unsqueeze::<3>().into_primitive().tensor(),
            None, rublas::tensor_matmul::MatmulStrategy::Naive, dtype,
        ).map_err(|error| GenerationError(format!("Qwen3.5 gate projection: {error:?}")))?;
        let output = Tensor::from_primitive(TensorPrimitive::Float(output));
        Ok(match &linear.bias { Some(bias) => output + bias.val().unsqueeze::<3>(), None => output })
    }

    pub(super) fn log_decay(
        a: Tensor<DeviceBackend<R, F, I, BT>, 3>,
        bias: Tensor<DeviceBackend<R, F, I, BT>, 1>,
        log_a: Tensor<DeviceBackend<R, F, I, BT>, 1>,
    ) -> Tensor<DeviceBackend<R, F, I, BT>, 3> {
        let a = a.cast(DType::F32) + bias.cast(DType::F32).unsqueeze::<3>();
        let softplus = a.clone().exp().log1p().mask_where(a.clone().greater_elem(20.0), a);
        -(log_a.cast(DType::F32).exp().unsqueeze::<3>() * softplus).swap_dims(1, 2)
    }

    pub(super) fn gated_normalize(
        y: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        z: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        norm: Tensor<DeviceBackend<R, F, I, BT>, 1>,
        dtype: DType,
        epsilon: f64,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 4>, GenerationError> {
        let gamma = Tensor::<DeviceBackend<R, F, I, BT>,1>::ones([y.dims()[3]],(&y.device(),DType::F32));
        let normalized = rudnn::normalization::rms_norm(
            y.into_primitive().tensor(), gamma.into_primitive().tensor(), epsilon as f32,
        ).map_err(|error| GenerationError(error.to_string()))?;
        let normalized = Tensor::<DeviceBackend<R, F, I, BT>,4>::from_primitive(TensorPrimitive::Float(normalized)).cast(dtype);
        let normalized = normalized.cast(norm.dtype()) * norm.unsqueeze::<4>();
        Ok((normalized.cast(DType::F32) * silu(z)).cast(dtype))
    }

    pub fn forward(
        &self,
        x: Tensor<DeviceBackend<R, F, I, BT>, 3>,
        c: &Qwen35TextConfig,
        conv_cache: &mut Option<Tensor<DeviceBackend<R, F, I, BT>, 3>>,
        state: &mut Option<Tensor<DeviceBackend<R, F, I, BT>, 4>>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        self.forward_observed(x, c, conv_cache, state, None)
    }

    pub(super) fn forward_observed(
        &self,
        x: Tensor<DeviceBackend<R, F, I, BT>, 3>,
        c: &Qwen35TextConfig,
        conv_cache: &mut Option<Tensor<DeviceBackend<R, F, I, BT>, 3>>,
        state: &mut Option<Tensor<DeviceBackend<R, F, I, BT>, 4>>,
        mut observe: Option<&mut dyn FnMut(&str, Tensor<DeviceBackend<R, F, I, BT>, 3>)>,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let [batch, seq, _] = x.dims();
        let (kh, vh, kd, vd) = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        );
        let (keys, values) = (kh * kd, vh * vd);
        let channels = keys * 2 + values;
        let width = c.linear_conv_kernel_dim;
        let dtype = x.dtype();
        let device = x.device();
        let mixed = self.qkv.forward(x.clone()).swap_dims(1, 2);
        let previous = conv_cache
            .as_ref()
            .map(Tensor::clone)
            .unwrap_or_else(|| Tensor::zeros([batch, channels, width], (&device, dtype)));
        let history = Tensor::cat(vec![previous, mixed], 2);
        *conv_cache = Some(
            history
                .clone()
                .slice([0..batch, 0..channels, seq..seq + width]),
        );
        let mixed = causal_convolution(history, self.conv.clone(), seq);
        if let Some(observe) = &mut observe { observe("conv0", mixed.clone()); }
        let query = mixed
            .clone()
            .slice([0..batch, 0..seq, 0..keys])
            .reshape([batch, seq, kh, kd])
            .swap_dims(1, 2);
        let key = mixed
            .clone()
            .slice([0..batch, 0..seq, keys..2 * keys])
            .reshape([batch, seq, kh, kd])
            .swap_dims(1, 2);
        let value = mixed
            .slice([0..batch, 0..seq, 2 * keys..channels])
            .reshape([batch, seq, vh, vd])
            .swap_dims(1, 2);
        let l2 = |x: Tensor<DeviceBackend<R, F, I, BT>, 4>| {
            let inv = (x.clone().square().sum_dim(3) + 1e-6)
                .cast(DType::F32)
                .rsqrt()
                .cast(dtype);
            x * inv
        };
        let query = attention::repeat_heads(l2(query), vh / kh);
        let key = attention::repeat_heads(l2(key), vh / kh);
        let beta = sigmoid(Self::gate_projection(&self.b, x.clone())?).swap_dims(1, 2);
        let decay = Self::log_decay(Self::gate_projection(&self.a, x.clone())?,
            self.dt_bias.clone(), self.a_log.clone());
        if let Some(observe) = &mut observe {
            observe("query0", query.clone().swap_dims(1,2).reshape([batch,seq,vh*kd]));
            observe("key0", key.clone().swap_dims(1,2).reshape([batch,seq,vh*kd]));
            observe("beta0", beta.clone().swap_dims(1,2));
            observe("decay0", decay.clone().swap_dims(1,2));
        }
        let initial = state
            .as_ref()
            .map(Tensor::clone)
            .unwrap_or_else(|| Tensor::zeros([batch, vh, kd, vd], (&device, DType::F32)));
        let recurrent_decode = state.is_some() && seq == 1;
        let input = GatedDeltaInput {
            query: query.into_primitive().tensor(),
            key: key.into_primitive().tensor(),
            value: value.into_primitive().tensor(),
            beta: beta.into_primitive().tensor(),
            log_decay: decay.into_primitive().tensor(),
            initial_state: initial.into_primitive().tensor(),
            query_scale: (kd as f32).sqrt().recip(),
        };
        let result = if recurrent_decode { gated_delta_rule(input) }
            else { chunk_gated_delta_rule(input, 64) }
        .map_err(|e| GenerationError(e.to_string()))?;
        *state = Some(Tensor::from_primitive(TensorPrimitive::Float(
            result.final_state,
        )));
        let y = Tensor::<DeviceBackend<R, F, I, BT>, 4>::from_primitive(TensorPrimitive::Float(
            result.output,
        ))
        .swap_dims(1, 2)
        .cast(DType::F32);
        if let Some(observe) = &mut observe { observe("core0", y.clone().reshape([batch,seq,values])); }
        let z = self
            .z
            .forward(x)
            .reshape([batch, seq, vh, vd])
            .cast(DType::F32);
        let gated = Self::gated_normalize(y, z, self.norm.clone(), dtype, c.rms_norm_eps)?
            .reshape([batch, seq, values]);
        if let Some(observe) = &mut observe { observe("gated0", gated.clone()); }
        Ok(self.out.forward(gated))
    }
}

#[cfg(all(test, feature = "nvidia"))]
mod tests {
    use super::*;
    use half::bf16;
    use ruda_tensor_device::cuda::{Cuda, CudaDevice};

    #[test]
    fn depthwise_convolution_preserves_low_precision_cancellation() {
        type B = Cuda<bf16,i32>;
        let device = CudaDevice::default();
        let history = Tensor::<B,3>::from_data(TensorData::new(
            vec![0.,0.,0.,0.,256.,1.,-256.,2., 0.,0.,0.,0.,-256.,-1.,256.,-2.], [1,2,8]), (&device,DType::BF16));
        let weight = Tensor::<B,3>::ones([2,1,4],(&device,DType::BF16));
        let actual = causal_convolution(history,weight,4).cast(DType::F32).into_data().to_vec::<f32>().unwrap();
        let raw = [256f32,-256.,256.,-256.,1.,-1.,3.,-3.];
        let expected = raw.map(|x| bf16::from_f32(x/(1.0+(-x).exp())).to_f32());
        assert_eq!(actual, expected);
    }
}
