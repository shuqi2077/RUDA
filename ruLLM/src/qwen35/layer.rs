use super::*;

impl<R, F, I, BT> Layer<DeviceBackend<R, F, I, BT>>
where
    R: DeviceRuntime, R::Server: ComputeServer, R::Device: DeviceOps,
    F: FloatElement, I: IntElement, BT: BoolElement,
{
    pub(super) fn forward(
        &self, hidden: Tensor<DeviceBackend<R, F, I, BT>, 3>, config: &Qwen35TextConfig,
        cos: Tensor<DeviceBackend<R, F, I, BT>, 4>, sin: Tensor<DeviceBackend<R, F, I, BT>, 4>,
        state: &mut LayerCache<DeviceBackend<R, F, I, BT>>, position: usize,
    ) -> Result<Tensor<DeviceBackend<R, F, I, BT>, 3>, GenerationError> {
        let normalized = self.input_norm.forward(hidden.clone())?;
        let update = match (&self.mixer,state) {
            (Mixer::Full(attn),LayerCache::Full{key,value}) => {
                attn.forward(normalized,config,cos,sin,key,value,position)?
            }
            (Mixer::Delta(delta),LayerCache::Delta{convolution,state}) => {
                delta.forward(normalized,config,convolution,state)?
            }
            _ => return Err(GenerationError("Qwen3.5 cache layer type mismatch".into())),
        };
        let hidden = hidden+update;
        Ok(hidden.clone()+self.mlp.forward(self.post_norm.forward(hidden)?))
    }
}
