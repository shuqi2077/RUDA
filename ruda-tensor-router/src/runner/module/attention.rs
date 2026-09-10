use super::*;
use ruda_tensor::graph::AttentionOpIr;

impl<B: BackendIr> Runner<B> {
    pub(super) fn apply_attention(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &AttentionOpIr,
    ) {
        let query = handles.get_float_tensor::<B>(&desc.query);
        let key = handles.get_float_tensor::<B>(&desc.key);
        let value = handles.get_float_tensor::<B>(&desc.value);
        let mask = desc.mask.as_ref().map(|m| handles.get_bool_tensor::<B>(m));
        let attn_bias = desc
            .attn_bias
            .as_ref()
            .map(|ab| handles.get_float_tensor::<B>(ab));

        let output = B::attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            desc.options.clone().into(),
        );

        handles.register_float_tensor::<B>(&desc.out.id, output);
    }
}
