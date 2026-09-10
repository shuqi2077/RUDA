use super::*;
use ruda_tensor::graph::{CtcLossBackwardOpIr, CtcLossOpIr};

impl<B: BackendIr> Runner<B> {
    pub(super) fn apply_ctc_loss(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &CtcLossOpIr,
    ) {
        let log_probs = handles.get_float_tensor::<B>(&desc.log_probs);
        let targets = handles.get_int_tensor::<B>(&desc.targets);
        let input_lengths = handles.get_int_tensor::<B>(&desc.input_lengths);
        let target_lengths = handles.get_int_tensor::<B>(&desc.target_lengths);

        let output = B::ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            desc.blank,
        );

        handles.register_float_tensor::<B>(&desc.out.id, output);
    }

    pub(super) fn apply_ctc_loss_backward(
        &self,
        handles: &mut HandleContainer<B::Handle>,
        desc: &CtcLossBackwardOpIr,
    ) {
        let log_probs = handles.get_float_tensor::<B>(&desc.log_probs);
        let targets = handles.get_int_tensor::<B>(&desc.targets);
        let input_lengths = handles.get_int_tensor::<B>(&desc.input_lengths);
        let target_lengths = handles.get_int_tensor::<B>(&desc.target_lengths);
        let grad_loss = handles.get_float_tensor::<B>(&desc.grad_loss);

        let output = B::ctc_loss_backward(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            grad_loss,
            desc.blank,
        );

        handles.register_float_tensor::<B>(&desc.out.id, output);
    }
}
