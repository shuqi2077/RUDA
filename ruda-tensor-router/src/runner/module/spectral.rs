use super::*;
use ruda_tensor::graph::{IRfftOpIr, RfftOpIr};

impl<B: BackendIr> Runner<B> {
    pub(super) fn apply_rfft(&self, handles: &mut HandleContainer<B::Handle>, desc: &RfftOpIr) {
        let signal = handles.get_float_tensor::<B>(&desc.signal);
        let (out_re, out_im) = B::rfft(signal, desc.dim, desc.n);

        handles.register_float_tensor::<B>(&desc.out_re.id, out_re);
        handles.register_float_tensor::<B>(&desc.out_im.id, out_im);
    }

    pub(super) fn apply_irfft(&self, handles: &mut HandleContainer<B::Handle>, desc: &IRfftOpIr) {
        let spectrum_re = handles.get_float_tensor::<B>(&desc.input_re);
        let spectrum_im = handles.get_float_tensor::<B>(&desc.input_im);
        let signal = B::irfft(spectrum_re, spectrum_im, desc.dim, desc.n);

        handles.register_float_tensor::<B>(&desc.out_signal.id, signal);
    }
}
