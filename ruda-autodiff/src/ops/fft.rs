use super::{Backward, Ops, OpsKind, unary};
use crate::{checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients, tensor::AutodiffTensor};
use alloc::vec::Vec;
use ruda_tensor::{Backend, Shape, Slice, TensorMetadata};

fn ranges(shape: &Shape) -> Vec<Slice> {
    shape.iter().map(|&length| Slice::from(0..length)).collect()
}

fn resize<B: Backend>(value: B::FloatTensorPrimitive, dim: usize, length: usize) -> B::FloatTensorPrimitive {
    let shape = value.shape();
    if shape[dim] == length { return value; }
    let mut slices = ranges(&shape);
    if shape[dim] > length {
        slices[dim] = Slice::from(0..length);
        return B::float_slice(value, &slices);
    }
    let mut output_shape = shape;
    output_shape[dim] = length;
    let output = B::float_zeros(output_shape, &B::float_device(&value), value.dtype().into());
    B::float_slice_assign(output, &slices, value)
}

// Only conjugate-paired bins receive this factor; DC and an even-N Nyquist bin do not.
fn scale_interior<B: Backend>(value: B::FloatTensorPrimitive, dim: usize, n: usize, factor: f64) -> B::FloatTensorPrimitive {
    let end = (n + 1) / 2;
    if end <= 1 { return value; }
    let mut slices = ranges(&value.shape());
    slices[dim] = Slice::from(1..end);
    let interior = B::float_slice(value.clone(), &slices);
    let interior = B::float_mul_scalar(interior, factor.into());
    B::float_slice_assign(value, &slices, interior)
}

fn zero_bin<B: Backend>(value: B::FloatTensorPrimitive, dim: usize, bin: usize) -> B::FloatTensorPrimitive {
    let mut shape = value.shape();
    let mut slices = ranges(&shape);
    slices[dim] = Slice::from(bin..bin + 1);
    shape[dim] = 1;
    let zeros = B::float_zeros(shape, &B::float_device(&value), value.dtype().into());
    B::float_slice_assign(value, &slices, zeros)
}

fn real_only_bins<B: Backend>(value: B::FloatTensorPrimitive, dim: usize, n: usize) -> B::FloatTensorPrimitive {
    let value = zero_bin::<B>(value, dim, 0);
    if n % 2 == 0 { zero_bin::<B>(value, dim, n / 2) } else { value }
}

#[derive(Debug)]
struct RfftPart { imaginary: bool }

impl<B: Backend> Backward<B, 1> for RfftPart {
    type State = (usize, usize, Shape);

    fn backward(self, ops: Ops<Self::State, 1>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let (dim, n, input_shape) = ops.state;
        unary::<B, _>(ops.parents, ops.node, grads, |grad| {
            let grad = scale_interior::<B>(grad, dim, n, 0.5);
            let zeros = B::float_zeros(grad.shape(), &B::float_device(&grad), grad.dtype().into());
            let result = if self.imaginary {
                B::irfft(zeros, real_only_bins::<B>(grad, dim, n), dim, Some(n))
            } else {
                B::irfft(grad, zeros, dim, Some(n))
            };
            let result = B::float_mul_scalar(result, (n as f64).into());
            resize::<B>(result, dim, input_shape[dim])
        });
    }
}

pub(super) fn rfft<B: Backend, C: CheckpointStrategy>(
    signal: AutodiffTensor<B>, dim: usize, n: Option<usize>,
) -> (AutodiffTensor<B>, AutodiffTensor<B>) {
    let shape = signal.primitive.shape();
    let length = n.unwrap_or(shape[dim]);
    assert!(length.is_power_of_two(), "rfft length must be a positive power of two");
    let real = RfftPart { imaginary: false }.prepare::<C>([signal.node.clone()]).compute_bound().stateful();
    let imag = RfftPart { imaginary: true }.prepare::<C>([signal.node.clone()]).compute_bound().stateful();
    let (output_re, output_im) = B::rfft(signal.primitive, dim, n);
    let output_re = match real {
        OpsKind::Tracked(prep) => prep.finish((dim, length, shape.clone()), output_re),
        OpsKind::UnTracked(prep) => prep.finish(output_re),
    };
    let output_im = match imag {
        OpsKind::Tracked(prep) => prep.finish((dim, length, shape), output_im),
        OpsKind::UnTracked(prep) => prep.finish(output_im),
    };
    (output_re, output_im)
}

#[derive(Debug)]
struct Irfft;

impl<B: Backend> Backward<B, 2> for Irfft {
    type State = (usize, usize, Shape, Shape);

    fn backward(self, ops: Ops<Self::State, 2>, grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        let (dim, n, real_shape, imag_shape) = ops.state;
        let [real_parent, imag_parent] = ops.parents;
        let grad = grads.consume::<B>(&ops.node);
        let (real, imag) = B::rfft(grad, dim, Some(n));
        if let Some(parent) = real_parent {
            let real = B::float_div_scalar(scale_interior::<B>(real, dim, n, 2.0), (n as f64).into());
            grads.register::<B>(parent.id, resize::<B>(real, dim, real_shape[dim]));
        }
        if let Some(parent) = imag_parent {
            let imag = real_only_bins::<B>(imag, dim, n);
            let imag = B::float_div_scalar(scale_interior::<B>(imag, dim, n, 2.0), (n as f64).into());
            grads.register::<B>(parent.id, resize::<B>(imag, dim, imag_shape[dim]));
        }
    }
}

pub(super) fn irfft<B: Backend, C: CheckpointStrategy>(
    spectrum_re: AutodiffTensor<B>, spectrum_im: AutodiffTensor<B>, dim: usize, n: Option<usize>,
) -> AutodiffTensor<B> {
    let real_shape = spectrum_re.primitive.shape();
    let imag_shape = spectrum_im.primitive.shape();
    let length = n.unwrap_or_else(|| {
        real_shape[dim].checked_sub(1).and_then(|bins| bins.checked_mul(2))
            .expect("irfft spectrum length cannot infer a valid signal length")
    });
    assert!(length.is_power_of_two(), "irfft length must be a positive power of two");
    let prep = Irfft.prepare::<C>([spectrum_re.node.clone(), spectrum_im.node.clone()]).compute_bound().stateful();
    let output = B::irfft(spectrum_re.primitive, spectrum_im.primitive, dim, n);
    match prep {
        OpsKind::Tracked(prep) => prep.finish((dim, length, real_shape, imag_shape), output),
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}
