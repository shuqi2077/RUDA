use super::*;
use crate::interpolate::cubic_weight;

pub(super) fn sample(input: HostTensor, grid: HostTensor, options: GridSampleOptions) -> HostTensor {
    let input = input.to_contiguous();
    let grid = grid.to_contiguous();
    assert_eq!(input.dtype(), grid.dtype());
    match input.dtype() {
        DType::F32 => typed::<f32>(input, grid, options),
        DType::F64 => typed::<f64>(input, grid, options),
        DType::F16 => typed::<f16>(input, grid, options),
        DType::BF16 => typed::<bf16>(input, grid, options),
        dtype => panic!("grid_sample_2d bicubic: unsupported dtype {dtype:?}"),
    }
}

fn coordinate(value: f64, size: usize, align: bool) -> f64 {
    let value = if value.is_nan() { -1.0 } else { value };
    if align { (value + 1.0) * (size - 1) as f64 / 2.0 }
    else { (value + 1.0) * size as f64 / 2.0 - 0.5 }
}

fn index(value: f64, size: usize, options: &GridSampleOptions) -> Option<usize> {
    let value = match options.padding_mode {
        GridSamplePaddingMode::Zeros => value,
        GridSamplePaddingMode::Border => value.clamp(0.0, (size - 1) as f64),
        GridSamplePaddingMode::Reflection => reflect_coordinate(value, size, options.align_corners)
            .clamp(0.0, (size - 1) as f64),
    };
    if value >= 0.0 && value < size as f64 { Some(value as usize) } else { None }
}

fn typed<T: Float + Element + bytemuck::Pod>(input: HostTensor, grid: HostTensor, options: GridSampleOptions) -> HostTensor {
    let [batch, channels, height, width] = input.layout().shape().dims::<4>();
    let [grid_batch, out_h, out_w, coordinates] = grid.layout().shape().dims::<4>();
    assert_eq!(grid_batch, batch);
    assert_eq!(coordinates, 2);
    assert!(height > 0 && width > 0);
    let values: &[T] = input.storage();
    let grid_values: &[T] = grid.storage();
    let positions = out_h * out_w;
    let mut output = vec![T::zero(); batch * channels * positions];
    for n in 0..batch {
        for p in 0..positions {
            let grid_index = (n * positions + p) * 2;
            let x = coordinate(ToElement::to_f64(&grid_values[grid_index]), width, options.align_corners);
            let y = coordinate(ToElement::to_f64(&grid_values[grid_index + 1]), height, options.align_corners);
            let x0 = x.floor();
            let y0 = y.floor();
            let wx = core::array::from_fn::<_, 4, _>(|tap| cubic_weight(x - x0 - (tap as f64 - 1.0), -0.75));
            let wy = core::array::from_fn::<_, 4, _>(|tap| cubic_weight(y - y0 - (tap as f64 - 1.0), -0.75));
            let ix = core::array::from_fn::<_, 4, _>(|tap| index(x0 + tap as f64 - 1.0, width, &options));
            let iy = core::array::from_fn::<_, 4, _>(|tap| index(y0 + tap as f64 - 1.0, height, &options));
            for c in 0..channels {
                let mut sum = 0.0;
                for ky in 0..4 {
                    let mut row = 0.0;
                    for kx in 0..4 {
                        let value = match (iy[ky], ix[kx]) {
                            (Some(yy), Some(xx)) => ToElement::to_f64(&values[((n * channels + c) * height + yy) * width + xx]),
                            _ => 0.0,
                        };
                        row += value * wx[kx];
                    }
                    sum += row * wy[ky];
                }
                let sum = if x.is_finite() && y.is_finite() { sum } else { f64::NAN };
                output[(n * channels + c) * positions + p] = <T as NumCast>::from(sum)
                    .expect("grid_sample_2d bicubic: output conversion failed");
            }
        }
    }
    HostTensor::new(Bytes::from_elems(output), Layout::contiguous(Shape::new([batch, channels, out_h, out_w])), input.dtype())
}
