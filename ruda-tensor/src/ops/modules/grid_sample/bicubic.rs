use crate::{Backend, DType, FloatDType, TensorMetadata, get_device_settings,
    ops::{GridSampleOptions, GridSamplePaddingMode}, tensor::FloatTensor};
use ruda_core::tensor::{Shape, Slice};

fn coordinate<B: Backend>(value: FloatTensor<B>, size: usize, align: bool) -> FloatTensor<B> {
    let settings = get_device_settings::<B>(&B::float_device(&value));
    let nan = B::float_is_nan(value.clone(), settings.bool_dtype);
    let value = B::float_add_scalar(B::float_mask_fill(value, nan, (-1.0).into()), 1.0.into());
    let value = B::float_div_scalar(B::float_mul_scalar(value, ((if align { size - 1 } else { size }) as f64).into()), 2.0.into());
    if align { value } else { B::float_sub_scalar(value, 0.5.into()) }
}

fn coefficient<B: Backend>(fraction: FloatTensor<B>, tap: usize) -> FloatTensor<B> {
    let x = match tap {
        0 => B::float_add_scalar(fraction, 1.0.into()),
        1 => fraction,
        2 => B::float_add_scalar(B::float_neg(fraction), 1.0.into()),
        _ => B::float_add_scalar(B::float_neg(fraction), 2.0.into()),
    };
    if tap == 1 || tap == 2 {
        let value = B::float_sub_scalar(B::float_mul_scalar(x.clone(), 1.25.into()), 2.25.into());
        B::float_add_scalar(B::float_mul(B::float_mul(value, x.clone()), x), 1.0.into())
    } else {
        let value = B::float_add_scalar(B::float_mul_scalar(x.clone(), (-0.75).into()), 3.75.into());
        let value = B::float_sub_scalar(B::float_mul(value, x.clone()), 6.0.into());
        B::float_add_scalar(B::float_mul(value, x), 3.0.into())
    }
}

fn bounded<B: Backend>(value: FloatTensor<B>, size: usize, options: &GridSampleOptions) -> FloatTensor<B> {
    match options.padding_mode {
        GridSamplePaddingMode::Zeros => value,
        GridSamplePaddingMode::Border => B::float_clamp(value, 0.0.into(), ((size - 1) as f64).into()),
        GridSamplePaddingMode::Reflection => B::float_clamp(super::reflect_coordinates::<B>(value, size as f64, options.align_corners), 0.0.into(), ((size - 1) as f64).into()),
    }
}

pub(super) fn sample<B: Backend>(input: FloatTensor<B>, grid: FloatTensor<B>, options: GridSampleOptions) -> FloatTensor<B> {
    assert_eq!(input.dtype(), grid.dtype());
    let dtype: FloatDType = input.dtype().into();
    let accumulator = if input.dtype() == DType::F64 { FloatDType::F64 } else { FloatDType::F32 };
    let device = B::float_device(&input);
    let settings = get_device_settings::<B>(&device);
    let input = B::float_cast(input, accumulator);
    let grid = B::float_cast(grid, accumulator);
    let [batch, channels, height, width] = input.shape().dims::<4>();
    let [grid_batch, out_h, out_w, coordinates] = grid.shape().dims::<4>();
    assert_eq!(batch, grid_batch);
    assert_eq!(coordinates, 2);
    assert!(height > 0 && width > 0);
    let ranges = |channel| [Slice::from(0..batch), Slice::from(0..out_h), Slice::from(0..out_w), Slice::from(channel..channel + 1)];
    let x = coordinate::<B>(B::float_slice(grid.clone(), &ranges(0)), width, options.align_corners);
    let y = coordinate::<B>(B::float_slice(grid, &ranges(1)), height, options.align_corners);
    let x0 = B::float_floor(x.clone());
    let y0 = B::float_floor(y.clone());
    let tx = B::float_sub(x, x0.clone());
    let ty = B::float_sub(y, y0.clone());
    let input = B::float_reshape(input, Shape::new([batch, channels, height * width]));
    let output_shape = Shape::new([batch, channels, out_h, out_w]);
    let weight_shape = Shape::new([batch, 1, out_h, out_w]);
    let mut output = B::float_zeros(output_shape.clone(), &device, accumulator);
    for ky in 0..4 {
        let yy = bounded::<B>(B::float_add_scalar(y0.clone(), (ky as f64 - 1.0).into()), height, &options);
        let wy = B::float_reshape(coefficient::<B>(ty.clone(), ky), weight_shape.clone());
        let mut row = B::float_zeros(output_shape.clone(), &device, accumulator);
        for kx in 0..4 {
            let xx = bounded::<B>(B::float_add_scalar(x0.clone(), (kx as f64 - 1.0).into()), width, &options);
            let invalid_y = B::bool_or(B::float_lower_elem(yy.clone(), 0.0.into(), settings.bool_dtype), B::float_greater_equal_elem(yy.clone(), (height as f64).into(), settings.bool_dtype));
            let invalid_x = B::bool_or(B::float_lower_elem(xx.clone(), 0.0.into(), settings.bool_dtype), B::float_greater_equal_elem(xx.clone(), (width as f64).into(), settings.bool_dtype));
            let nan_y = B::float_is_nan(yy.clone(), settings.bool_dtype);
            let nan_x = B::float_is_nan(xx.clone(), settings.bool_dtype);
            let invalid = B::bool_or(B::bool_or(invalid_y, invalid_x), B::bool_or(nan_y.clone(), nan_x.clone()));
            let iy = B::float_into_int(B::float_clamp(B::float_mask_fill(yy.clone(), nan_y, 0.0.into()), 0.0.into(), ((height - 1) as f64).into()), settings.int_dtype);
            let ix = B::float_into_int(B::float_clamp(B::float_mask_fill(xx, nan_x, 0.0.into()), 0.0.into(), ((width - 1) as f64).into()), settings.int_dtype);
            let indices = B::int_add(B::int_mul_scalar(iy, (width as i64).into()), ix);
            let indices = B::int_expand(B::int_reshape(indices, Shape::new([batch, 1, out_h * out_w])), Shape::new([batch, channels, out_h * out_w]));
            let values = B::float_reshape(B::float_gather(2, input.clone(), indices), output_shape.clone());
            let mask = B::bool_expand(B::bool_reshape(invalid, weight_shape.clone()), output_shape.clone());
            let values = B::float_mask_fill(values, mask, 0.0.into());
            let wx = B::float_reshape(coefficient::<B>(tx.clone(), kx), weight_shape.clone());
            row = B::float_add(row, B::float_mul(values, wx));
        }
        output = B::float_add(output, B::float_mul(row, wy));
    }
    B::float_cast(output, dtype)
}
