use crate::{Backend, TensorMetadata, get_device_settings,
    ops::{GridSampleOptions, GridSamplePaddingMode}, tensor::FloatTensor};
use ruda_core::tensor::{Shape, Slice};

fn coordinate<B: Backend>(value: FloatTensor<B>, size: usize, align: bool) -> FloatTensor<B> {
    let value = B::float_add_scalar(value, 1.0.into());
    let value = B::float_mul_scalar(value, ((if align { size - 1 } else { size }) as f64).into());
    let value = B::float_div_scalar(value, 2.0.into());
    if align { value } else { B::float_sub_scalar(value, 0.5.into()) }
}

fn rounded<B: Backend>(value: FloatTensor<B>) -> FloatTensor<B> {
    let settings = get_device_settings::<B>(&B::float_device(&value));
    let nan = B::float_is_nan(value.clone(), settings.bool_dtype);
    let value = B::float_mask_fill(value, nan, 0.0.into());
    let absolute = B::float_abs(value.clone());
    let floor = B::float_floor(absolute.clone());
    let up = B::float_greater_equal_elem(B::float_sub(absolute, floor.clone()), 0.5.into(), settings.bool_dtype);
    let magnitude = B::float_mask_where(floor.clone(), up, B::float_add_scalar(floor, 1.0.into()));
    let negative = B::float_lower_elem(value, 0.0.into(), settings.bool_dtype);
    B::float_mask_where(magnitude.clone(), negative, B::float_neg(magnitude))
}

pub(super) fn sample<B: Backend>(input: FloatTensor<B>, grid: FloatTensor<B>, options: GridSampleOptions) -> FloatTensor<B> {
    let [batch, channels, height, width] = input.shape().dims::<4>();
    let [grid_batch, out_h, out_w, coordinates] = grid.shape().dims::<4>();
    assert_eq!(grid_batch, batch);
    assert_eq!(coordinates, 2);
    assert!(height > 0 && width > 0);
    let settings = get_device_settings::<B>(&B::float_device(&input));
    let ranges = |channel| [Slice::from(0..batch), Slice::from(0..out_h), Slice::from(0..out_w), Slice::from(channel..channel + 1)];
    let mut x = coordinate::<B>(B::float_slice(grid.clone(), &ranges(0)), width, options.align_corners);
    let mut y = coordinate::<B>(B::float_slice(grid, &ranges(1)), height, options.align_corners);
    match options.padding_mode {
        GridSamplePaddingMode::Border => {
            let nonfinite = B::bool_or(
                B::bool_or(B::float_is_nan(x.clone(), settings.bool_dtype), B::float_is_inf(x.clone(), settings.bool_dtype)),
                B::bool_or(B::float_is_nan(y.clone(), settings.bool_dtype), B::float_is_inf(y.clone(), settings.bool_dtype)));
            x = B::float_clamp(B::float_mask_fill(x, nonfinite.clone(), ((width - 1) as f64 / 2.0).into()), 0.0.into(), ((width - 1) as f64).into());
            y = B::float_clamp(B::float_mask_fill(y, nonfinite, ((height - 1) as f64 / 2.0).into()), 0.0.into(), ((height - 1) as f64).into());
        }
        GridSamplePaddingMode::Reflection => {
            x = super::reflect_coordinates::<B>(x, width as f64, options.align_corners);
            y = super::reflect_coordinates::<B>(y, height as f64, options.align_corners);
        }
        GridSamplePaddingMode::Zeros => {}
    }
    let x = rounded::<B>(x);
    let y = rounded::<B>(y);
    let outside = if options.padding_mode == GridSamplePaddingMode::Zeros {
        Some(B::bool_or(
            B::bool_or(B::float_lower_elem(x.clone(), 0.0.into(), settings.bool_dtype), B::float_greater_equal_elem(x.clone(), (width as f64).into(), settings.bool_dtype)),
            B::bool_or(B::float_lower_elem(y.clone(), 0.0.into(), settings.bool_dtype), B::float_greater_equal_elem(y.clone(), (height as f64).into(), settings.bool_dtype))))
    } else { None };
    let iy = B::float_into_int(B::float_clamp(y, 0.0.into(), ((height - 1) as f64).into()), settings.int_dtype);
    let ix = B::float_into_int(B::float_clamp(x, 0.0.into(), ((width - 1) as f64).into()), settings.int_dtype);
    let indices = B::int_add(B::int_mul_scalar(iy, (width as i64).into()), ix);
    let indices = B::int_expand(B::int_reshape(indices, Shape::new([batch, 1, out_h * out_w])), Shape::new([batch, channels, out_h * out_w]));
    let output_shape = Shape::new([batch, channels, out_h, out_w]);
    let output = B::float_reshape(B::float_gather(2, B::float_reshape(input, Shape::new([batch, channels, height * width])), indices), output_shape.clone());
    match outside {
        Some(mask) => B::float_mask_fill(output, B::bool_expand(B::bool_reshape(mask, Shape::new([batch, 1, out_h, out_w])), output_shape), 0.0.into()),
        None => output,
    }
}
