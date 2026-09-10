// Separable Keys bicubic (a = -0.5), half-pixel centers and antialias support.
// The uint8 contract uses normalized f64 coefficients, signed fixed-point
// rounding and saturation after each axis, matching the reference CPU processor.

struct Filter {
    start: usize,
    weights: Vec<i64>,
}

fn cubic(distance: f64) -> f64 {
    let x = distance.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

fn filters(input: usize, output: usize) -> (Vec<Filter>, u32) {
    let scale = input as f64 / output as f64;
    let stretch = scale.max(1.0);
    let support = 2.0 * stretch;
    let mut largest = 0.0f64;
    let floating = (0..output).map(|i| {
        let center = scale * (i as f64 + 0.5);
        let start = (center - support + 0.5).max(0.0) as usize;
        let end = ((center + support + 0.5) as usize).min(input);
        let inverse = 1.0 / stretch;
        let mut weights = (start..end)
            .map(|j| cubic((j as f64 - center + 0.5) * inverse)).collect::<Vec<_>>();
        let sum: f64 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= sum;
            largest = largest.max(*weight);
        }
        (start, weights)
    }).collect::<Vec<_>>();
    let precision = (0..22).find(|&p| (0.5 + largest * ((1u32 << (p + 1)) as f64)) as i64 >= 32768).unwrap_or(22);
    let filters = floating.into_iter().map(|(start, weights)| Filter {
        start,
        weights: weights.into_iter().map(|w| (w * (1u32 << precision) as f64).round() as i64).collect(),
    }).collect();
    (filters, precision)
}

pub(super) fn bicubic_rgb(
    input: &[u8], height: usize, width: usize, out_height: usize, out_width: usize,
) -> Vec<u8> {
    let horizontal = if width == out_width {
        input.to_vec()
    } else {
        let (filters, precision) = filters(width, out_width);
        let mut pixels = vec![0; height * out_width * 3];
        for y in 0..height {
            for (x, filter) in filters.iter().enumerate() {
                for c in 0..3 {
                    let mut sum = 1i64 << (precision - 1);
                    for (k, &weight) in filter.weights.iter().enumerate() {
                        sum += weight * input[(y * width + filter.start + k) * 3 + c] as i64;
                    }
                    pixels[(y * out_width + x) * 3 + c] = (sum >> precision).clamp(0, 255) as u8;
                }
            }
        }
        pixels
    };
    if height == out_height { return horizontal; }
    let (filters, precision) = filters(height, out_height);
    let mut pixels = vec![0; out_height * out_width * 3];
    for (y, filter) in filters.iter().enumerate() {
        for x in 0..out_width {
            for c in 0..3 {
                let mut sum = 1i64 << (precision - 1);
                for (k, &weight) in filter.weights.iter().enumerate() {
                    sum += weight * horizontal[((filter.start + k) * out_width + x) * 3 + c] as i64;
                }
                pixels[(y * out_width + x) * 3 + c] = (sum >> precision).clamp(0, 255) as u8;
            }
        }
    }
    pixels
}
