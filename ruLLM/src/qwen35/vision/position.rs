pub(super) struct Positions {
    pub indices: Vec<i32>,
    pub weights: Vec<f32>,
    pub coordinates: Vec<f32>,
    pub segments: Vec<usize>,
}

pub(super) fn positions(grids: &[[usize; 3]], merge: usize, side: usize) -> Positions {
    let mut result = Positions {
        indices: vec![],
        weights: vec![],
        coordinates: vec![],
        segments: vec![],
    };
    for &[t, h, w] in grids {
        for _ in 0..t {
            result.segments.push(h * w);
            for by in 0..h / merge {
                for bx in 0..w / merge {
                    for iy in 0..merge {
                        for ix in 0..merge {
                            let (y, x) = (by * merge + iy, bx * merge + ix);
                            result.coordinates.extend([y as f32, x as f32]);
                            let fy = y as f32 * (side - 1) as f32 / (h - 1).max(1) as f32;
                            let fx = x as f32 * (side - 1) as f32 / (w - 1).max(1) as f32;
                            let (y0, x0) = (fy.floor() as usize, fx.floor() as usize);
                            let (dy, dx) = (fy - fy.floor(), fx - fx.floor());
                            for (row, wy) in [(y0, 1.0 - dy), ((y0 + 1).min(side - 1), dy)] {
                                for (col, wx) in [(x0, 1.0 - dx), ((x0 + 1).min(side - 1), dx)] {
                                    result.indices.push((row * side + col) as i32);
                                    result.weights.push(wy * wx);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_order_and_frame_boundaries() {
        let p = positions(&[[2, 2, 4], [1, 4, 2]], 2, 3);
        assert_eq!(p.segments, [8, 8, 8]);
        assert_eq!(
            &p.coordinates[..16],
            &[
                0., 0., 0., 1., 1., 0., 1., 1., 0., 2., 0., 3., 1., 2., 1., 3.
            ]
        );
        assert_eq!(&p.coordinates[..16], &p.coordinates[16..32]);
        assert!(
            p.weights
                .chunks_exact(4)
                .all(|w| (w.iter().sum::<f32>() - 1.0).abs() < 1e-6)
        );
        assert!(p.indices.iter().all(|&i| (0..9).contains(&i)));
    }

    #[test]
    fn aligned_corners_and_singleton_grid() {
        let p = positions(&[[1, 3, 3]], 1, 2);
        assert_eq!(&p.indices[16..20], &[0, 1, 2, 3]);
        assert_eq!(&p.weights[16..20], &[0.25; 4]);
        let p = positions(&[[1, 1, 1]], 1, 3);
        assert_eq!(p.indices, [0, 1, 3, 4]);
        assert_eq!(p.weights, [1., 0., 0., 0.]);
    }
}
