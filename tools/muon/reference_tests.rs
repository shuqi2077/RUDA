// SPDX-License-Identifier: Apache-2.0
// Runs with rustc alone; does NOT compile or execute RUDA tensors.
#[path = "../../ruda-optim/src/optim/muon/test_reference.rs"]
mod reference;
use reference::{Settings, orthogonalize, step};

#[test]
fn zero_update_only_decays() {
    let (out, momentum) = step(&[1.0; 6], &[0.0; 6], None, 2, 3, &Settings::default());
    assert_eq!(out, vec![0.9998; 6]);
    assert_eq!(momentum, vec![0.0; 6]);
}
#[test]
fn ema_first_state_is_not_sgd_state() {
    let (_, ema) = step(&[0.0; 6], &[0.25; 6], None, 2, 3, &Settings { ema: true, ..Default::default() });
    let (_, sgd) = step(&[0.0; 6], &[0.25; 6], None, 2, 3, &Settings::default());
    assert!(ema.iter().all(|x| (x-0.0125).abs() < 1e-14));
    assert_eq!(sgd, vec![0.25; 6]);
}
#[test]
fn transpose_equivariance() {
    let g = [1.0, 0.25, -0.5, 0.5, 0.75, -0.25];
    let gt = [1.0, 0.5, 0.25, 0.75, -0.5, -0.25];
    let a = orthogonalize(&g, 2, 3, 5, 1e-7, true);
    let b = orthogonalize(&gt, 3, 2, 5, 1e-7, true);
    for i in 0..2 { for j in 0..3 { assert!((a[i*3+j]-b[j*2+i]).abs() < 1e-12); } }
}
#[test]
fn stable_extremes() {
    for scale in [0.0, 1e-30, 1e30] {
        let g = [scale, -scale*0.5, scale*0.25, scale*0.75];
        assert!(orthogonalize(&g, 2, 2, 5, 1e-7, true).iter().all(|x| x.is_finite()));
    }
}
#[test]
fn continuation_with_saved_buffer() {
    let config = Settings { ema: true, ..Default::default() };
    let (a, m) = step(&[1.0; 6], &[0.25; 6], None, 2, 3, &config);
    let expected = step(&a, &[0.5; 6], Some(&m), 2, 3, &config);
    let restored = step(&a.clone(), &[0.5; 6], Some(&m.clone()), 2, 3, &config);
    assert_eq!(expected, restored);
}
