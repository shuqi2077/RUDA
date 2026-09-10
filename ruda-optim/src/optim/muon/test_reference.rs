// SPDX-License-Identifier: Apache-2.0
// Independent scalar FP64 test oracle. Not a runtime/device fallback.
#![allow(dead_code)]

#[derive(Clone, Copy)]
pub struct Settings {
    pub lr: f64,
    pub beta: f64,
    pub dampening: f64,
    pub nesterov: bool,
    pub ema: bool,
    pub stable: bool,
    pub rms: bool,
    pub input_output: bool,
    pub decay: f64,
    pub epsilon: f64,
    pub steps: usize,
}
impl Default for Settings {
    fn default() -> Self {
        Self { lr: 0.02, beta: 0.95, dampening: 0.0, nesterov: true,
            ema: false, stable: false, rms: false, input_output: false,
            decay: 0.01, epsilon: 1e-7, steps: 5 }
    }
}
fn transpose(x: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    for i in 0..rows { for j in 0..cols { out[j*rows+i] = x[i*cols+j]; } }
    out
}
fn multiply(x: &[f64], y: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows*cols];
    for i in 0..rows { for j in 0..cols { for k in 0..inner {
        out[i*cols+j] += x[i*inner+k] * y[k*cols+j];
    } } }
    out
}
pub fn orthogonalize(input: &[f64], rows: usize, cols: usize, steps: usize, epsilon: f64, stable: bool) -> Vec<f64> {
    assert!(rows > 0 && cols > 0 && rows.checked_mul(cols) == Some(input.len()));
    assert!(epsilon.is_finite() && epsilon > 0.0 && (1..100).contains(&steps));
    assert!(input.iter().all(|x| x.is_finite()));
    let transposed = rows > cols;
    let mut x = if transposed { transpose(input, rows, cols) } else { input.to_vec() };
    let (r, c) = if transposed { (cols, rows) } else { (rows, cols) };
    if stable {
        let scale = x.iter().map(|x| x.abs()).fold(f64::MIN_POSITIVE, f64::max);
        for value in &mut x { *value /= scale; }
        let denominator = x.iter().map(|v| v*v).sum::<f64>().sqrt().max(epsilon/scale);
        for value in &mut x { *value /= denominator; }
    } else {
        let denominator = x.iter().map(|v| v*v).sum::<f64>().sqrt().max(epsilon);
        for value in &mut x { *value /= denominator; }
    }
    for _ in 0..steps {
        let gram = multiply(&x, &transpose(&x, r, c), r, c, r);
        let square = multiply(&gram, &gram, r, r, r);
        let polynomial: Vec<_> = gram.iter().zip(square).map(|(g, q)| -4.775*g + 2.0315*q).collect();
        let correction = multiply(&polynomial, &x, r, r, c);
        x = x.iter().zip(correction).map(|(x, y)| 3.4445*x + y).collect();
    }
    if transposed { transpose(&x, r, c) } else { x }
}
pub fn step(w: &[f64], grad: &[f64], prior: Option<&[f64]>, rows: usize, cols: usize, s: &Settings) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(w.len(), grad.len());
    if let Some(p) = prior { assert_eq!(p.len(), grad.len()); }
    let buffer: Vec<f64> = grad.iter().enumerate().map(|(i, g)| {
        if s.ema { s.beta*prior.map_or(0.0, |m| m[i]) + (1.0-s.beta)*g }
        else { prior.map_or(*g, |m| s.beta*m[i] + (1.0-s.dampening)*g) }
    }).collect();
    let direction: Vec<f64> = grad.iter().zip(&buffer).map(|(g, m)| {
        if !s.nesterov { *m }
        else if s.ema { (1.0-s.beta)*g+s.beta*m }
        else { g+s.beta*m }
    }).collect();
    let update = orthogonalize(&direction, rows, cols, s.steps, s.epsilon, s.stable);
    let (outputs, inputs) = if s.input_output { (cols, rows) } else { (rows, cols) };
    let multiplier = if s.rms { 0.2*(rows.max(cols) as f64).sqrt() }
        else { (outputs as f64/inputs as f64).max(1.0).sqrt() };
    (w.iter().zip(update).map(|(w, u)| w*(1.0-s.lr*s.decay)-s.lr*multiplier*u).collect(), buffer)
}
