// SPDX-License-Identifier: Apache-2.0
//! Explicit hardware tests. Never silently skipped if the requested CUDA backend fails.
#[path = "fused_adamw/common.rs"]
mod common;
#[path = "../examples/fused_adamw/staged.rs"]
mod staged;
use common::*;
use ruda_core::tensor::{DType, data::TensorData};
use ruda_kernel::tensor::{permutation::swap_dims, transfer::from_data};
use ruda_driver_cuda::CudaDevice;
use ruda_optim::fused_adamw::{AdamWOptions, AdamWState, FusedAdamWError, StepControl, adamw_step, reference::adamw_reference};

#[test]
fn fused_adamw_dtype_tail_and_multistep_reference() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for n in [1, 31, 32, 33, 127, 256, 257, 4097] {
            for amsgrad in [false, true] {
                let initial: Vec<_> = (0..n).map(|i| ((i % 19) as f32 - 9.0) * 0.125).collect();
                let mut p = tensor(&initial, [n], DType::F32);
                let input_alias = p.clone();
                let mut s = None;
                let mut host_p = initial.clone();
                let mut host_s = None;
                let o = AdamWOptions { amsgrad, weight_decay: 0.1, ..Default::default() };
                for step in 0..5 {
                    let source: Vec<_> = (0..n).map(|i| ((i % 11) as f32 - 5.0 + step as f32 * 0.1) * 4.0).collect();
                    let values = rounded(&source, dtype);
                    let g = tensor(&values, [n], dtype);
                    let control = StepControl { gradient_scale: 16.0, skip_update: false };
                    let expected = adamw_reference(&host_p, &values, host_s.as_ref(), &o, control).unwrap();
                    let actual = adamw_step(&p, &g, s.as_ref(), &o, control).unwrap();
                    assert!(actual.updated);
                    close(&floats(&actual.parameters), &expected.parameters, 1e-5);
                    check_state(actual.state.as_ref().unwrap(), expected.state.as_ref().unwrap());
                    close(&floats(&g), &values, 0.0);
                    p = actual.parameters;
                    s = actual.state;
                    host_p = expected.parameters;
                    host_s = expected.state;
                }
                close(&floats(&input_alias), &initial, 0.0);
            }
        }
    }
}

#[test]
fn fused_and_explicit_staged_kernels_agree() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for amsgrad in [false, true] {
            for maximize in [false, true] {
                let n = 513;
                let mut a = tensor(&(0..n).map(|i| i as f32 * 0.001 - 0.25).collect::<Vec<_>>(), [n], DType::F32);
                let mut b = a.clone();
                let g = tensor(&(0..n).map(|i| (i % 7) as f32 * 0.02 - 0.06).collect::<Vec<_>>(), [n], dtype);
                let (mut sa, mut sb) = (None, None);
                let o = AdamWOptions { amsgrad, maximize, weight_decay: 0.05, ..Default::default() };
                for _ in 0..7 {
                    let x = adamw_step(&a, &g, sa.as_ref(), &o, StepControl::default()).unwrap();
                    let y = staged::staged_step(&b, &g, sb.as_ref(), &o);
                    close(&floats(&x.parameters), &floats(&y.parameters), 1e-5);
                    let sx = x.state.as_ref().unwrap(); let sy = y.state.as_ref().unwrap();
                    close_moments(&floats(sx.first_moment()), &floats(sy.first_moment()));
                    close_moments(&floats(sx.second_moment()), &floats(sy.second_moment()));
                    a = x.parameters; sa = x.state; b = y.parameters; sb = y.state;
                }
            }
        }
    }
}

#[test]
fn overflow_skip_and_restore_keep_state_and_inputs() {
    let o = AdamWOptions { amsgrad: true, ..Default::default() };
    let p = tensor(&[0.5, -0.7, 0.3], [3], DType::F32);
    let g = tensor(&[0.2, -0.1, 0.4], [3], DType::BF16);
    let first = adamw_step(&p, &g, None, &o, StepControl::default()).unwrap();
    let state = first.state.as_ref().unwrap();
    let invalid_g = tensor(&[f32::NAN, f32::INFINITY, -f32::INFINITY], [3], DType::F32);
    let skipped = adamw_step(&first.parameters, &invalid_g, Some(state), &o,
        StepControl { skip_update: true, ..Default::default() }).unwrap();
    assert!(!skipped.updated);
    assert_eq!(skipped.state.as_ref().unwrap().step(), 1);
    close(&floats(&skipped.parameters), &floats(&first.parameters), 0.0);
    close(&floats(skipped.state.as_ref().unwrap().first_moment()), &floats(state.first_moment()), 0.0);
    let (step, m, v, max) = state.clone().into_parts();
    // Device buffers read back/re-uploaded to simulate an explicit checkpoint restore.
    let restored = AdamWState::from_parts(step,
        tensor(&floats(&m), [3], DType::F32), tensor(&floats(&v), [3], DType::F32),
        max.map(|x| tensor(&floats(&x), [3], DType::F32))).unwrap();
    let a = adamw_step(&first.parameters, &g, Some(state), &o, StepControl::default()).unwrap();
    let b = adamw_step(&skipped.parameters, &g, Some(&restored), &o, StepControl::default()).unwrap();
    close(&floats(&a.parameters), &floats(&b.parameters), 0.0);
    assert_eq!(b.state.unwrap().step(), 2);
}

#[test]
fn empty_and_first_skip_do_not_create_state() {
    let o = AdamWOptions::default();
    let empty = tensor(&[], [0, 3], DType::F32);
    let r = adamw_step(&empty, &empty, None, &o, StepControl::default()).unwrap();
    assert!(!r.updated && r.state.is_none());
    assert_eq!(r.parameters.meta.shape()[..], [0, 3]);
    let p = tensor(&[1.0], [1], DType::F32);
    let g = tensor(&[f32::NAN], [1], DType::F32);
    let r = adamw_step(&p, &g, None, &o, StepControl { skip_update: true, ..Default::default() }).unwrap();
    assert!(!r.updated && r.state.is_none());
    close(&floats(&r.parameters), &[1.0], 0.0);
}

#[test]
fn invalid_metadata_dtype_state_and_options_are_rejected() {
    let o = AdamWOptions::default();
    let p = ruda_kernel::tensor::contiguous::into_contiguous(tensor(&[1.0, 2.0, 3.0, 4.0], [2, 2], DType::F32));
    assert!(p.is_contiguous());
    let g = p.clone();
    assert!(adamw_step(&p, &tensor(&[1.0, 2.0], [2], DType::F32), None, &o, StepControl::default()).is_err());
    let half_master = tensor(&[1.0; 4], [2, 2], DType::F16);
    assert!(adamw_step(&half_master, &g, None, &o, StepControl::default()).is_err());
    let int_gradient = from_data::<ruda_driver_cuda::CudaRuntime>(TensorData::new(vec![1i32; 4], [2, 2]), &CudaDevice::default());
    assert!(adamw_step(&p, &int_gradient, None, &o, StepControl::default()).is_err());
    let transposed = swap_dims(g.clone(), 0, 1);
    assert!(adamw_step(&p, &transposed, None, &o, StepControl::default()).is_err());
    let mut invalid_range = g.clone();
    invalid_range.handle.offset_end = Some(invalid_range.handle.size());
    assert!(adamw_step(&p, &invalid_range, None, &o, StepControl::default()).is_err());
    let mut invalid_stride = g.clone();
    invalid_stride.meta.strides = [1].into();
    assert!(adamw_step(&p, &invalid_stride, None, &o, StepControl::default()).is_err());
    let s = AdamWState::from_parts(u64::MAX, p.clone(), p.clone(), None).unwrap();
    assert!(matches!(adamw_step(&p, &g, Some(&s), &o, StepControl::default()), Err(FusedAdamWError::StepOverflow)));
    let s = AdamWState::from_parts(1, p.clone(), p.clone(), Some(p.clone())).unwrap();
    assert!(matches!(adamw_step(&p, &g, Some(&s), &o, StepControl::default()), Err(FusedAdamWError::InvalidState(_))));
    for epsilon in [0.0, -1.0, f32::NAN] {
        assert!(adamw_step(&p, &g, None, &AdamWOptions { epsilon, ..o }, StepControl::default()).is_err());
    }
    close(&floats(&p), &[1.0, 2.0, 3.0, 4.0], 0.0);
}

#[test]
fn aliased_read_only_inputs_are_not_modified() {
    let p = tensor(&[0.25, 0.5, -0.5], [3], DType::F32);
    let alias = p.clone();
    let o = AdamWOptions::default();
    let expected = adamw_reference(&[0.25, 0.5, -0.5], &[0.25, 0.5, -0.5], None, &o, StepControl::default()).unwrap();
    let r = adamw_step(&p, &p, None, &o, StepControl::default()).unwrap();
    close(&floats(&r.parameters), &expected.parameters, 1e-5);
    close(&floats(&alias), &[0.25, 0.5, -0.5], 0.0);
}

#[test]
fn reference_fixtures_from_real_pytorch_cpu_adamw() {
    let document: serde_json::Value = serde_json::from_str(include_str!("fused_adamw/pytorch_fixtures.json")).unwrap();
    let vector = |value: &serde_json::Value| value.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect::<Vec<_>>();
    for case in document["cases"].as_array().unwrap() {
        let dtype = match case["dtype"].as_str().unwrap() { "f32" => DType::F32, "f16" => DType::F16, "bf16" => DType::BF16, _ => panic!("fixture dtype") };
        let c = &case["options"];
        let o = AdamWOptions {
            learning_rate: c["learning_rate"].as_f64().unwrap() as f32,
            beta1: c["beta1"].as_f64().unwrap() as f32, beta2: c["beta2"].as_f64().unwrap() as f32,
            epsilon: c["epsilon"].as_f64().unwrap() as f32, weight_decay: c["weight_decay"].as_f64().unwrap() as f32,
            amsgrad: c["amsgrad"].as_bool().unwrap(), maximize: c["maximize"].as_bool().unwrap(),
        };
        let initial = vector(&case["initial"]);
        let n = initial.len();
        let mut p = tensor(&initial, [n], DType::F32);
        let mut s = None;
        for step in case["steps"].as_array().unwrap() {
            let g = tensor(&vector(&step["stored_gradient"]), [n], dtype);
            let r = adamw_step(&p, &g, s.as_ref(), &o, StepControl {
                gradient_scale: case["gradient_scale"].as_f64().unwrap() as f32, skip_update: false,
            }).unwrap();
            close(&floats(&r.parameters), &vector(&step["parameters"]), 1e-5);
            let state = r.state.as_ref().unwrap();
            close_moments(&floats(state.first_moment()), &vector(&step["first"]));
            close_moments(&floats(state.second_moment()), &vector(&step["second"]));
            if o.amsgrad { close_moments(&floats(state.max_second_moment().unwrap()), &vector(&step["maximum"])); }
            p = r.parameters; s = r.state;
        }
    }
}

#[test]
fn epsilon_placement_and_zero_betas_have_known_device_results() {
    let o = AdamWOptions { learning_rate: 0.1, beta1: 0.0, beta2: 0.0,
        epsilon: 0.5, weight_decay: 0.2, ..Default::default() };
    let p = tensor(&[2.0, -1.0, 0.0], [3], DType::F32);
    let g = tensor(&[1.0, -2.0, 0.0], [3], DType::F16);
    let r = adamw_step(&p, &g, None, &o, StepControl::default()).unwrap();
    close(&floats(&r.parameters), &[2.0 * 0.98 - 0.1 / 1.5, -0.98 + 0.2 / 2.5, 0.0], 2e-6);
    close_moments(&floats(r.state.as_ref().unwrap().first_moment()), &[1.0, -2.0, 0.0]);
    close_moments(&floats(r.state.as_ref().unwrap().second_moment()), &[1.0, 4.0, 0.0]);
}

#[test]
fn zero_lr_is_not_an_overflow_skip() {
    let o = AdamWOptions { learning_rate: 0.0, ..Default::default() };
    let p = tensor(&[1.0], [1], DType::F32);
    let g = tensor(&[3.0], [1], DType::BF16);
    let r = adamw_step(&p, &g, None, &o, StepControl::default()).unwrap();
    assert!(r.updated);
    close(&floats(&r.parameters), &[1.0], 0.0);
    let s = r.state.unwrap();
    assert_eq!(s.step(), 1);
    assert!(floats(s.first_moment())[0] > 0.0 && floats(s.second_moment())[0] > 0.0);
}
