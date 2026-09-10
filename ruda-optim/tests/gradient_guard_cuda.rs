// SPDX-License-Identifier: Apache-2.0
//! Hardware tests: explicitly opt in, no simulator/CPU fallback and no ignored tests.
#[path = "fused_adamw/common.rs"] mod common;
#[path = "../examples/gradient_guard/staged.rs"] mod staged;
use common::*;
use ruda_core::tensor::DType;
use ruda_kernel::tensor::permutation::swap_dims;
use ruda_optim::fused_adamw::{
    AdamWEntry, AdamWOptions, AdamWState, StepControl, SkipReason, adamw_step,
    gradient_stats_sync, guarded_adamw_step,
    gradient_norm::{gradient_stats_reference, GradientGuardOptions, GradientGuardError, NonFinitePolicy},
    reference::adamw_reference,
};

#[test]
fn dtype_tail_single_and_two_stage_norms_match_reference() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for n in [1, 31, 255, 256, 257, 1023, 1024, 1025, 65537, 1_048_577] {
            let values = rounded(&(0..n).map(|i| ((i % 37) as f32 - 18.0) * 0.125).collect::<Vec<_>>(), dtype);
            let g = tensor(&values, [n], dtype);
            let actual = gradient_stats_sync(&[&g], 8.0).unwrap();
            let expected = gradient_stats_reference(&[&values], 8.0).unwrap();
            assert!(actual.stats.all_finite());
            assert_eq!(actual.stats.elements(), n as u64);
            assert_eq!(actual.stats.tensors(), 1);
            assert!((actual.stats.total_norm() - expected.total_norm()).abs() <= 2e-5 * expected.total_norm().max(1e-30));
            assert_eq!(actual.readback_bytes, 12);
            assert_eq!(actual.reduction_launches, if n <= 1024 { 1 } else { 2 });
            assert!(actual.scratch_bytes <= 12 * 1024 + 12);
            close(&floats(&g), &values, 0.0);
        }
    }
}
#[test]
fn extreme_finite_gradients_have_finite_norm() {
    for values in [vec![1e30, -1e30], vec![1e-30, -1e-30], vec![3e38, 3e38]] {
        let g = tensor(&values, [values.len()], DType::F32);
        let s = gradient_stats_sync(&[&g], 1.0).unwrap().stats;
        let expected = gradient_stats_reference(&[&values], 1.0).unwrap();
        assert!(s.all_finite() && s.total_norm().is_finite());
        assert!((s.total_norm() / expected.total_norm() - 1.0).abs() < 1e-5);
    }
}
#[test]
fn empty_and_mixed_empty_summaries_do_not_launch_for_empty_tensor() {
    let e = tensor(&[], [0], DType::F32);
    let g = tensor(&[3.0, 4.0], [2], DType::BF16);
    let a = gradient_stats_sync(&[&e], 1.0).unwrap();
    assert_eq!((a.reduction_launches, a.readback_bytes), (0, 0));
    assert_eq!(a.stats.total_norm(), 0.0);
    let b = gradient_stats_sync(&[&e, &g, &e], 1.0).unwrap();
    assert_eq!((b.stats.elements(), b.stats.tensors()), (2, 3));
    assert!((b.stats.total_norm() - 5.0).abs() < 1e-6);
    assert_eq!(b.readback_bytes, 12);
}
#[test]
fn common_group_clip_agrees_with_staged_and_independent_oracle() {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for amsgrad in [false, true] {
            for maximize in [false, true] {
                let p1 = tensor(&[0.5, -0.75], [2], DType::F32);
                let p2 = tensor(&[1.0], [1], DType::F32);
                let g1 = tensor(&[384.0, 0.0], [2], dtype);
                let g2 = tensor(&[512.0], [1], dtype);
                let entries = [AdamWEntry { parameters: &p1, gradients: &g1, state: None }, AdamWEntry { parameters: &p2, gradients: &g2, state: None }];
                let o = AdamWOptions { amsgrad, maximize, ..Default::default() };
                let control = StepControl { gradient_scale: 128.0, skip_update: false };
                let a = guarded_adamw_step(&entries, &o, control, Default::default()).unwrap();
                let b = staged::step(&entries, &o, control, Default::default()).unwrap();
                assert!(a.skip_reason.is_none());
                assert!((a.statistics.as_ref().unwrap().stats.total_norm() - 5.0).abs() < 1e-6);
                assert!((a.clip_multiplier - 1.0 / (5.0 + 1e-6)).abs() < 1e-7);
                for (i, original) in [&p1, &p2].iter().enumerate() {
                    let raw = floats(entries[i].gradients);
                    let clipped: Vec<_> = raw.iter().map(|x| (*x / 128.0) * a.clip_multiplier).collect();
                    let expected = adamw_reference(&floats(original), &clipped, None, &o, StepControl::default()).unwrap();
                    close(&floats(&a.updates[i].parameters), &floats(&b.updates[i].parameters), 1e-5);
                    close(&floats(&a.updates[i].parameters), &expected.parameters, 1e-5);
                    check_state(a.updates[i].state.as_ref().unwrap(), expected.state.as_ref().unwrap());
                }
            }
        }
    }
}
#[test]
fn nonfinite_in_last_tensor_skips_entire_group_and_can_resume() {
    let o = AdamWOptions { amsgrad: true, ..Default::default() };
    let p = tensor(&[0.25], [1], DType::F32);
    let good = tensor(&[0.5], [1], DType::F32);
    let prior = adamw_step(&p, &good, None, &o, StepControl::default()).unwrap();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let g = tensor(&[bad], [1], DType::F32);
        let entries = [
            AdamWEntry { parameters: &prior.parameters, gradients: &good, state: prior.state.as_ref() },
            AdamWEntry { parameters: &p, gradients: &g, state: None },
        ];
        let skipped = guarded_adamw_step(&entries, &o, StepControl::default(), Default::default()).unwrap();
        assert_eq!(skipped.skip_reason, Some(SkipReason::NonFinite));
        assert!(skipped.updates.iter().all(|u| !u.updated));
        assert_eq!(skipped.updates[0].state.as_ref().unwrap().step(), 1);
        assert!(skipped.updates[1].state.is_none());
        close(&floats(&skipped.updates[0].parameters), &floats(&prior.parameters), 0.0);
        close(&floats(skipped.updates[0].state.as_ref().unwrap().first_moment()), &floats(prior.state.as_ref().unwrap().first_moment()), 0.0);
        assert!(matches!(guarded_adamw_step(&entries, &o, StepControl::default(), GradientGuardOptions { nonfinite: NonFinitePolicy::Error, ..Default::default() }), Err(GradientGuardError::NonFiniteGradients)));
    }
    let resumed = guarded_adamw_step(&[AdamWEntry { parameters: &prior.parameters, gradients: &good, state: prior.state.as_ref() }], &o, StepControl::default(), Default::default()).unwrap();
    assert_eq!(resumed.updates[0].state.as_ref().unwrap().step(), 2);
}
#[test]
fn finite_raw_but_overflowing_unscale_skips() {
    let p = tensor(&[1.0], [1], DType::F32);
    let g = tensor(&[f32::MAX], [1], DType::F32);
    let a = guarded_adamw_step(&[AdamWEntry { parameters: &p, gradients: &g, state: None }], &Default::default(), StepControl { gradient_scale: 0.5, skip_update: false }, Default::default()).unwrap();
    assert_eq!(a.skip_reason, Some(SkipReason::NonFinite));
}
#[test]
fn requested_skip_and_empty_group_avoid_statistics() {
    let p = tensor(&[1.0], [1], DType::F32);
    let g = tensor(&[f32::NAN], [1], DType::F32);
    let a = guarded_adamw_step(&[AdamWEntry { parameters: &p, gradients: &g, state: None }], &Default::default(), StepControl { skip_update: true, ..Default::default() }, Default::default()).unwrap();
    assert_eq!(a.skip_reason, Some(SkipReason::Requested)); assert!(a.statistics.is_none());
    let empty = tensor(&[], [0], DType::F32);
    let a = guarded_adamw_step(&[AdamWEntry { parameters: &empty, gradients: &empty, state: None }], &Default::default(), StepControl::default(), Default::default()).unwrap();
    assert_eq!(a.skip_reason, Some(SkipReason::Empty)); assert!(a.statistics.is_none());
}
#[test]
fn zero_clip_still_decays_parameters_and_advances_moments() {
    let p = tensor(&[2.0], [1], DType::F32);
    let g = tensor(&[3.0], [1], DType::F32);
    let o = AdamWOptions { learning_rate: 0.1, weight_decay: 0.2, ..Default::default() };
    let a = guarded_adamw_step(&[AdamWEntry { parameters: &p, gradients: &g, state: None }], &o, StepControl::default(), GradientGuardOptions { max_norm: Some(0.0), ..Default::default() }).unwrap();
    assert!(a.updates[0].updated);
    close(&floats(&a.updates[0].parameters), &[1.96], 1e-6);
    close(&floats(a.updates[0].state.as_ref().unwrap().first_moment()), &[0.0], 0.0);
}
#[test]
fn invalid_late_entry_and_noncontiguous_gradient_reject() {
    let p = tensor(&[1.0, 2.0, 3.0, 4.0], [2, 2], DType::F32);
    let g = p.clone(); let wrong = tensor(&[1.0], [1], DType::F32);
    assert!(guarded_adamw_step(&[
        AdamWEntry { parameters: &p, gradients: &g, state: None },
        AdamWEntry { parameters: &p, gradients: &wrong, state: None },
    ], &Default::default(), StepControl::default(), Default::default()).is_err());
    let transposed = swap_dims(g, 0, 1);
    assert!(gradient_stats_sync(&[&transposed], 1.0).is_err());
    let s = AdamWState::from_parts(u64::MAX, p.clone(), p.clone(), None).unwrap();
    assert!(guarded_adamw_step(&[AdamWEntry { parameters: &p, gradients: &p, state: Some(&s) }], &Default::default(), StepControl::default(), Default::default()).is_err());
    close(&floats(&p), &[1.0, 2.0, 3.0, 4.0], 0.0);
}
#[test]
fn no_clip_matches_original_optimizer() {
    let p = tensor(&[0.3, -0.1], [2], DType::F32);
    let g = tensor(&[128.0, -32.0], [2], DType::BF16);
    let control = StepControl { gradient_scale: 128.0, skip_update: false };
    let o = AdamWOptions::default();
    let old = adamw_step(&p, &g, None, &o, control).unwrap();
    let new = guarded_adamw_step(&[AdamWEntry { parameters: &p, gradients: &g, state: None }], &o, control, GradientGuardOptions { max_norm: None, ..Default::default() }).unwrap();
    close(&floats(&new.updates[0].parameters), &floats(&old.parameters), 0.0);
    close_moments(&floats(new.updates[0].state.as_ref().unwrap().second_moment()), &floats(old.state.as_ref().unwrap().second_moment()));
}

#[test]
fn multistep_clipped_amsgrad_matches_materialized_state() {
    let o = AdamWOptions { amsgrad: true, maximize: true, weight_decay: 0.03, ..Default::default() };
    let mut a = tensor(&(0..1033).map(|i| i as f32 * 0.001).collect::<Vec<_>>(), [1033], DType::F32);
    let mut b = a.clone();
    let g = tensor(&(0..1033).map(|i| ((i % 23) as f32 - 11.0)*128.0).collect::<Vec<_>>(), [1033], DType::BF16);
    let (mut sa, mut sb) = (None, None);
    let control = StepControl { gradient_scale: 128.0, skip_update: false };
    for _ in 0..5 {
        let x = guarded_adamw_step(&[AdamWEntry { parameters: &a, gradients: &g, state: sa.as_ref() }], &o, control, Default::default()).unwrap();
        let y = staged::step(&[AdamWEntry { parameters: &b, gradients: &g, state: sb.as_ref() }], &o, control, Default::default()).unwrap();
        let x = x.updates.into_iter().next().unwrap();
        let y = y.updates.into_iter().next().unwrap();
        close(&floats(&x.parameters), &floats(&y.parameters), 1e-5);
        let sx = x.state.as_ref().unwrap(); let sy = y.state.as_ref().unwrap();
        close_moments(&floats(sx.first_moment()), &floats(sy.first_moment()));
        close_moments(&floats(sx.second_moment()), &floats(sy.second_moment()));
        close_moments(&floats(sx.max_second_moment().unwrap()), &floats(sy.max_second_moment().unwrap()));
        assert_eq!(sx.step(), sy.step());
        a = x.parameters; sa = x.state; b = y.parameters; sb = y.state;
    }
}
