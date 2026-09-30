//! Optional router-weight API 1. Uses public ruDNN kernels and original PyTorch
//! output allocations. No CPU fallback, dense score temporary or ABI replacement.
use super::*;
use rudnn::moe::{RouterScoring, RouterWeightOptions,
    selected_router_weights_into, selected_router_backward_into};

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_router_api_version() -> u32 { 1 }

/// Called only by the same-process C++ adapter. Descriptors remain valid during
/// submission; adapter supplies fresh disjoint output and validates every input.
/// op 0: logits, indices, weights. op 1: logits, indices, dweights, dlogits.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_router(op: u32, descriptors: *const Descriptor,
    count: usize, scoring: u32, renormalize: bool, scale: f32) -> i32
{
    checked(|| {
        assert!(!descriptors.is_null());
        assert_eq!(count, match op { 0 => 3, 1 => 4, _ => panic!("unknown router opcode") });
        let scoring = match scoring { 0 => RouterScoring::Softmax, 1 => RouterScoring::Sigmoid,
            _ => panic!("unknown router scoring policy") };
        let options = RouterWeightOptions { scoring, renormalize, scale };
        options.validate().expect("invalid router options");
        let descriptors = unsafe { std::slice::from_raw_parts(descriptors, count) };
        let views: Vec<View> = descriptors.iter().map(|d| unsafe { View::read(d) }).collect();
        assert!(views[0].dtype <= 2 && matches!(views[1].dtype, 4 | 5));
        let tensors: Vec<_> = views.iter().map(primitives::tensor).collect();
        if op == 0 {
            unsafe { selected_router_weights_into(&tensors[0], &tensors[1], &tensors[2], options) }
                .expect("RUDA router weight forward failed");
        } else {
            unsafe { selected_router_backward_into(&tensors[0], &tensors[1], &tensors[2], &tensors[3], options) }
                .expect("RUDA router weight backward failed");
        }
        if views[0].len != 0 {
            LAUNCHES.fetch_add(1, Ordering::Relaxed);
            finish_dispatch(&client());
        }
    })
}
