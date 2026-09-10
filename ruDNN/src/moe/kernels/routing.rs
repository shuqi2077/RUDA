use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(crate) fn softmax<F: Float>(
    logits: &Array<F>,
    probabilities: &mut Array<f32>,
    experts: u32,
    #[define(F)] _dtype: StorageType,
) {
    let token = ABSOLUTE_POS;
    let e = experts as usize;
    if token >= logits.len() / e {
        terminate!();
    }
    let start = token * e;
    let mut maximum = f32::cast_from(logits[start]);
    let mut i = 1usize;
    while i < e {
        maximum = f32::max(maximum, f32::cast_from(logits[start + i]));
        i += 1;
    }
    let mut sum = 0.0f32;
    i = 0;
    while i < e {
        let probability = f32::exp(f32::cast_from(logits[start + i]) - maximum);
        probabilities[start + i] = probability;
        sum += probability;
        i += 1;
    }
    i = 0;
    while i < e {
        probabilities[start + i] = probabilities[start + i] / sum;
        i += 1;
    }
}

#[ruda(launch)]
pub(crate) fn topk<F: Float>(
    probabilities: &Array<f32>,
    indices: &mut Array<u32>,
    weights: &mut Array<F>,
    experts: u32,
    top_k: u32,
    renormalize: u32,
    #[define(F)] _dtype: StorageType,
) {
    let token = ABSOLUTE_POS;
    let e = experts as usize;
    let k = top_k as usize;
    if token >= probabilities.len() / e {
        terminate!();
    }
    let mut previous_value = 0.0f32;
    let mut previous_id = 0usize;
    let mut selected_sum = 0.0f32;
    let mut slot = 0usize;
    while slot < k {
        let mut found = false;
        let mut best = 0.0f32;
        let mut best_id = 0usize;
        let mut expert = 0usize;
        while expert < e {
            let value = probabilities[token * e + expert];
            let nan = value != value;
            let eligible = slot == 0
                || value < previous_value
                || ((value == previous_value || nan) && expert > previous_id);
            if eligible && (!found || value > best) {
                found = true;
                best = value;
                best_id = expert;
            }
            expert += 1;
        }
        indices[token * k + slot] = best_id as u32;
        selected_sum += best;
        previous_value = best;
        previous_id = best_id;
        slot += 1;
    }
    slot = 0;
    while slot < k {
        let expert = indices[token * k + slot] as usize;
        let mut weight = probabilities[token * e + expert];
        if renormalize != 0 {
            weight = weight / selected_sum;
        }
        weights[token * k + slot] = F::cast_from(weight);
        slot += 1;
    }
}
