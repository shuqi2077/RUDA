use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(crate) fn recurrent<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    beta: &Array<F>,
    log_decay: &Array<f32>,
    initial: &Array<f32>,
    output: &mut Array<F>,
    state: &mut Array<f32>,
    sequence: u32,
    key_dim: u32,
    value_dim: u32,
    query_scale: f32,
    #[define(F)] _dtype: StorageType,
) {
    let column = ABSOLUTE_POS;
    let kd = key_dim as usize;
    let vd = value_dim as usize;
    let seq = sequence as usize;
    if column >= state.len() / kd {
        terminate!();
    }
    let head = column / vd;
    let value_column = column % vd;
    let base = head * kd * vd + value_column;
    let mut key_column = 0usize;
    while key_column < kd {
        let index = base + key_column * vd;
        state[index] = initial[index];
        key_column += 1;
    }
    let mut token = 0usize;
    while token < seq {
        let row = head * seq + token;
        let decay = log_decay[row].exp();
        let mut prediction = 0.0f32;
        key_column = 0;
        while key_column < kd {
            let index = base + key_column * vd;
            let decayed = state[index] * decay;
            state[index] = decayed;
            prediction += decayed * f32::cast_from(k[row * kd + key_column]);
            key_column += 1;
        }
        let delta =
            (f32::cast_from(v[row * vd + value_column]) - prediction) * f32::cast_from(beta[row]);
        let mut result = 0.0f32;
        key_column = 0;
        while key_column < kd {
            let index = base + key_column * vd;
            let updated = state[index] + f32::cast_from(k[row * kd + key_column]) * delta;
            state[index] = updated;
            result += updated * (f32::cast_from(q[row * kd + key_column]) * query_scale);
            key_column += 1;
        }
        output[row * vd + value_column] = F::cast_from(result);
        token += 1;
    }
}
