//! Counter-based device random filling. Seed/counter ownership stays explicit at the Python boundary.
use super::*;

#[ruda]
fn philox_word(counter_low: u32, counter_high: u32, seed_low: u32, seed_high: u32, lane: u32) -> u32 {
    let mut a = counter_low;
    let mut b = counter_high;
    let mut c = 0u32;
    let mut d = 0u32;
    let mut key_a = seed_low;
    let mut key_b = seed_high;
    for _round in 0..10 {
        let high_a = a.mul_hi(0xD2511F53u32);
        let high_c = c.mul_hi(0xCD9E8D57u32);
        let low_a = a * 0xD2511F53u32;
        let low_c = c * 0xCD9E8D57u32;
        a = high_c ^ b ^ key_a;
        b = low_c;
        c = high_a ^ d ^ key_b;
        d = low_a;
        key_a += 0x9E3779B9u32;
        key_b += 0xBB67AE85u32;
    }
    let mut result = a;
    if lane == 1 {result = b;}
    else if lane == 2 {result = c;}
    else if lane == 3 {result = d;}
    result
}

#[ruda(launch)]
fn fill<F: Float + RudaElement>(out: &mut Tensor<F>, seed_low: u32, seed_high: u32,
    base_low: u32, base_high: u32, first: f32, second: f32, #[comptime] distribution: u32) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let block = u32::cast_from(pos / 4);
        let low = base_low + block;
        let mut high = base_high;
        if low < base_low {high += 1;}
        let lane = u32::cast_from(pos % 4);
        let word = philox_word(low, high, seed_low, seed_high, lane);
        let unit = f32::cast_from(word >> 8) * (1.0f32 / 16777216.0f32);
        let mut value = first + (second-first)*unit;
        if comptime!(distribution == 1) {
            let pair = (lane / 2) * 2;
            let word_u = philox_word(low, high, seed_low, seed_high, pair);
            let word_v = philox_word(low, high, seed_low, seed_high, pair+1);
            let u = (f32::cast_from(word_u >> 9)+1.0f32)/8388609.0f32;
            let angle = f32::cast_from(word_v >> 8)*(6.283185307179586f32/16777216.0f32);
            let radius = (-2.0f32*u.ln()).sqrt();
            let mut sample = radius*angle.cos();
            if lane%2 == 1 {sample = radius*angle.sin();}
            value = first+second*sample;
        } else if comptime!(distribution == 2) {
            value = 0.0;
            if unit < first {value = 1.0;}
        }
        out[pos] = F::cast_from(value);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ruda_torch_random_api_version() -> u32 {1}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_random_fill(descriptor: *const Descriptor, seed: u64, counter: u64,
    distribution: u32, first: f32, second: f32) -> i32 {
    checked(|| {
        assert!(!descriptor.is_null() && distribution<=2 && first.is_finite() && second.is_finite());
        assert!(distribution!=0 || first<=second);
        assert!(distribution!=1 || second>=0.0);
        assert!(distribution!=2 || (0.0..=1.0).contains(&first));
        let out = unsafe {View::read(&*descriptor)};
        assert!(out.dtype<=2 && out.len<=u32::MAX as usize);
        let mut stride = 1;
        for d in (0..out.shape.len()).rev() {
            assert!(out.shape[d]<=1 || out.strides[d]==stride,"random fill requires contiguous output");
            stride *= out.shape[d];
        }
        if out.len==0 {return;}
        counter.checked_add(out.len.div_ceil(4) as u64).expect("random counter exhausted");
        let c = client();
        let grid = RudaCount::Static(u32::try_from(out.len.div_ceil(128)).unwrap(),1,1);
        macro_rules! launch {($f:ty)=>{unsafe {fill::launch::<$f,CudaRuntime>(&c,grid,RudaDim::new_1d(128),out.arg(),
            seed as u32,(seed>>32) as u32,counter as u32,(counter>>32) as u32,first,second,distribution)}};}
        match out.dtype {0=>launch!(f32),1=>launch!(f16),2=>launch!(bf16),_=>unreachable!()}
        LAUNCHES.fetch_add(1,Ordering::Relaxed);
        finish_dispatch(&c);
    })
}
