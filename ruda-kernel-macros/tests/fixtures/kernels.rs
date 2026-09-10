#[derive(RudaType, RudaLaunch, Clone)]
pub struct Pair {
    pub left: u32,
    pub right: u32,
}

#[derive(RudaType, Clone)]
pub enum Choice {
    Empty,
    Value(u32),
    Named { value: u32 },
}

#[derive(RudaType, RudaLaunch, Clone)]
#[ruda(runtime_variants)]
pub enum RuntimeChoice {
    Empty,
    Value(u32),
}

#[derive(AutotuneKey, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TuneKey {
    #[autotune(anchor)]
    pub length: usize,
}

#[ruda(launch)]
pub fn add<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i >= input.len() {
        terminate!();
    }
    output[i] = input[i] + F::new(1.0f32);
}
