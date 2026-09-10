use ruda_core::{rand::get_seeded_rng, stub::Mutex};
use rand::{RngExt, SeedableRng, rngs::StdRng};

static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

pub fn seed(seed: u64) {
    let rng = StdRng::seed_from_u64(seed);
    let mut seed = SEED.lock().unwrap();
    *seed = Some(rng);
}

pub(crate) fn get_seeds() -> [u32; 4] {
    let mut seed = SEED.lock().unwrap();
    let mut rng: StdRng = match seed.take() {
        Some(rng_seeded) => rng_seeded,
        None => get_seeded_rng(),
    };
    let mut seeds: Vec<u32> = Vec::with_capacity(4);
    for _ in 0..4 {
        seeds.push(rng.random());
    }
    *seed = Some(rng);

    seeds.try_into().unwrap()
}
