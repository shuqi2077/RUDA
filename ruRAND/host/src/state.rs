use ruda_core::rand::{SeedableRng, StdRng};
use ruda_core::stub::Mutex;

/// Type alias for the RNG used by Flex.
pub type HostRng = StdRng;

/// Global seed storage for reproducible random number generation.
/// Uses Mutex for thread-safe RNG state management.
pub(crate) static SEED: Mutex<Option<HostRng>> = Mutex::new(None);

/// Fallback RNG when `SEED` is empty (consumed or never set).
///
/// The seeding flow is: `Backend::seed()` stores a `FlexRng` in `SEED`. Random
/// ops (`float_random`, `int_random`) call `SEED.lock().take()`, consuming it for
/// that op and falling back to this function for subsequent calls. This function
/// delegates to ruda_core's own entropy source.
pub(crate) fn get_seeded_rng() -> HostRng {
    ruda_core::rand::get_seeded_rng()
}

pub fn seed(seed: u64) {
    let rng = HostRng::seed_from_u64(seed);
    let mut seed_lock = SEED.lock().unwrap();
    *seed_lock = Some(rng);
}
