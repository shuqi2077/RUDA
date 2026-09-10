//! CPU-only scheduling example. The integer state below is NOT a model or GPU.
use rullm::runtime::{BackendKind, BoundedWorker, DeviceKey, ReplicaPool, RuntimeError};

fn main() -> Result<(), RuntimeError> {
    let workers = (0..2).map(|ordinal| {
        BoundedWorker::spawn(DeviceKey { backend: BackendKind::Host, ordinal }, 4,
            move || Ok((ordinal, 0usize)))
    }).collect::<Result<Vec<_>,_>>()?;
    let mut pool = ReplicaPool::new(workers)?;
    let mut jobs = Vec::new();
    for request in 0..4 {
        let (device, handle) = pool.try_submit(move |(ordinal, count), cancellation| {
            if cancellation.is_cancelled() {
                return Err(RuntimeError::new(rullm::runtime::RuntimeErrorKind::Cancelled, "cancelled"));
            }
            *count += 1;
            Ok((request, *ordinal, *count))
        })?;
        jobs.push((device,handle));
    }
    for (device,handle) in jobs { println!("{device:?}: {:?}",handle.wait()?); }
    pool.shutdown()
}
