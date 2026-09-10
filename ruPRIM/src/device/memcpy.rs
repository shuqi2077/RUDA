use ruda_core::device::Device;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use ruda_kernel::tensor::{RudaTensor, layout::address_type};
use super::{RudaPrimitiveError, check_type};

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn batched_kernel(
    sources: &LinearView<u64>, destinations: &LinearView<u64>, sizes: &LinearView<u64>,
    #[comptime] threads: u64,
) {
    let batch = RUDA_POS as usize;
    if batch < sizes.shape() {
        let length = sizes[batch];
        if length > 0 {
            let source = sources[batch];
            let destination = destinations[batch];
            let mut byte = UNIT_POS as u64;
            while byte < length {
                native_store(destination + byte, native_load::<u8>(source + byte));
                if length - byte <= threads { break; }
                byte += threads;
            }
        }
    }
}

/// Copy arbitrary device buffers described by native U64 byte-address tables
/// and U64 byte counts. Input buffers may overlap; output buffers must not
/// overlap any input or other output. Zero-length buffers are not dereferenced.
///
/// # Safety
/// Every nonempty range must be valid on the active device, and its allocation
/// must remain alive until queued execution completes. The backend must expose
/// native addresses (direct PTX, CUDA C++, or HIP C++).
pub unsafe fn batched<R: Runtime>(
    sources: &RudaTensor<R>, destinations: &RudaTensor<R>, sizes: &RudaTensor<R>, threads: u32,
) -> Result<(), RudaPrimitiveError> {
    check_type::<R, u64>(sources)?;
    check_type::<R, u64>(destinations)?;
    check_type::<R, u64>(sizes)?;
    let batches = sizes.meta.num_elements();
    if sources.meta.num_elements() != batches || destinations.meta.num_elements() != batches { return Err(RudaPrimitiveError::Length); }
    if sources.device.to_id() != destinations.device.to_id() || sources.device.to_id() != sizes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    let hardware = &sources.client.properties().hardware;
    if threads == 0 || threads > hardware.max_units_per_ruda || threads > hardware.max_ruda_dim.0 {
        return Err(RudaPrimitiveError::Configuration("invalid memcpy block size"));
    }
    if batches == 0 { return Ok(()); }
    let work = batches.checked_mul(threads as usize).ok_or(RudaPrimitiveError::Configuration("memcpy launch size overflow"))?;
    let dim = RudaDim::new_1d(threads);
    let grid = calculate_ruda_count_elemwise(&sources.client, work, dim);
    unsafe {
        batched_kernel::launch_unchecked::<R>(&sources.client, grid, dim, address_type!(sources, destinations, sizes),
            sources.clone().into_linear_view(), destinations.clone().into_linear_view(), sizes.clone().into_linear_view(), threads as u64);
    }
    Ok(())
}
