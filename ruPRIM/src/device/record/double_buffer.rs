use ruda_core::device::Device;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype};
use ruda_core::tensor::{DType, Shape};
use crate::collective::{RudaSum, RudaSumLaunch};
use crate::collective::record::RudaRecord;
use crate::collective::decompose::RudaDecomposer;
use crate::device::{RudaPrimitiveError, scan, scan_threads, segmented, segmented_sort};
use super::RudaRecordBuffer;

/// The two allocations must not overlap. Paired sorts additionally require
/// key and value allocations to be disjoint. Each pass updates the selector.
pub struct RudaRecordDoubleBuffer<R: Runtime, T: RudaRecord> {
    buffers: [RudaRecordBuffer<R, T>; 2],
    selector: usize,
}

impl<R: Runtime, T: RudaRecord> RudaRecordDoubleBuffer<R, T> {
    pub fn new(first: RudaRecordBuffer<R, T>, second: RudaRecordBuffer<R, T>, selector: usize)
        -> Result<Self, RudaPrimitiveError>
    {
        if selector > 1 { return Err(RudaPrimitiveError::Configuration("double-buffer selector must be zero or one")); }
        if first.len() != second.len() { return Err(RudaPrimitiveError::Length); }
        if first.bytes.device.to_id() != second.bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
        Ok(Self { buffers: [first, second], selector })
    }
    pub fn current(&self) -> &RudaRecordBuffer<R, T> { &self.buffers[self.selector] }
    pub fn alternate(&self) -> &RudaRecordBuffer<R, T> { &self.buffers[self.selector ^ 1] }
    pub fn selector(&self) -> usize { self.selector }
}

fn radix_impl<R, K, V, D>(keys: &mut RudaRecordDoubleBuffer<R, K>,
    mut values: Option<&mut RudaRecordDoubleBuffer<R, V>>, ranges: Option<(&RudaTensor<R>, &RudaTensor<R>)>,
    decomposer: D::RuntimeArg<R>, begin_bit: usize, end_bit: usize, descending: bool, threads: u32)
    -> Result<(), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    if begin_bit > end_bit || end_bit > D::BITS { return Err(RudaPrimitiveError::Configuration("invalid decomposed radix bit interval")); }
    scan_threads::<R, u64>(&keys.current().bytes, threads)?;
    let count = keys.current().len();
    if let Some(v) = values.as_ref() {
        if count != v.current().len() { return Err(RudaPrimitiveError::Length); }
        if keys.current().bytes.device.to_id() != v.current().bytes.device.to_id() { return Err(RudaPrimitiveError::Device); }
    }
    let map = match ranges {
        Some((begins, ends)) => Some(segmented_sort::make_map_len(&keys.current().bytes, count, begins, ends, threads)?),
        None => None,
    };
    if count < 2 || begin_bit == end_bit { return Ok(()); }
    let flags = empty_device_dtype(keys.current().bytes.client.clone(), keys.current().bytes.device.clone(), Shape::new([count]), DType::U64);
    let dim = RudaDim::new(keys.current().client().properties(), count);
    let paired = values.is_some();
    for bit in begin_bit..end_bit {
        let source = keys.current().clone();
        let output = keys.alternate().clone();
        let source_values = values.as_ref().map(|v| v.current().view()).unwrap_or_else(|| source.view());
        let output_values = values.as_ref().map(|v| v.alternate().view()).unwrap_or_else(|| output.view());
        let grid = calculate_ruda_count_elemwise(source.client(), count, dim);
        unsafe { super::digit_flags::launch_unchecked::<K, D, R>(source.client(), grid, dim, source.view(),
            flags.clone().into_linear_view(), decomposer.clone(), count, bit, descending); }
        let prefixes = match &map {
            Some(map) => segmented::scan_by_heads::<R, u64, RudaSum>(&flags, &map.heads, RudaSumLaunch::new(), None)?,
            None => scan::inclusive_scan::<R, u64, RudaSum>(&flags, RudaSumLaunch::new(), threads)?,
        };
        let grid = calculate_ruda_count_elemwise(source.client(), count, dim);
        unsafe {
            if let Some(map) = &map {
                super::segmented_sort::radix_pass::launch_unchecked::<K, V, R>(source.client(), grid, dim,
                    source.view(), source_values, map.ends.clone().into_linear_view(), flags.clone().into_linear_view(),
                    prefixes.into_linear_view(), output.view(), output_values, count, paired);
            } else {
                super::radix_scatter::launch_unchecked::<K, V, R>(source.client(), grid, dim,
                    source.view(), source_values, flags.clone().into_linear_view(), prefixes.into_linear_view(),
                    output.view(), output_values, count, paired);
            }
        }
        keys.selector ^= 1;
        if let Some(v) = values.as_deref_mut() { v.selector ^= 1; }
    }
    Ok(())
}

pub fn radix_keys<R, K, D>(keys: &mut RudaRecordDoubleBuffer<R, K>, decomposer: D::RuntimeArg<R>,
    begin_bit: usize, end_bit: usize, descending: bool, threads: u32) -> Result<(), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    radix_impl::<R, K, K, D>(keys, None, None, decomposer, begin_bit, end_bit, descending, threads)
}

pub fn radix_pairs<R, K, V, D>(keys: &mut RudaRecordDoubleBuffer<R, K>, values: &mut RudaRecordDoubleBuffer<R, V>,
    decomposer: D::RuntimeArg<R>, begin_bit: usize, end_bit: usize, descending: bool, threads: u32)
    -> Result<(), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    radix_impl::<R, K, V, D>(keys, Some(values), None, decomposer, begin_bit, end_bit, descending, threads)
}

pub fn segmented_radix_keys<R, K, D>(keys: &mut RudaRecordDoubleBuffer<R, K>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    decomposer: D::RuntimeArg<R>, begin_bit: usize, end_bit: usize, descending: bool, threads: u32) -> Result<(), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    radix_impl::<R, K, K, D>(keys, None, Some((begins, ends)), decomposer, begin_bit, end_bit, descending, threads)
}

pub fn segmented_radix_pairs<R, K, V, D>(keys: &mut RudaRecordDoubleBuffer<R, K>, values: &mut RudaRecordDoubleBuffer<R, V>,
    begins: &RudaTensor<R>, ends: &RudaTensor<R>, decomposer: D::RuntimeArg<R>,
    begin_bit: usize, end_bit: usize, descending: bool, threads: u32) -> Result<(), RudaPrimitiveError>
where R: Runtime, K: RudaRecord, V: RudaRecord, D: RudaDecomposer<K> + LaunchArg, D::RuntimeArg<R>: Clone,
{
    radix_impl::<R, K, V, D>(keys, Some(values), Some((begins, ends)), decomposer, begin_bit, end_bit, descending, threads)
}
