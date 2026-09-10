use ruda_core::device::Device;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::tensor::{RudaTensor, allocation::empty_device_dtype, element::TensorElement, layout::address_type};
use ruda_core::tensor::{DType, Shape};
use crate::collective::radix::RudaRadixKey;
use super::{RudaPrimitiveError, check_type, radix, scan_threads};
use crate::collective::{RudaSum, RudaSumLaunch};

/// Two nonoverlapping device allocations with a host-visible current selector.
/// Sort dispatches alternate between these allocations; no result readback is
/// required to determine which buffer contains the final sorted sequence.
pub struct RudaDoubleBuffer<R: Runtime> {
    buffers: [RudaTensor<R>; 2],
    selector: usize,
}

impl<R: Runtime> RudaDoubleBuffer<R> {
    pub fn new(first: RudaTensor<R>, second: RudaTensor<R>, selector: usize) -> Result<Self, RudaPrimitiveError> {
        if selector > 1 { return Err(RudaPrimitiveError::Configuration("double-buffer selector must be zero or one")); }
        if first.dtype != second.dtype { return Err(RudaPrimitiveError::Dtype); }
        if first.meta.num_elements() != second.meta.num_elements() { return Err(RudaPrimitiveError::Length); }
        if first.device.to_id() != second.device.to_id() { return Err(RudaPrimitiveError::Device); }
        Ok(Self { buffers: [first, second], selector })
    }
    pub fn current(&self) -> &RudaTensor<R> { &self.buffers[self.selector] }
    pub fn alternate(&self) -> &RudaTensor<R> { &self.buffers[self.selector ^ 1] }
    pub fn selector(&self) -> usize { self.selector }
}

pub fn radix_keys<R: Runtime, K: TensorElement + RudaRadixKey>(
    keys: &mut RudaDoubleBuffer<R>, begin_bit: u32, end_bit: u32, descending: bool, threads: u32,
) -> Result<(), RudaPrimitiveError> {
    check_type::<R, K>(keys.current())?;
    radix::check_bits::<K>(begin_bit, end_bit)?;
    scan_threads::<R, u64>(keys.current(), threads)?;
    let count = keys.current().meta.num_elements();
    if count < 2 || begin_bit == end_bit { return Ok(()); }
    let flags = empty_device_dtype(keys.current().client.clone(), keys.current().device.clone(), Shape::new([count]), DType::U64);
    let dim = RudaDim::new(keys.current().client.properties(), count);
    for bit in begin_bit..end_bit {
        let source = keys.current().clone();
        let output = keys.alternate().clone();
        let prefixes = radix::bit_prefixes::<R, K>(&source, &flags, bit, descending, threads)?;
        let grid = calculate_ruda_count_elemwise(&source.client, count, dim);
        unsafe {
            radix::scatter_keys::launch_unchecked::<K, R>(&source.client, grid, dim, address_type!(source, flags, prefixes, output),
                source.clone().into_linear_view(), flags.clone().into_linear_view(), prefixes.into_linear_view(), output.into_linear_view());
        }
        keys.selector ^= 1;
    }
    Ok(())
}

pub fn radix_pairs<R: Runtime, K: TensorElement + RudaRadixKey, V: TensorElement>(
    keys: &mut RudaDoubleBuffer<R>, values: &mut RudaDoubleBuffer<R>,
    begin_bit: u32, end_bit: u32, descending: bool, threads: u32,
) -> Result<(), RudaPrimitiveError> {
    check_type::<R, K>(keys.current())?;
    check_type::<R, V>(values.current())?;
    radix::check_bits::<K>(begin_bit, end_bit)?;
    scan_threads::<R, u64>(keys.current(), threads)?;
    let count = keys.current().meta.num_elements();
    if count != values.current().meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if keys.current().device.to_id() != values.current().device.to_id() { return Err(RudaPrimitiveError::Device); }
    if count < 2 || begin_bit == end_bit { return Ok(()); }
    let flags = empty_device_dtype(keys.current().client.clone(), keys.current().device.clone(), Shape::new([count]), DType::U64);
    let dim = RudaDim::new(keys.current().client.properties(), count);
    for bit in begin_bit..end_bit {
        let source_keys = keys.current().clone();
        let source_values = values.current().clone();
        let output_keys = keys.alternate().clone();
        let output_values = values.alternate().clone();
        let prefixes = radix::bit_prefixes::<R, K>(&source_keys, &flags, bit, descending, threads)?;
        let grid = calculate_ruda_count_elemwise(&source_keys.client, count, dim);
        unsafe {
            radix::scatter_pairs::launch_unchecked::<K, V, R>(&source_keys.client, grid, dim,
                address_type!(source_keys, source_values, flags, prefixes, output_keys, output_values),
                source_keys.clone().into_linear_view(), source_values.into_linear_view(), flags.clone().into_linear_view(),
                prefixes.into_linear_view(), output_keys.into_linear_view(), output_values.into_linear_view());
        }
        keys.selector ^= 1;
        values.selector ^= 1;
    }
    Ok(())
}

pub fn segmented_radix_keys<R: Runtime, K: TensorElement + RudaRadixKey>(
    keys: &mut RudaDoubleBuffer<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    begin_bit: u32, end_bit: u32, descending: bool, threads: u32,
) -> Result<(), RudaPrimitiveError> {
    check_type::<R, K>(keys.current())?;
    radix::check_bits::<K>(begin_bit, end_bit)?;
    let count = keys.current().meta.num_elements();
    let map = super::segmented_sort::make_map_len(keys.current(), count, begins, ends, threads)?;
    if count < 2 || begin_bit == end_bit { return Ok(()); }
    let flags = empty_device_dtype(keys.current().client.clone(), keys.current().device.clone(), Shape::new([count]), DType::U64);
    let dim = RudaDim::new(keys.current().client.properties(), count);
    for bit in begin_bit..end_bit {
        let source = keys.current().clone();
        let output = keys.alternate().clone();
        let grid = calculate_ruda_count_elemwise(&source.client, count, dim);
        unsafe {
            radix::flags_kernel::launch_unchecked::<K, R>(&source.client, grid, dim, address_type!(source, flags),
                source.clone().into_linear_view(), flags.clone().into_linear_view(), bit, descending);
        }
        let prefixes = super::segmented::scan_by_heads::<R, u64, RudaSum>(&flags, &map.heads, RudaSumLaunch::new(), None)?;
        let grid = calculate_ruda_count_elemwise(&source.client, count, dim);
        unsafe {
            super::segmented_sort::radix_scatter_keys::launch_unchecked::<K, R>(&source.client, grid, dim,
                address_type!(source, (map.begins), (map.ends), flags, prefixes, output),
                source.clone().into_linear_view(), map.begins.clone().into_linear_view(), map.ends.clone().into_linear_view(),
                flags.clone().into_linear_view(), prefixes.into_linear_view(), output.into_linear_view());
        }
        keys.selector ^= 1;
    }
    Ok(())
}

pub fn segmented_radix_pairs<R: Runtime, K: TensorElement + RudaRadixKey, V: TensorElement>(
    keys: &mut RudaDoubleBuffer<R>, values: &mut RudaDoubleBuffer<R>, begins: &RudaTensor<R>, ends: &RudaTensor<R>,
    begin_bit: u32, end_bit: u32, descending: bool, threads: u32,
) -> Result<(), RudaPrimitiveError> {
    check_type::<R, K>(keys.current())?;
    check_type::<R, V>(values.current())?;
    radix::check_bits::<K>(begin_bit, end_bit)?;
    let count = keys.current().meta.num_elements();
    if count != values.current().meta.num_elements() { return Err(RudaPrimitiveError::Length); }
    if keys.current().device.to_id() != values.current().device.to_id() { return Err(RudaPrimitiveError::Device); }
    let map = super::segmented_sort::make_map_len(keys.current(), count, begins, ends, threads)?;
    if count < 2 || begin_bit == end_bit { return Ok(()); }
    let flags = empty_device_dtype(keys.current().client.clone(), keys.current().device.clone(), Shape::new([count]), DType::U64);
    let dim = RudaDim::new(keys.current().client.properties(), count);
    for bit in begin_bit..end_bit {
        let source = keys.current().clone();
        let source_values = values.current().clone();
        let output = keys.alternate().clone();
        let output_values = values.alternate().clone();
        let grid = calculate_ruda_count_elemwise(&source.client, count, dim);
        unsafe {
            radix::flags_kernel::launch_unchecked::<K, R>(&source.client, grid, dim, address_type!(source, flags),
                source.clone().into_linear_view(), flags.clone().into_linear_view(), bit, descending);
        }
        let prefixes = super::segmented::scan_by_heads::<R, u64, RudaSum>(&flags, &map.heads, RudaSumLaunch::new(), None)?;
        let grid = calculate_ruda_count_elemwise(&source.client, count, dim);
        unsafe {
            super::segmented_sort::radix_scatter_pairs::launch_unchecked::<K, V, R>(&source.client, grid, dim,
                address_type!(source, source_values, (map.begins), (map.ends), flags, prefixes, output, output_values),
                source.clone().into_linear_view(), source_values.into_linear_view(), map.begins.clone().into_linear_view(), map.ends.clone().into_linear_view(),
                flags.clone().into_linear_view(), prefixes.into_linear_view(), output.into_linear_view(), output_values.into_linear_view());
        }
        keys.selector ^= 1;
        values.selector ^= 1;
    }
    Ok(())
}
