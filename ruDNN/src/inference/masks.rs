//! Native adapters from flattened/segment-packed custom masks to fixed row
//! strides used by dense, ragged and paged attention. No mask data is read back.
use super::{InferenceError, PreparedPack, alloc, check, metadata::PreparedIndices};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
#[derive(Clone, Copy, Debug)]
pub enum MaskStorage {
    Bytes,
    PackedLittle,
    PackedBig,
}
pub struct PreparedMask<R: Runtime> {
    source: PreparedPack<R>,
    offsets: PreparedIndices<R>,
    lengths: PreparedIndices<R>,
    pub output: RudaTensor<R>,
    max_history: usize,
    storage: MaskStorage,
    packed_output: bool,
}
impl<R: Runtime> PreparedMask<R> {
    /// Offsets are bits for packed input, bytes for byte input. Lengths count
    /// logical boolean elements. The target packed representation is little-endian.
    pub fn prepare(
        source: &RudaTensor<R>,
        offsets: &RudaTensor<R>,
        lengths: &RudaTensor<R>,
        max_history: usize,
        storage: MaskStorage,
        packed_output: bool,
    ) -> Result<Self, InferenceError> {
        let source = PreparedPack::prepare(source)?;
        let offsets = PreparedIndices::prepare(offsets, DType::U64)?;
        let lengths = PreparedIndices::prepare(lengths, DType::I32)?;
        for t in [&offsets.output, &lengths.output] {
            check(t, &source.output)?;
        }
        if source.output.dtype != DType::U8
            || offsets.output.meta.num_dims() != 1
            || offsets.output.meta.shape() != lengths.output.meta.shape()
            || max_history > i32::MAX as usize
        {
            return Err(InferenceError(
                "custom mask byte/offset/length geometry mismatch",
            ));
        }
        let rows = offsets.output.meta.num_elements();
        let width = if packed_output {
            max_history.div_ceil(8)
        } else {
            max_history
        };
        rows.checked_mul(width)
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(InferenceError("row mask exceeds U32"))?;
        let output = alloc(&source.output, [rows, width], DType::U8);
        Ok(Self {
            source,
            offsets,
            lengths,
            output,
            max_history,
            storage,
            packed_output,
        })
    }
    pub unsafe fn run(&self) -> Result<&RudaTensor<R>, InferenceError> {
        unsafe {
            self.source.run();
            self.offsets.run();
            self.lengths.run();
        }
        let n = self.output.meta.num_elements();
        if n > 0 {
            let dim = RudaDim::new(self.output.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.output.client, n, dim);
            convert::launch::<R>(
                &self.output.client,
                count,
                dim,
                self.source.output.clone().into_array_arg(),
                self.offsets.output.clone().into_array_arg(),
                self.lengths.output.clone().into_array_arg(),
                self.output.clone().into_array_arg(),
                n as u32,
                self.output.meta.shape()[1] as u32,
                self.max_history as u32,
                self.source.output.meta.num_elements() as u64,
                match self.storage {
                    MaskStorage::Bytes => 0,
                    MaskStorage::PackedLittle => 1,
                    MaskStorage::PackedBig => 2,
                },
                self.packed_output,
            );
        }
        Ok(&self.output)
    }
}
/// Ragged sequence masks are individually byte-padded, but rows within a
/// sequence may start at any bit. Device CSR metadata is refreshed per replay.
pub struct PreparedRaggedMask<R: Runtime> {
    qcu: PreparedIndices<R>,
    kvcu: PreparedIndices<R>,
    starts: PreparedIndices<R>,
    offsets: RudaTensor<R>,
    lengths: RudaTensor<R>,
    mask: PreparedMask<R>,
    queries: usize,
    sequences: usize,
}
impl<R: Runtime> PreparedRaggedMask<R> {
    pub fn prepare(
        source: &RudaTensor<R>,
        qcu: &RudaTensor<R>,
        kvcu: &RudaTensor<R>,
        sequence_byte_offsets: &RudaTensor<R>,
        queries: usize,
        max_history: usize,
        big_endian: bool,
        packed_output: bool,
    ) -> Result<Self, InferenceError> {
        let qcu = PreparedIndices::prepare(qcu, DType::I32)?;
        let kvcu = PreparedIndices::prepare(kvcu, DType::I32)?;
        let starts = PreparedIndices::prepare(sequence_byte_offsets, DType::U64)?;
        if qcu.output.meta.num_dims() != 1
            || qcu.output.meta.num_elements() < 2
            || qcu.output.meta.shape() != kvcu.output.meta.shape()
            || starts.output.meta.num_dims() != 1
            || starts.output.meta.num_elements() < qcu.output.meta.num_elements() - 1
            || queries > i32::MAX as usize
        {
            return Err(InferenceError("ragged custom-mask CSR mismatch"));
        }
        let sequences = qcu.output.meta.num_elements() - 1;
        let offsets = alloc(source, [queries], DType::U64);
        let lengths = alloc(source, [queries], DType::I32);
        let mask = PreparedMask::prepare(
            source,
            &offsets,
            &lengths,
            max_history,
            if big_endian {
                MaskStorage::PackedBig
            } else {
                MaskStorage::PackedLittle
            },
            packed_output,
        )?;
        Ok(Self {
            qcu,
            kvcu,
            starts,
            offsets,
            lengths,
            mask,
            queries,
            sequences,
        })
    }
    pub unsafe fn run(&self) -> Result<&RudaTensor<R>, InferenceError> {
        unsafe {
            self.qcu.run();
            self.kvcu.run();
            self.starts.run();
        }
        if self.queries > 0 {
            let dim = RudaDim::new(self.offsets.client.properties(), self.queries);
            let count =
                kernel_dsl::calculate_ruda_count_elemwise(&self.offsets.client, self.queries, dim);
            ragged_rows::launch::<R>(
                &self.offsets.client,
                count,
                dim,
                self.qcu.output.clone().into_array_arg(),
                self.kvcu.output.clone().into_array_arg(),
                self.starts.output.clone().into_array_arg(),
                self.offsets.clone().into_array_arg(),
                self.lengths.clone().into_array_arg(),
                self.queries as u32,
                self.sequences as u32,
            );
        }
        unsafe { self.mask.run() }
    }
}
#[ruda(launch)]
pub fn ragged_rows(
    qcu: &Array<i32>,
    kvcu: &Array<i32>,
    starts: &Array<u64>,
    offsets: &mut Array<u64>,
    lengths: &mut Array<i32>,
    queries: u32,
    sequences: u32,
) {
    let row = ABSOLUTE_POS as usize;
    if row < queries as usize {
        let mut lo = 0usize;
        let mut hi = sequences as usize;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            if qcu[mid] <= row as i32 {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let mut offset = 0u64;
        let mut len = 0i32;
        if lo < sequences as usize {
            len = i32::max(0, kvcu[lo + 1] - kvcu[lo]);
            offset = starts[lo] * 8u64 + (row as i32 - qcu[lo]) as u64 * len as u64;
        }
        offsets[row] = offset;
        lengths[row] = len;
    }
}
#[ruda(launch)]
pub fn convert(
    source: &Array<u8>,
    offsets: &Array<u64>,
    lengths: &Array<i32>,
    out: &mut Array<u8>,
    n: u32,
    row_width: u32,
    max_history: u32,
    source_bytes: u64,
    #[comptime] storage: u32,
    #[comptime] packed: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / row_width as usize;
        let col = i % row_width as usize;
        let mut width = 1usize;
        if comptime!(packed) {
            width = 8usize;
        }
        let mut result = 0u32;
        for bit in 0..width {
            let token = col * width + bit;
            let mut live = false;
            if token < max_history as usize && token < i32::max(0, lengths[row]) as usize {
                let address = offsets[row] + token as u64;
                if comptime!(storage == 0) {
                    if address < source_bytes {
                        live = source[address as usize] != 0;
                    }
                } else {
                    if address / 8u64 < source_bytes {
                        let mut shift = (address % 8u64) as u32;
                        if comptime!(storage == 2) {
                            shift = 7u32 - shift;
                        }
                        live = (u32::cast_from(source[(address / 8u64) as usize])
                            & (1u32 << shift))
                            != 0;
                    }
                }
            }
            if live {
                result |= 1u32 << bit;
            }
        }
        out[i] = u8::cast_from(result);
    }
}
