//! Boolean mask packing, including independently padded ragged segments.
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitOrder {
    Big,
    Little,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitpackError(pub &'static str);
impl core::fmt::Display for BitpackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for BitpackError {}

pub struct PackedSegments<R: Runtime> {
    pub data: RudaTensor<R>,
    pub indptr: RudaTensor<R>,
    pub host_indptr: Vec<u32>,
}
pub fn packbits<R: Runtime>(
    input: &RudaTensor<R>,
    order: BitOrder,
) -> Result<RudaTensor<R>, BitpackError> {
    if input.meta.num_dims() != 1 || input.meta.shape()[0] > u32::MAX as usize {
        return Err(BitpackError(
            "packbits expects a U32-indexed one-dimensional mask",
        ));
    }
    Ok(segment_packbits(input, &[0, input.meta.shape()[0] as u32], order)?.data)
}

pub fn segment_packbits<R: Runtime>(
    input: &RudaTensor<R>,
    indptr: &[u32],
    order: BitOrder,
) -> Result<PackedSegments<R>, BitpackError> {
    if input.meta.num_dims() != 1
        || !input.is_contiguous()
        || !matches!(input.dtype, DType::Bool(_) | DType::U8)
        || indptr.is_empty()
        || indptr[0] != 0
        || indptr.last().copied().unwrap() as usize != input.meta.shape()[0]
        || indptr.windows(2).any(|v| v[0] > v[1])
    {
        return Err(BitpackError(
            "segments must partition a contiguous Bool/U8 vector",
        ));
    }
    let segments = indptr.len() - 1;
    if segments > input.client.properties().hardware.max_ruda_count.0 as usize
        || segments > u32::MAX as usize / 3
    {
        return Err(BitpackError("segment count exceeds launch grid"));
    }
    let mut host_indptr = vec![0u32];
    let mut metadata = Vec::<u32>::with_capacity(segments * 3);
    for pair in indptr.windows(2) {
        let n = pair[1] - pair[0];
        let start = *host_indptr.last().unwrap();
        metadata.extend([pair[0], n, start]);
        host_indptr.push(
            start
                .checked_add(n.div_ceil(8))
                .ok_or(BitpackError("packed offsets overflow"))?,
        );
    }
    let data = empty_device_contiguous_dtype(
        input.client.clone(),
        input.device.clone(),
        Shape::from([*host_indptr.last().unwrap() as usize]),
        DType::U8,
    );
    let out_indptr = RudaTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        Shape::from([host_indptr.len()]),
        input
            .client
            .create_from_slice(bytemuck::cast_slice(&host_indptr)),
        DType::U32,
    );
    if segments != 0 {
        let metadata = RudaTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            Shape::from([segments, 3]),
            input
                .client
                .create_from_slice(bytemuck::cast_slice(&metadata)),
            DType::U32,
        );
        pack::launch::<R>(
            &input.client,
            RudaCount::Static(segments as u32, 1, 1),
            RudaDim::new_1d(128),
            input.clone().into_array_arg(),
            metadata.into_array_arg(),
            data.clone().into_array_arg(),
            order == BitOrder::Big,
            if input.dtype.size() == 4 {
                DType::U32.into()
            } else {
                DType::U8.into()
            },
        );
    }
    Ok(PackedSegments {
        data,
        indptr: out_indptr,
        host_indptr,
    })
}

#[ruda(launch)]
pub fn pack<I: Int>(
    input: &Array<I>,
    metadata: &Array<u32>,
    out: &mut Array<u8>,
    #[comptime] big: bool,
    #[define(I)] _input_dtype: StorageType,
) {
    let segment = RUDA_POS_X as usize;
    let start = metadata[segment * 3] as usize;
    let n = metadata[segment * 3 + 1] as usize;
    let output = metadata[segment * 3 + 2] as usize;
    let mut byte = UNIT_POS as usize;
    while byte < (n + 7) / 8 {
        let mut bits = 0u32;
        #[unroll]
        for i in 0usize..8 {
            if byte * 8 + i < n && input[start + byte * 8 + i] != I::cast_from(0u32) {
                let mut shift = i;
                if comptime!(big) {
                    shift = 7 - i;
                }
                bits |= 1u32 << shift as u32;
            }
        }
        out[output + byte] = u8::cast_from(bits);
        byte += RUDA_DIM as usize;
    }
}
