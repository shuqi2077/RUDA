#![allow(unsafe_code)]
//! Native block-floating FP4/FP8 storage, scale layouts and matrix products.
//! Raw byte payloads make checkpoint formats independent of the host/runtime's
//! native FP8 type support. Conversion and numerical work stay on the device.
use ruda_core::{
    device::Device,
    tensor::{DType, Shape},
};
use ruda_kernel::{
    dsl as kernel_dsl,
    dsl::prelude::*,
    tensor::{RudaTensor, allocation::empty_device_contiguous_dtype, contiguous::into_contiguous},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatFormat {
    E2m1,
    E4m3,
    E5m2,
}
impl FloatFormat {
    fn code(self) -> u32 {
        match self {
            Self::E2m1 => 0,
            Self::E4m3 => 1,
            Self::E5m2 => 2,
        }
    }
    pub fn packing(self) -> usize {
        if self == Self::E2m1 { 2 } else { 1 }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleFormat {
    E4m3,
    E8m0,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleLayout {
    Linear,
    Block128x4,
    Block8x4,
    PagedV4x4,
}
impl ScaleLayout {
    fn code(self) -> u32 {
        match self {
            Self::Linear => 0,
            Self::Block128x4 => 1,
            Self::Block8x4 => 2,
            Self::PagedV4x4 => 3,
        }
    }
    fn row_alignment(self) -> usize {
        match self {
            Self::Linear => 1,
            Self::Block128x4 => 128,
            Self::Block8x4 => 8,
            Self::PagedV4x4 => 4,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct QuantOptions {
    pub format: FloatFormat,
    pub scale_format: ScaleFormat,
    pub scale_layout: ScaleLayout,
    pub block_size: usize,
    pub alignment: usize,
    pub four_over_six: bool,
    pub error_absolute: bool,
    pub error_fast_f16: bool,
    pub scale_limit: u32,
}
impl QuantOptions {
    pub fn nvfp4() -> Self {
        Self {
            format: FloatFormat::E2m1,
            scale_format: ScaleFormat::E4m3,
            scale_layout: ScaleLayout::Block128x4,
            block_size: 16,
            alignment: 16,
            four_over_six: false,
            error_absolute: true,
            error_fast_f16: false,
            scale_limit: 448,
        }
    }
    pub fn mxfp4() -> Self {
        Self {
            scale_format: ScaleFormat::E8m0,
            block_size: 32,
            alignment: 32,
            ..Self::nvfp4()
        }
    }
    pub fn mxfp8() -> Self {
        Self {
            format: FloatFormat::E4m3,
            scale_format: ScaleFormat::E8m0,
            block_size: 32,
            alignment: 32,
            ..Self::nvfp4()
        }
    }
    pub fn with_four_over_six(mut self, config: FourOverSix) -> Result<Self, FloatQuantError> {
        if self.format != FloatFormat::E2m1
            || self.scale_format != ScaleFormat::E4m3
            || !matches!(config.scale_max, 256 | 448)
        {
            return Err(FloatQuantError(
                "4over6 requires NVFP4 with scale maximum 256 or 448",
            ));
        }
        self.four_over_six = true;
        self.error_absolute = config.error == QuantizationError::Mae;
        self.error_fast_f16 = config.fast_f16;
        self.scale_limit = config.scale_max;
        Ok(self)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantizationError {
    Mae,
    Mse,
}
#[derive(Clone, Copy, Debug)]
pub struct FourOverSix {
    pub scale_max: u32,
    pub error: QuantizationError,
    pub fast_f16: bool,
}
impl Default for FourOverSix {
    fn default() -> Self {
        Self {
            scale_max: 448,
            error: QuantizationError::Mae,
            fast_f16: false,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FloatQuantError(pub &'static str);
impl core::fmt::Display for FloatQuantError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for FloatQuantError {}
pub struct BlockFloatTensor<R: Runtime> {
    pub data: RudaTensor<R>,
    pub scales: RudaTensor<R>,
    /// Decoder scale: original = decoded_payload * decoded_block_scale * global.
    pub global: RudaTensor<R>,
    pub logical_shape: Shape,
    pub options: QuantOptions,
    groups: usize,
    rows: usize,
    width: usize,
    storage_width: usize,
    scale_rows: usize,
    scale_cols: usize,
    global_mode: u32,
}
pub struct GroupedMxfp8<R: Runtime> {
    pub native: BlockFloatTensor<R>,
    pub data_view: RudaTensor<R>,
    pub scale_view: RudaTensor<R>,
}
/// Grouped public layout: data [M,padded_K,B], scales
/// [32,4,padded_M/128,4,padded_K/128,B]. Views share the native byte storage;
/// no matrix transpose or device readback is needed.
pub fn mxfp8_grouped_quantize<R: Runtime>(
    input: &RudaTensor<R>,
    mask: &RudaTensor<R>,
) -> Result<GroupedMxfp8<R>, FloatQuantError> {
    if input.meta.num_dims() != 3 {
        return Err(FloatQuantError("grouped MXFP8 requires [B,M,K]"));
    }
    let native = quantize(
        input,
        None,
        Some(mask),
        QuantOptions {
            alignment: 128,
            ..QuantOptions::mxfp8()
        },
    )?;
    let mut data_view = native.data.clone();
    data_view.meta = Box::new(ruda_core::tensor::Metadata::new(
        Shape::from([native.rows, native.storage_width, native.groups]),
        [native.storage_width, 1, native.rows * native.storage_width],
    ));
    let mut scale_view = native.scales.clone();
    scale_view.meta = Box::new(ruda_core::tensor::Metadata::new(
        Shape::from([
            32,
            4,
            native.scale_rows / 128,
            4,
            native.storage_width / 128,
            native.groups,
        ]),
        [
            16,
            4,
            (native.scale_cols / 4) * 512,
            1,
            512,
            native.scale_rows * native.scale_cols,
        ],
    ));
    Ok(GroupedMxfp8 {
        native,
        data_view,
        scale_view,
    })
}
pub struct GroupedFp4<R: Runtime> {
    pub native: BlockFloatTensor<R>,
    pub data_view: RudaTensor<R>,
    pub scale_view: RudaTensor<R>,
}
pub fn scaled_fp4_grouped_quantize<R: Runtime>(
    input: &RudaTensor<R>,
    mask: &RudaTensor<R>,
    decoder_globals: &RudaTensor<R>,
    recipe: Option<FourOverSix>,
) -> Result<GroupedFp4<R>, FloatQuantError> {
    if input.meta.num_dims() != 3 || input.meta.shape()[2] % 16 != 0 {
        return Err(FloatQuantError(
            "grouped NVFP4 requires [B,M,K], K divisible by 16",
        ));
    }
    let options = if let Some(c) = recipe {
        QuantOptions::nvfp4().with_four_over_six(c)?
    } else {
        QuantOptions::nvfp4()
    };
    let native = quantize(input, Some(decoder_globals), Some(mask), options)?;
    let payload = native.storage_width / 2;
    let mut data_view = native.data.clone();
    data_view.meta = Box::new(ruda_core::tensor::Metadata::new(
        Shape::from([native.rows, payload, native.groups]),
        [payload, 1, native.rows * payload],
    ));
    let mut scale_view = native.scales.clone();
    scale_view.meta = Box::new(ruda_core::tensor::Metadata::new(
        Shape::from([
            32,
            4,
            native.scale_rows / 128,
            4,
            native.scale_cols / 4,
            native.groups,
        ]),
        [
            16,
            4,
            (native.scale_cols / 4) * 512,
            1,
            512,
            native.scale_rows * native.scale_cols,
        ],
    ));
    Ok(GroupedFp4 {
        native,
        data_view,
        scale_view,
    })
}
fn alloc<R: Runtime>(x: &RudaTensor<R>, shape: impl Into<Shape>, dtype: DType) -> RudaTensor<R> {
    empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), shape.into(), dtype)
}
fn check<R: Runtime>(x: &RudaTensor<R>, like: &RudaTensor<R>) -> Result<(), FloatQuantError> {
    if !x.is_contiguous()
        || x.qparams.is_some()
        || x.meta.num_elements() > u32::MAX as usize
        || x.device.to_id() != like.device.to_id()
        || !x.client.same_execution_queue(&like.client)
    {
        return Err(FloatQuantError(
            "block floats need same-queue contiguous U32-indexed tensors",
        ));
    }
    Ok(())
}
/// Input [...,M,K]. Optional I32 [groups] masks give the valid row count of
/// each group; masked rows and all physical padding are deterministically zero.
/// Global decoder scales may be scalar, per group, or per logical row.
pub fn quantize<R: Runtime>(
    input: &RudaTensor<R>,
    global: Option<&RudaTensor<R>>,
    row_mask: Option<&RudaTensor<R>>,
    options: QuantOptions,
) -> Result<BlockFloatTensor<R>, FloatQuantError> {
    quantize_impl(input, global, row_mask, options, true)
}
/// Allocate reusable payload/scale buffers without launching quantization.
pub fn allocate_quantized<R: Runtime>(
    input: &RudaTensor<R>,
    global: Option<&RudaTensor<R>>,
    options: QuantOptions,
) -> Result<BlockFloatTensor<R>, FloatQuantError> {
    quantize_impl(input, global, None, options, false)
}
fn quantize_impl<R: Runtime>(
    input: &RudaTensor<R>,
    global: Option<&RudaTensor<R>>,
    row_mask: Option<&RudaTensor<R>>,
    options: QuantOptions,
    launch: bool,
) -> Result<BlockFloatTensor<R>, FloatQuantError> {
    let input = into_contiguous(input.clone());
    check(&input, &input)?;
    let shape = input.meta.shape();
    let rank = shape.len();
    if rank < 2
        || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
        || options.block_size == 0
        || options.block_size % 2 != 0
        || !matches!(options.scale_limit, 256 | 448)
        || options.alignment == 0
        || options.alignment % options.block_size != 0
    {
        return Err(FloatQuantError("invalid block-floating input/recipe"));
    }
    let rows = shape[rank - 2];
    let width = shape[rank - 1];
    if rows == 0 || width == 0 || width % options.format.packing() != 0 {
        return Err(FloatQuantError(
            "block-floating matrix must have nonempty packable axes",
        ));
    }
    let groups = input.meta.num_elements() / rows / width;
    let storage_width = width.div_ceil(options.alignment) * options.alignment;
    let scale_rows =
        rows.div_ceil(options.scale_layout.row_alignment()) * options.scale_layout.row_alignment();
    let blocks = storage_width / options.block_size;
    let scale_cols = if options.scale_layout == ScaleLayout::Linear {
        blocks
    } else {
        blocks.div_ceil(4) * 4
    };
    let global = match global {
        Some(g) => {
            check(g, &input)?;
            if g.dtype != DType::F32 {
                return Err(FloatQuantError("global decoder scales must be F32"));
            }
            g.clone()
        }
        None => RudaTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            Shape::from([1]),
            input.client.create_from_slice(&1.0f32.to_ne_bytes()),
            DType::F32,
        ),
    };
    let global_mode = match global.meta.num_elements() {
        1 => 0,
        n if n == groups => 1,
        n if n == groups * rows => 2,
        _ => {
            return Err(FloatQuantError(
                "global scale count must be 1, groups, or groups*rows",
            ));
        }
    };
    if let Some(mask) = row_mask {
        check(mask, &input)?;
        if mask.dtype != DType::I32 || mask.meta.shape()[..] != [groups] {
            return Err(FloatQuantError("group mask must be I32 [groups]"));
        }
    }
    let size = groups
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(storage_width))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or(FloatQuantError("block-floating payload exceeds U32"))?;
    let sf_size = groups
        .checked_mul(scale_rows)
        .and_then(|n| n.checked_mul(scale_cols))
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or(FloatQuantError("block-floating scales exceed U32"))?;
    let data = alloc(
        &input,
        [groups, rows, storage_width / options.format.packing()],
        DType::U8,
    );
    let scales = alloc(&input, [sf_size], DType::U8);
    let hw = &input.client.properties().hardware;
    let lanes = hw.plane_size_max;
    if !matches!(lanes, 32 | 64)
        || groups * scale_rows > hw.max_ruda_count.0 as usize
        || scale_cols > hw.max_ruda_count.1 as usize
    {
        return Err(FloatQuantError(
            "block-floating recipe exceeds grid/plane contract",
        ));
    }
    if launch && size != 0 {
        quantize_blocks::launch::<R>(
            &input.client,
            RudaCount::Static((groups * scale_rows) as u32, scale_cols as u32, 1),
            RudaDim::new_1d(lanes),
            input.clone().into_array_arg(),
            global.clone().into_array_arg(),
            row_mask.unwrap_or(&global).clone().into_array_arg(),
            data.clone().into_array_arg(),
            scales.clone().into_array_arg(),
            rows as u32,
            width as u32,
            storage_width as u32,
            scale_rows as u32,
            scale_cols as u32,
            options.block_size as u32,
            global_mode,
            options.format.code(),
            options.scale_format == ScaleFormat::E8m0,
            options.scale_layout.code(),
            row_mask.is_some(),
            options.four_over_six,
            options.error_absolute,
            options.error_fast_f16,
            options.scale_limit,
            input.dtype.into(),
        );
    }
    Ok(BlockFloatTensor {
        data,
        scales,
        global,
        logical_shape: shape.clone(),
        options,
        groups,
        rows,
        width,
        storage_width,
        scale_rows,
        scale_cols,
        global_mode,
    })
}
/// Derive NVFP4 decoder scales on the device, either once for the tensor or
/// separately for every row. Zero tensors use unit scale; no host amax read.
pub fn nvfp4_global_scale<R: Runtime>(
    input: &RudaTensor<R>,
    per_token: bool,
    scale_max: u32,
) -> Result<RudaTensor<R>, FloatQuantError> {
    let x = into_contiguous(input.clone());
    check(&x, &x)?;
    if !matches!(x.dtype, DType::F32 | DType::F16 | DType::BF16)
        || x.meta.num_dims() < 2
        || !matches!(scale_max, 256 | 448)
    {
        return Err(FloatQuantError(
            "NVFP4 global scale requires float matrices and maximum 256/448",
        ));
    }
    let width = *x.meta.shape().last().unwrap();
    if width == 0 {
        return Err(FloatQuantError("NVFP4 empty width"));
    }
    let rows = if per_token {
        x.meta.num_elements() / width
    } else {
        1
    };
    let row_width = if per_token {
        width
    } else {
        x.meta.num_elements()
    };
    let out = alloc(&x, [rows], DType::F32);
    let lanes = x.client.properties().hardware.plane_size_max;
    if !matches!(lanes, 32 | 64) || rows > x.client.properties().hardware.max_ruda_count.0 as usize
    {
        return Err(FloatQuantError("NVFP4 global scale grid invalid"));
    }
    if rows > 0 {
        global_amax::launch::<R>(
            &x.client,
            RudaCount::Static(rows as u32, 1, 1),
            RudaDim::new_1d(lanes),
            x.clone().into_array_arg(),
            out.clone().into_array_arg(),
            row_width as u32,
            (scale_max * 6) as f32,
            x.dtype.into(),
        );
    }
    Ok(out)
}
pub fn quantize_nvfp4<R: Runtime>(
    input: &RudaTensor<R>,
    per_token: bool,
    recipe: Option<FourOverSix>,
) -> Result<BlockFloatTensor<R>, FloatQuantError> {
    let options = if let Some(c) = recipe {
        QuantOptions::nvfp4().with_four_over_six(c)?
    } else {
        QuantOptions::nvfp4()
    };
    let globals = nvfp4_global_scale(input, per_token, options.scale_limit)?;
    quantize(input, Some(&globals), None, options)
}
/// Explicit encoder-scale compatibility: FlashInfer's global multiplier is the
/// reciprocal of this library's decoder scale. Conversion remains on-device.
pub fn invert_global_scale<R: Runtime>(
    input: &RudaTensor<R>,
) -> Result<RudaTensor<R>, FloatQuantError> {
    check(input, input)?;
    if input.dtype != DType::F32 {
        return Err(FloatQuantError("global scale must be F32"));
    }
    let out = alloc(input, input.meta.shape().clone(), DType::F32);
    let n = input.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(input.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, n, dim);
        invert::launch::<R>(
            &input.client,
            count,
            dim,
            input.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
        );
    }
    Ok(out)
}
#[ruda(launch)]
pub fn global_amax<F: Float>(
    x: &Array<F>,
    out: &mut Array<f32>,
    width: u32,
    divisor: f32,
    #[define(F)] _dtype: StorageType,
) {
    let row = RUDA_POS_X as usize;
    let lane = UNIT_POS as usize;
    let mut j = lane;
    let mut maximum: f32 = 0.0f32;
    while j < width as usize {
        maximum = f32::max(maximum, f32::cast_from(x[row * width as usize + j]).abs());
        j += PLANE_DIM as usize;
    }
    maximum = plane_max(maximum);
    if lane == 0 {
        let mut value = maximum / divisor;
        if value == 0.0 {
            value = 1.0;
        }
        out[row] = value;
    }
}
#[ruda(launch)]
pub fn invert(x: &Array<f32>, out: &mut Array<f32>, n: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        out[i] = 1.0 / x[i];
    }
}
/// Graph-prepared quantized row append. Scratch scales with the update size,
/// not the persistent cache size. Only selected payload/SF rows are mutated.
pub struct PreparedRowAppend<'a, R: Runtime> {
    target: &'a BlockFloatTensor<R>,
    source: RudaTensor<R>,
    packed: RudaTensor<R>,
    copy: bool,
    destinations: RudaTensor<R>,
    globals: RudaTensor<R>,
    rows: BlockFloatTensor<R>,
}
impl<'a, R: Runtime> PreparedRowAppend<'a, R> {
    pub fn prepare(
        target: &'a BlockFloatTensor<R>,
        input: &RudaTensor<R>,
        destinations: &RudaTensor<R>,
    ) -> Result<Self, FloatQuantError> {
        if input.qparams.is_some()
            || input.meta.num_elements() > u32::MAX as usize
            || input.meta.num_dims() != 2
            || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
            || input.meta.shape()[1] != target.width
            || input.meta.shape()[0] == 0
        {
            return Err(FloatQuantError(
                "prepared quantized append requires nonempty [updates,K]",
            ));
        }
        let copy = !input.is_contiguous();
        let packed = if copy {
            alloc(input, input.meta.shape().clone(), input.dtype)
        } else {
            input.clone()
        };
        for t in [
            &packed,
            destinations,
            &target.data,
            &target.scales,
            &target.global,
        ] {
            check(t, &target.data)?;
        }
        let q = input.meta.shape()[0];
        if destinations.dtype != DType::I32 || destinations.meta.shape()[..] != [q] {
            return Err(FloatQuantError(
                "quantized append destinations require I32 [updates]",
            ));
        }
        let globals = alloc(&packed, [q], DType::F32);
        let rows = allocate_quantized(
            &packed,
            Some(&globals),
            QuantOptions {
                scale_layout: ScaleLayout::Linear,
                ..target.options
            },
        )?;
        Ok(Self {
            target,
            source: input.clone(),
            packed,
            copy,
            destinations: destinations.clone(),
            globals,
            rows,
        })
    }
    /// Destination row IDs must be unique and target storage exclusive. Global
    /// decoder scale values are read afresh, and unselected cache bytes remain
    /// unchanged. All retained owners must outlive device execution.
    pub unsafe fn run(&self) -> Result<(), FloatQuantError> {
        let x = &self.packed;
        if self.copy {
            ruda_kernel::library::tensor::copy_into(
                &x.client,
                self.source.clone().binding(),
                x.clone().binding(),
                x.dtype.into(),
            );
        }
        let q = x.meta.shape()[0];
        let target = self.target;
        let dim = RudaDim::new(x.client.properties(), q);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, q, dim);
        row_globals::launch::<R>(
            &x.client,
            count,
            dim,
            target.global.clone().into_array_arg(),
            self.destinations.clone().into_array_arg(),
            self.globals.clone().into_array_arg(),
            q as u32,
            target.rows as u32,
            (target.groups * target.rows) as u32,
            target.global_mode,
        );
        unsafe {
            self.rows.quantize_into(x, None)?;
        }
        let payload = target.storage_width / target.options.format.packing();
        let blocks = target.storage_width / target.options.block_size;
        let n = q
            .checked_mul(payload.max(blocks))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(FloatQuantError("prepared row append exceeds U32"))?;
        let dim = RudaDim::new(x.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x.client, n, dim);
        append_quantized::launch::<R>(
            &x.client,
            count,
            dim,
            self.rows.data.clone().into_array_arg(),
            self.rows.scales.clone().into_array_arg(),
            self.destinations.clone().into_array_arg(),
            target.data.clone().into_array_arg(),
            target.scales.clone().into_array_arg(),
            q as u32,
            target.rows as u32,
            (target.groups * target.rows) as u32,
            payload as u32,
            blocks as u32,
            target.scale_rows as u32,
            target.scale_cols as u32,
            target.options.scale_layout.code(),
        );
        Ok(())
    }
}
impl<R: Runtime> BlockFloatTensor<R> {
    /// expanded row -> permuted row. Payload, block scales and decoder globals
    /// move together. IDs must be a permutation of live rows; -1 skips a row.
    pub fn permute_rows(
        &self,
        expanded_to_permuted: &RudaTensor<R>,
    ) -> Result<Self, FloatQuantError> {
        check(expanded_to_permuted, &self.data)?;
        if self.groups != 1
            || expanded_to_permuted.dtype != DType::I32
            || expanded_to_permuted.meta.shape()[..] != [self.rows]
        {
            return Err(FloatQuantError(
                "row permutation needs one matrix and I32 [M] mapping",
            ));
        }
        let copy = |t: &RudaTensor<R>| {
            let out = alloc(t, t.meta.shape().clone(), t.dtype);
            ruda_kernel::library::tensor::copy_into(
                &t.client,
                t.clone().binding(),
                out.clone().binding(),
                t.dtype.into(),
            );
            out
        };
        let data = copy(&self.data);
        let scales = copy(&self.scales);
        let global = if self.global_mode == 2 {
            copy(&self.global)
        } else {
            self.global.clone()
        };
        let payload = self.storage_width / self.options.format.packing();
        let n = self
            .rows
            .checked_mul(payload.max(self.scale_cols))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(FloatQuantError("row permutation exceeds U32"))?;
        if n > 0 {
            let dim = RudaDim::new(self.data.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.data.client, n, dim);
            permute_quantized_rows::launch::<R>(
                &self.data.client,
                count,
                dim,
                self.data.clone().into_array_arg(),
                self.scales.clone().into_array_arg(),
                self.global.clone().into_array_arg(),
                expanded_to_permuted.clone().into_array_arg(),
                data.clone().into_array_arg(),
                scales.clone().into_array_arg(),
                global.clone().into_array_arg(),
                n as u32,
                self.rows as u32,
                payload as u32,
                self.scale_rows as u32,
                self.scale_cols as u32,
                self.global_mode == 2,
                self.options.scale_layout.code(),
            );
        }
        Ok(Self {
            data,
            scales,
            global,
            logical_shape: self.logical_shape.clone(),
            options: self.options,
            groups: self.groups,
            rows: self.rows,
            width: self.width,
            storage_width: self.storage_width,
            scale_rows: self.scale_rows,
            scale_cols: self.scale_cols,
            global_mode: self.global_mode,
        })
    }
    /// Fixed-address quantization into this reusable storage. Callers retain
    /// exclusive output ownership and must not alias input and payload/scales.
    pub unsafe fn quantize_into(
        &self,
        input: &RudaTensor<R>,
        row_mask: Option<&RudaTensor<R>>,
    ) -> Result<(), FloatQuantError> {
        check(input, &self.data)?;
        check(&self.global, &self.data)?;
        if input.meta.shape() != &self.logical_shape
            || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return Err(FloatQuantError(
                "quantization input no longer matches prepared shape",
            ));
        }
        if let Some(mask) = row_mask {
            check(mask, input)?;
            if mask.dtype != DType::I32 || mask.meta.shape()[..] != [self.groups] {
                return Err(FloatQuantError("quantization row mask needs I32 [groups]"));
            }
        }
        let hw = &input.client.properties().hardware;
        let lanes = hw.plane_size_max;
        if !matches!(lanes, 32 | 64)
            || self.groups * self.scale_rows > hw.max_ruda_count.0 as usize
            || self.scale_cols > hw.max_ruda_count.1 as usize
        {
            return Err(FloatQuantError("prepared quantization grid invalid"));
        }
        let o = self.options;
        if self.groups * self.rows != 0 {
            quantize_blocks::launch::<R>(
                &input.client,
                RudaCount::Static(
                    (self.groups * self.scale_rows) as u32,
                    self.scale_cols as u32,
                    1,
                ),
                RudaDim::new_1d(lanes),
                input.clone().into_array_arg(),
                self.global.clone().into_array_arg(),
                row_mask.unwrap_or(&self.global).clone().into_array_arg(),
                self.data.clone().into_array_arg(),
                self.scales.clone().into_array_arg(),
                self.rows as u32,
                self.width as u32,
                self.storage_width as u32,
                self.scale_rows as u32,
                self.scale_cols as u32,
                o.block_size as u32,
                self.global_mode,
                o.format.code(),
                o.scale_format == ScaleFormat::E8m0,
                o.scale_layout.code(),
                row_mask.is_some(),
                o.four_over_six,
                o.error_absolute,
                o.error_fast_f16,
                o.scale_limit,
                input.dtype.into(),
            );
        }
        Ok(())
    }
    pub unsafe fn dequantize_into(&self, out: &RudaTensor<R>) -> Result<(), FloatQuantError> {
        check(out, &self.data)?;
        if out.meta.shape() != &self.logical_shape
            || !matches!(out.dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return Err(FloatQuantError("prepared dequantization output mismatch"));
        }
        let n = self.groups * self.rows * self.width;
        let o = self.options;
        if n > 0 {
            let dim = RudaDim::new(out.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&out.client, n, dim);
            dequantize_values::launch::<R>(
                &out.client,
                count,
                dim,
                self.data.clone().into_array_arg(),
                self.scales.clone().into_array_arg(),
                self.global.clone().into_array_arg(),
                out.clone().into_array_arg(),
                n as u32,
                self.rows as u32,
                self.width as u32,
                self.storage_width as u32,
                self.scale_rows as u32,
                self.scale_cols as u32,
                o.block_size as u32,
                self.global_mode,
                o.format.code(),
                o.scale_format == ScaleFormat::E8m0,
                o.scale_layout.code(),
                out.dtype.into(),
            );
        }
        Ok(())
    }

    /// Import strided page-pool views by packing their raw bytes on the device.
    /// The scale swizzle is retained, not interpreted as numerical U8 values.
    pub fn from_views(
        data: RudaTensor<R>,
        scales: RudaTensor<R>,
        global: RudaTensor<R>,
        logical_shape: Shape,
        options: QuantOptions,
    ) -> Result<Self, FloatQuantError> {
        Self::from_parts(
            into_contiguous(data),
            into_contiguous(scales),
            into_contiguous(global),
            logical_shape,
            options,
        )
    }
    pub fn matrix_geometry(&self) -> (usize, usize, usize) {
        (self.groups, self.rows, self.width)
    }
    /// Update only selected logical rows. Untouched payload and scale bytes,
    /// including checkpoint padding, are preserved exactly. IDs < 0 skip.
    pub fn append_rows(
        &self,
        input: &RudaTensor<R>,
        destination_rows: &RudaTensor<R>,
    ) -> Result<Self, FloatQuantError> {
        let input = into_contiguous(input.clone());
        check(&input, &self.data)?;
        check(destination_rows, &self.data)?;
        if input.meta.num_dims() != 2
            || input.meta.shape()[1] != self.width
            || !matches!(input.dtype, DType::F32 | DType::F16 | DType::BF16)
            || destination_rows.dtype != DType::I32
            || destination_rows.meta.shape()[..] != [input.meta.shape()[0]]
        {
            return Err(FloatQuantError(
                "quantized row append expects [Q,K] and I32 destinations[Q]",
            ));
        }
        let q = input.meta.shape()[0];
        if q == 0 {
            return Ok(Self {
                data: self.data.clone(),
                scales: self.scales.clone(),
                global: self.global.clone(),
                logical_shape: self.logical_shape.clone(),
                options: self.options,
                groups: self.groups,
                rows: self.rows,
                width: self.width,
                storage_width: self.storage_width,
                scale_rows: self.scale_rows,
                scale_cols: self.scale_cols,
                global_mode: self.global_mode,
            });
        }
        let globals = alloc(&input, [q], DType::F32);
        if q > 0 {
            let dim = RudaDim::new(input.client.properties(), q);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, q, dim);
            row_globals::launch::<R>(
                &input.client,
                count,
                dim,
                self.global.clone().into_array_arg(),
                destination_rows.clone().into_array_arg(),
                globals.clone().into_array_arg(),
                q as u32,
                self.rows as u32,
                (self.groups * self.rows) as u32,
                self.global_mode,
            );
        }
        let rows = quantize(
            &input,
            Some(&globals),
            None,
            QuantOptions {
                scale_layout: ScaleLayout::Linear,
                ..self.options
            },
        )?;
        let copy = |t: &RudaTensor<R>| {
            let out = alloc(t, t.meta.shape().clone(), t.dtype);
            ruda_kernel::library::tensor::copy_into(
                &t.client,
                t.clone().binding(),
                out.clone().binding(),
                t.dtype.into(),
            );
            out
        };
        let data = copy(&self.data);
        let scales = copy(&self.scales);
        let payload_width = self.storage_width / self.options.format.packing();
        let blocks = self.storage_width / self.options.block_size;
        let units = q
            .checked_mul(payload_width.max(blocks))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(FloatQuantError("quantized append exceeds U32"))?;
        if units > 0 {
            let dim = RudaDim::new(input.client.properties(), units);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, units, dim);
            append_quantized::launch::<R>(
                &input.client,
                count,
                dim,
                rows.data.into_array_arg(),
                rows.scales.into_array_arg(),
                destination_rows.clone().into_array_arg(),
                data.clone().into_array_arg(),
                scales.clone().into_array_arg(),
                q as u32,
                self.rows as u32,
                (self.groups * self.rows) as u32,
                payload_width as u32,
                blocks as u32,
                self.scale_rows as u32,
                self.scale_cols as u32,
                self.options.scale_layout.code(),
            );
        }
        Ok(Self {
            data,
            scales,
            global: self.global.clone(),
            logical_shape: self.logical_shape.clone(),
            options: self.options,
            groups: self.groups,
            rows: self.rows,
            width: self.width,
            storage_width: self.storage_width,
            scale_rows: self.scale_rows,
            scale_cols: self.scale_cols,
            global_mode: self.global_mode,
        })
    }
    pub fn dequantize(&self, dtype: DType) -> Result<RudaTensor<R>, FloatQuantError> {
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(FloatQuantError("dequantized dtype must be F32/F16/BF16"));
        }
        check(&self.data, &self.global)?;
        check(&self.scales, &self.global)?;
        let out = alloc(&self.global, self.logical_shape.clone(), dtype);
        let n = self.groups * self.rows * self.width;
        if n != 0 {
            let dim = RudaDim::new(self.global.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.global.client, n, dim);
            dequantize_values::launch::<R>(
                &self.global.client,
                count,
                dim,
                self.data.clone().into_array_arg(),
                self.scales.clone().into_array_arg(),
                self.global.clone().into_array_arg(),
                out.clone().into_array_arg(),
                n as u32,
                self.rows as u32,
                self.width as u32,
                self.storage_width as u32,
                self.scale_rows as u32,
                self.scale_cols as u32,
                self.options.block_size as u32,
                self.global_mode,
                self.options.format.code(),
                self.options.scale_format == ScaleFormat::E8m0,
                self.options.scale_layout.code(),
                dtype.into(),
            );
        }
        Ok(out)
    }
    /// Import raw checkpoint buffers without reading their values. Scale and
    /// payload dimensions describe physical storage; logical_shape omits padding.
    pub fn from_parts(
        data: RudaTensor<R>,
        scales: RudaTensor<R>,
        global: RudaTensor<R>,
        logical_shape: Shape,
        options: QuantOptions,
    ) -> Result<Self, FloatQuantError> {
        check(&data, &global)?;
        check(&scales, &global)?;
        check(&global, &global)?;
        let rank = logical_shape.len();
        if rank < 2
            || data.dtype != DType::U8
            || scales.dtype != DType::U8
            || global.dtype != DType::F32
            || options.block_size == 0
            || options.alignment == 0
            || options.alignment % options.block_size != 0
        {
            return Err(FloatQuantError("invalid raw block-floating parts"));
        }
        let rows = logical_shape[rank - 2];
        let width = logical_shape[rank - 1];
        if rows == 0 || width == 0 {
            return Err(FloatQuantError("empty raw block-floating matrix"));
        }
        let groups = logical_shape.iter().take(rank - 2).product::<usize>();
        let storage_width = width.div_ceil(options.alignment) * options.alignment;
        let scale_rows = rows.div_ceil(options.scale_layout.row_alignment())
            * options.scale_layout.row_alignment();
        let blocks = storage_width / options.block_size;
        let scale_cols = if options.scale_layout == ScaleLayout::Linear {
            blocks
        } else {
            blocks.div_ceil(4) * 4
        };
        let global_mode = match global.meta.num_elements() {
            1 => 0,
            n if n == groups => 1,
            n if n == groups * rows => 2,
            _ => return Err(FloatQuantError("invalid raw global scale count")),
        };
        if data.meta.num_elements() != groups * rows * storage_width / options.format.packing()
            || scales.meta.num_elements() != groups * scale_rows * scale_cols
        {
            return Err(FloatQuantError("raw block-floating storage size mismatch"));
        }
        Ok(Self {
            data,
            scales,
            global,
            logical_shape,
            options,
            groups,
            rows,
            width,
            storage_width,
            scale_rows,
            scale_cols,
            global_mode,
        })
    }
}
/// Layout conversion also supports BF16 scale containers. Padding is zeroed.
pub fn block_scale_interleave<R: Runtime>(
    input: &RudaTensor<R>,
    layout: ScaleLayout,
    inverse: bool,
    logical_rows: usize,
    logical_cols: usize,
) -> Result<RudaTensor<R>, FloatQuantError> {
    check(input, input)?;
    if !matches!(input.dtype, DType::U8 | DType::BF16) {
        return Err(FloatQuantError("scale interleave requires U8/BF16"));
    }
    let pr = logical_rows.div_ceil(layout.row_alignment()) * layout.row_alignment();
    let pc = if layout == ScaleLayout::Linear {
        logical_cols
    } else {
        logical_cols.div_ceil(4) * 4
    };
    let per = if inverse {
        pr * pc
    } else {
        logical_rows * logical_cols
    };
    if per == 0 || input.meta.num_elements() % per != 0 {
        return Err(FloatQuantError("invalid scale layout dimensions"));
    }
    let groups = input.meta.num_elements() / per;
    let n = groups
        * if inverse {
            logical_rows * logical_cols
        } else {
            pr * pc
        };
    if n > u32::MAX as usize {
        return Err(FloatQuantError("scale layout exceeds U32"));
    }
    let out = alloc(input, [n], input.dtype);
    let dim = RudaDim::new(input.client.properties(), n);
    let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, n, dim);
    interleave::launch::<R>(
        &input.client,
        count,
        dim,
        input.clone().into_array_arg(),
        out.clone().into_array_arg(),
        groups as u32,
        logical_rows as u32,
        logical_cols as u32,
        pr as u32,
        pc as u32,
        layout.code(),
        inverse,
        input.dtype.into(),
    );
    Ok(out)
}
/// Matrix multiplication consumes native decoded device tensors and the
/// existing RUDA GEMM engine; it never calls a vendor GEMM implementation.
pub fn matmul<R: Runtime>(
    a: &BlockFloatTensor<R>,
    b: &BlockFloatTensor<R>,
    transpose_b: bool,
) -> Result<RudaTensor<R>, FloatQuantError> {
    let a = a.dequantize(DType::F32)?;
    let b = b.dequantize(DType::F32)?;
    crate::tensor_matmul::gemm(
        a,
        b,
        None,
        crate::tensor_matmul::GemmOptions {
            transpose_rhs: transpose_b,
            ..Default::default()
        },
        crate::tensor_matmul::MatmulStrategy::default(),
    )
    .map_err(|_| FloatQuantError("block-floating GEMM failed"))
}

#[ruda]
pub fn e2m1_decode(code: u32) -> f32 {
    let magnitude = code & 7u32;
    let mut value: f32 = 0.0f32;
    if magnitude == 1 {
        value = 0.5;
    }
    if magnitude == 2 {
        value = 1.0;
    }
    if magnitude == 3 {
        value = 1.5;
    }
    if magnitude == 4 {
        value = 2.0;
    }
    if magnitude == 5 {
        value = 3.0;
    }
    if magnitude == 6 {
        value = 4.0;
    }
    if magnitude == 7 {
        value = 6.0;
    }
    if code & 8u32 != 0 {
        value = -value;
    }
    value
}
#[ruda]
pub fn e2m1_encode(x: f32) -> u32 {
    let value = x.abs();
    let mut best: u32 = 0u32;
    let mut distance = value;
    #[unroll]
    for code in 1u32..8 {
        let d = (value - e2m1_decode(code)).abs();
        if d < distance || (d == distance && code & 1u32 == 0) {
            best = code;
            distance = d;
        }
    }
    if value == f32::INFINITY || value != value {
        best = 7;
    }
    best | ((u32::reinterpret(x) >> 28) & 8u32)
}
#[ruda]
pub fn fp8_decode(code: u32, #[comptime] e5m2: bool) -> f32 {
    let mut exponent = (code >> 3) & 15u32;
    let mut mantissa = code & 7u32;
    let mut bias: i32 = 7i32;
    let mut mantissa_bits: u32 = 3u32;
    if comptime!(e5m2) {
        exponent = (code >> 2) & 31u32;
        mantissa = code & 3u32;
        bias = 15;
        mantissa_bits = 2;
    }
    let mut value: f32 = 0.0f32;
    if exponent == 0 {
        value = mantissa as f32 * f32::powf(2.0, 1.0 - bias as f32 - mantissa_bits as f32);
    } else {
        value = (1.0 + mantissa as f32 / f32::powf(2.0, mantissa_bits as f32))
            * f32::powf(2.0, exponent as f32 - bias as f32);
    }
    if comptime!(e5m2) {
        if exponent == 31 {
            value = f32::INFINITY;
            if mantissa != 0 {
                value = f32::NAN;
            }
        }
    } else {
        if exponent == 15 && mantissa == 7 {
            value = f32::NAN;
        }
    }
    if code & 128u32 != 0 {
        value = -value;
    }
    value
}
#[ruda]
pub fn fp8_encode(x: f32, #[comptime] e5m2: bool) -> u32 {
    let mut max_code: u32 = 126u32;
    if comptime!(e5m2) {
        max_code = 123;
    }
    let mut target = x.abs();
    if target != target {
        target = fp8_decode(max_code, e5m2);
    }
    target = f32::min(target, fp8_decode(max_code, e5m2));
    let mut lo: u32 = 0u32;
    let mut hi = max_code;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if fp8_decode(mid, e5m2) <= target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dl = (target - fp8_decode(lo, e5m2)).abs();
    let dh = (fp8_decode(hi, e5m2) - target).abs();
    let mut code = hi;
    if dl < dh || (dl == dh && lo & 1u32 == 0) {
        code = lo;
    }
    let mut result = code | ((u32::reinterpret(x) >> 24) & 128u32);
    if x != x {
        result = 127u32;
        if comptime!(e5m2) {
            result = 127u32;
        }
    }
    result
}
#[ruda]
pub fn scale_offset(
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
    #[comptime] layout: u32,
) -> usize {
    let mut offset = row * cols + col;
    if comptime!(layout == 1) {
        offset = (row / 128) * (cols / 4) * 512
            + (col / 4) * 512
            + (row % 32) * 16
            + (row % 128 / 32) * 4
            + col % 4;
    }
    if comptime!(layout == 2) {
        offset = (row / 8) * (cols / 4) * 32 + (col / 4) * 32 + (row % 8) * 4 + col % 4;
    }
    if comptime!(layout == 3) {
        offset = (row / 4 * 4 + col / (cols / 4)) * cols + (col % (cols / 4)) * 4 + row % 4;
    }
    let _ = rows;
    offset
}
#[ruda]
fn decode_scale(bits: u32, #[comptime] e8m0: bool) -> f32 {
    let mut scale = fp8_decode(bits, false);
    if comptime!(e8m0) {
        scale = f32::powf(2.0, bits as f32 - 127.0);
        if bits == 255 {
            scale = f32::NAN;
        }
    }
    scale
}
#[ruda]
fn choose_scale(maximum: f32, limit: f32, #[comptime] e8m0: bool) -> u32 {
    let mut result = fp8_encode(maximum / limit, false);
    if comptime!(e8m0) {
        let value = maximum / limit;
        let bits = u32::reinterpret(value);
        let exponent = (bits >> 23) & 255u32;
        let mantissa = bits & 0x7fffffu32;
        result = exponent;
        if exponent == 0 {
            result = 0u32;
            if value > f32::reinterpret(0x00400000u32) {
                result = 1u32;
            }
        } else {
            if mantissa != 0 {
                result += 1u32;
            }
            result = u32::min(result, 254u32);
        }
        if value != value {
            result = 255u32;
        }
    }

    result
}
#[ruda(launch)]
pub fn quantize_blocks<F: Float>(
    x: &Array<F>,
    global: &Array<f32>,
    mask: &Array<i32>,
    data: &mut Array<u8>,
    scales: &mut Array<u8>,
    rows: u32,
    width: u32,
    storage_width: u32,
    scale_rows: u32,
    scale_cols: u32,
    block: u32,
    global_mode: u32,
    #[comptime] format: u32,
    #[comptime] e8m0: bool,
    #[comptime] layout: u32,
    #[comptime] masked: bool,
    #[comptime] four_over_six: bool,
    #[comptime] error_absolute: bool,
    #[comptime] error_fast_f16: bool,
    #[comptime] scale_limit: u32,
    #[define(F)] _dtype: StorageType,
) {
    let physical_row = RUDA_POS_X as usize;
    let group = physical_row / scale_rows as usize;
    let row = physical_row % scale_rows as usize;
    let chunk = RUDA_POS_Y as usize;
    let lane = UNIT_POS as usize;
    let lanes = PLANE_DIM as usize;
    let mut valid = row < rows as usize && chunk * (block as usize) < storage_width as usize;
    if comptime!(masked) {
        if row >= usize::cast_from(i32::max(0, mask[group])) {
            valid = false;
        }
    }
    let mut gi: usize = 0usize;
    if global_mode == 1 {
        gi = group;
    }
    if global_mode == 2 {
        gi = group * rows as usize + usize::min(row, rows as usize - 1);
    }
    let gs = global[gi];
    let mut maximum: f32 = 0.0f32;
    let mut j = lane;
    while j < block as usize {
        let col = chunk * block as usize + j;
        if valid && col < width as usize {
            maximum = f32::max(
                maximum,
                f32::cast_from(x[(group * rows as usize + row) * width as usize + col]).abs() / gs,
            );
        }
        j += lanes;
    }
    maximum = plane_max(maximum);
    let mut limit: f32 = 448.0f32;
    if comptime!(format == 0) {
        limit = 6.0;
    }
    if comptime!(format == 2) {
        limit = 57344.0;
    }
    let mut bits = choose_scale(maximum, limit, e8m0);
    if comptime!(!e8m0) {
        bits = u32::min(bits, fp8_encode(scale_limit as f32, false));
    }
    let mut scale = decode_scale(bits, e8m0);
    if comptime!(four_over_six && format == 0 && !e8m0) {
        let candidate = u32::min(
            choose_scale(maximum, 4.0, false),
            fp8_encode(scale_limit as f32, false),
        );
        let candidate_scale = decode_scale(candidate, false);
        let mut first_loss: f32 = 0.0f32;
        let mut second_loss: f32 = 0.0f32;
        j = lane;
        while j < block as usize {
            let col = chunk * block as usize + j;
            if valid && col < width as usize {
                let v =
                    f32::cast_from(x[(group * rows as usize + row) * width as usize + col]) / gs;
                let mut a: f32 = 0.0f32;
                let mut b: f32 = 0.0f32;
                if scale > 0.0 {
                    a = e2m1_decode(e2m1_encode(v / scale)) * scale;
                }
                if candidate_scale > 0.0 {
                    b = e2m1_decode(e2m1_encode(v / candidate_scale)) * candidate_scale;
                }
                let mut first = a - v;
                let mut second = b - v;
                if comptime!(error_fast_f16) {
                    first = f32::cast_from(half::f16::cast_from(first));
                    second = f32::cast_from(half::f16::cast_from(second));
                }
                if comptime!(error_absolute) {
                    first_loss += first.abs();
                    second_loss += second.abs();
                } else {
                    first_loss += first * first;
                    second_loss += second * second;
                }
            }
            j += lanes;
        }
        first_loss = plane_sum(first_loss);
        second_loss = plane_sum(second_loss);
        if second_loss < first_loss {
            bits = candidate;
            scale = candidate_scale;
        }
    }
    if lane == 0 {
        scales[group * scale_rows as usize * scale_cols as usize
            + scale_offset(row, chunk, scale_rows as usize, scale_cols as usize, layout)] =
            u8::cast_from(bits);
    }
    if row < rows as usize {
        if comptime!(format == 0) {
            j = lane * 2;
            while j < block as usize {
                let col = chunk * block as usize + j;
                if col < storage_width as usize {
                    let mut a: f32 = 0.0f32;
                    let mut b: f32 = 0.0f32;
                    if valid && scale > 0.0 {
                        if col < width as usize {
                            a = f32::cast_from(
                                x[(group * rows as usize + row) * width as usize + col],
                            ) / gs
                                / scale;
                        }
                        if col + 1 < width as usize {
                            b = f32::cast_from(
                                x[(group * rows as usize + row) * width as usize + col + 1],
                            ) / gs
                                / scale;
                        }
                    }
                    data[(group * rows as usize + row) * (storage_width as usize / 2) + col / 2] =
                        u8::cast_from(e2m1_encode(a) | (e2m1_encode(b) << 4));
                }
                j += lanes * 2;
            }
        } else {
            j = lane;
            while j < block as usize {
                let col = chunk * block as usize + j;
                if col < storage_width as usize {
                    let mut value: f32 = 0.0f32;
                    if valid && scale > 0.0 && col < width as usize {
                        value =
                            f32::cast_from(x[(group * rows as usize + row) * width as usize + col])
                                / gs
                                / scale;
                    }
                    data[(group * rows as usize + row) * storage_width as usize + col] =
                        u8::cast_from(fp8_encode(value, format == 2));
                }
                j += lanes;
            }
        }
    }
}
#[ruda(launch)]
pub fn dequantize_values<O: Float>(
    data: &Array<u8>,
    scales: &Array<u8>,
    global: &Array<f32>,
    out: &mut Array<O>,
    n: u32,
    rows: u32,
    width: u32,
    storage_width: u32,
    scale_rows: u32,
    scale_cols: u32,
    block: u32,
    global_mode: u32,
    #[comptime] format: u32,
    #[comptime] e8m0: bool,
    #[comptime] layout: u32,
    #[define(O)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let row = i / width as usize;
        let col = i % width as usize;
        let group = row / rows as usize;
        let local = row % rows as usize;
        let mut gi: usize = 0usize;
        if global_mode == 1 {
            gi = group;
        }
        if global_mode == 2 {
            gi = row;
        }
        let mut value: f32 = 0.0f32;
        if comptime!(format == 0) {
            let byte = u32::cast_from(data[row * (storage_width as usize / 2) + col / 2]);
            value = e2m1_decode((byte >> ((col % 2) * 4) as u32) & 15u32);
        } else {
            value = fp8_decode(
                u32::cast_from(data[row * storage_width as usize + col]),
                format == 2,
            );
        }
        let offset = group * scale_rows as usize * scale_cols as usize
            + scale_offset(
                local,
                col / block as usize,
                scale_rows as usize,
                scale_cols as usize,
                layout,
            );
        out[i] =
            O::cast_from(value * decode_scale(u32::cast_from(scales[offset]), e8m0) * global[gi]);
    }
}
#[ruda(launch)]
pub fn interleave<I: Numeric>(
    input: &Array<I>,
    out: &mut Array<I>,
    groups: u32,
    rows: u32,
    cols: u32,
    pr: u32,
    pc: u32,
    #[comptime] layout: u32,
    #[comptime] inverse: bool,
    #[define(I)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    let mut per = pr as usize * pc as usize;
    if comptime!(inverse) {
        per = rows as usize * cols as usize;
    }
    if i < groups as usize * per {
        let group = i / per;
        let local = i % per;
        if comptime!(inverse) {
            let row = local / cols as usize;
            let col = local % cols as usize;
            let source = group * pr as usize * pc as usize
                + scale_offset(row, col, pr as usize, pc as usize, layout);
            out[i] = input[source];
        } else {
            let row = local / pc as usize;
            let col = local % pc as usize;
            let dest = group * pr as usize * pc as usize
                + scale_offset(row, col, pr as usize, pc as usize, layout);
            let mut value = I::cast_from(0u32);
            if row < rows as usize && col < cols as usize {
                value = input[group * rows as usize * cols as usize + row * cols as usize + col];
            }
            out[dest] = value;
        }
    }
}
#[ruda(launch)]
pub fn row_globals(
    global: &Array<f32>,
    ids: &Array<i32>,
    out: &mut Array<f32>,
    q: u32,
    rows: u32,
    total: u32,
    mode: u32,
) {
    let i = ABSOLUTE_POS as usize;
    if i < q as usize {
        let row = ids[i];
        let mut value: f32 = 1.0f32;
        if row >= 0 && row < total as i32 {
            let mut gi: usize = 0usize;
            if mode == 1 {
                gi = row as usize / rows as usize;
            }
            if mode == 2 {
                gi = row as usize;
            }
            value = global[gi];
        }
        out[i] = value;
    }
}
#[ruda(launch)]
pub fn permute_quantized_rows(
    source: &Array<u8>,
    source_sf: &Array<u8>,
    source_global: &Array<f32>,
    ids: &Array<i32>,
    out: &mut Array<u8>,
    sf: &mut Array<u8>,
    global: &mut Array<f32>,
    n: u32,
    rows: u32,
    payload: u32,
    sr: u32,
    sc: u32,
    #[comptime] row_globals: bool,
    #[comptime] layout: u32,
) {
    let i = ABSOLUTE_POS as usize;
    let width = u32::max(payload, sc) as usize;
    if i < n as usize {
        let row = i / width;
        let col = i % width;
        let dest = ids[row];
        if dest >= 0 && dest < rows as i32 {
            if col < payload as usize {
                out[dest as usize * payload as usize + col] = source[row * payload as usize + col];
            }
            if col < sc as usize {
                sf[scale_offset(dest as usize, col, sr as usize, sc as usize, layout)] =
                    source_sf[scale_offset(row, col, sr as usize, sc as usize, layout)];
            }
            if comptime!(row_globals) {
                if col == 0 {
                    global[dest as usize] = source_global[row];
                }
            }
        }
    }
}
#[ruda(launch)]
pub fn append_quantized(
    source: &Array<u8>,
    sf: &Array<u8>,
    ids: &Array<i32>,
    data: &mut Array<u8>,
    scales: &mut Array<u8>,
    q: u32,
    rows: u32,
    total: u32,
    payload_width: u32,
    blocks: u32,
    sr: u32,
    sc: u32,
    #[comptime] layout: u32,
) {
    let i = ABSOLUTE_POS as usize;
    let width = u32::max(payload_width, blocks) as usize;
    if i < q as usize * width {
        let query = i / width;
        let col = i % width;
        let row = ids[query];
        if row >= 0 && row < total as i32 {
            if col < payload_width as usize {
                data[row as usize * payload_width as usize + col] =
                    source[query * payload_width as usize + col];
            }
            if col < blocks as usize {
                let group = row as usize / rows as usize;
                let local = row as usize % rows as usize;
                scales[group * sr as usize * sc as usize
                    + scale_offset(local, col, sr as usize, sc as usize, layout)] =
                    sf[query * blocks as usize + col];
            }
        }
    }
}

/// CPU checkpoint decoder. This is a public data conversion, not a GPU test or
/// fallback: callers supply host byte slices and receive ordinary Rust values.
pub fn dequantize_host(
    data: &[u8],
    scales: &[u8],
    global: &[f32],
    groups: usize,
    rows: usize,
    width: usize,
    storage_width: usize,
    options: QuantOptions,
) -> Result<Vec<f32>, FloatQuantError> {
    if groups == 0
        || rows == 0
        || width == 0
        || options.block_size == 0
        || storage_width < width
        || storage_width % options.block_size != 0
        || storage_width % options.format.packing() != 0
    {
        return Err(FloatQuantError("invalid host block-float geometry"));
    }
    let sr =
        rows.div_ceil(options.scale_layout.row_alignment()) * options.scale_layout.row_alignment();
    let mut sc = storage_width / options.block_size;
    if options.scale_layout != ScaleLayout::Linear {
        sc = sc.div_ceil(4) * 4;
    }
    let count = groups
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(width))
        .ok_or(FloatQuantError("host decode size overflow"))?;
    let bytes = groups
        .checked_mul(rows)
        .and_then(|n| n.checked_mul(storage_width / options.format.packing()))
        .ok_or(FloatQuantError("host payload size overflow"))?;
    let sf = groups
        .checked_mul(sr)
        .and_then(|n| n.checked_mul(sc))
        .ok_or(FloatQuantError("host scale size overflow"))?;
    if data.len() != bytes
        || scales.len() != sf
        || !(global.len() == 1 || global.len() == groups || global.len() == groups * rows)
    {
        return Err(FloatQuantError("host payload/scale size mismatch"));
    }
    let mut out = vec![0.0; count];
    for group in 0..groups {
        for row in 0..rows {
            for col in 0..width {
                let offset = (group * rows + row) * storage_width + col;
                let payload = if options.format == FloatFormat::E2m1 {
                    host_e2m1((data[offset / 2] >> ((col % 2) * 4)) & 15)
                } else {
                    host_fp8(data[offset], options.format == FloatFormat::E5m2)
                };
                let c = col / options.block_size;
                let local = host_scale_offset(row, c, sr, sc, options.scale_layout);
                let bits = scales[group * sr * sc + local];
                let scale = if options.scale_format == ScaleFormat::E8m0 {
                    if bits == 255 {
                        f32::NAN
                    } else {
                        2.0f32.powi(bits as i32 - 127)
                    }
                } else {
                    host_fp8(bits, false)
                };
                let gi = if global.len() == 1 {
                    0
                } else if global.len() == groups {
                    group
                } else {
                    group * rows + row
                };
                out[(group * rows + row) * width + col] = payload * scale * global[gi];
            }
        }
    }
    Ok(out)
}
pub fn host_e2m1(code: u8) -> f32 {
    let values = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let x = values[(code & 7) as usize];
    if code & 8 != 0 { -x } else { x }
}
pub fn host_fp8(code: u8, e5m2: bool) -> f32 {
    let (e, m, bias, bits) = if e5m2 {
        ((code >> 2) & 31, code & 3, 15, 2)
    } else {
        ((code >> 3) & 15, code & 7, 7, 3)
    };
    let x = if e5m2 && e == 31 {
        if m == 0 { f32::INFINITY } else { f32::NAN }
    } else if !e5m2 && e == 15 && m == 7 {
        f32::NAN
    } else if e == 0 {
        (m as f32) * 2.0f32.powi(1 - bias - bits)
    } else {
        (1.0 + m as f32 / 2.0f32.powi(bits)) * 2.0f32.powi(e as i32 - bias)
    };
    if code & 128 != 0 { -x } else { x }
}
pub fn host_scale_offset(
    row: usize,
    col: usize,
    _rows: usize,
    cols: usize,
    layout: ScaleLayout,
) -> usize {
    match layout {
        ScaleLayout::Linear => row * cols + col,
        ScaleLayout::Block128x4 => {
            (row / 128) * (cols / 4) * 512
                + (col / 4) * 512
                + (row % 32) * 16
                + (row % 128 / 32) * 4
                + col % 4
        }
        ScaleLayout::Block8x4 => {
            (row / 8) * (cols / 4) * 32 + (col / 4) * 32 + (row % 8) * 4 + col % 4
        }
        ScaleLayout::PagedV4x4 => {
            (row / 4 * 4 + col / (cols / 4)) * cols + (col % (cols / 4)) * 4 + row % 4
        }
    }
}

/// Checkpoint row permutation shared by payload and its un-interleaved scales.
pub fn shuffle_rows<R: Runtime>(
    input: &RudaTensor<R>,
    epilogue_tile_m: usize,
    inverse: bool,
) -> Result<RudaTensor<R>, FloatQuantError> {
    check(input, input)?;
    let rank = input.meta.num_dims();
    if rank < 2
        || !matches!(
            input.dtype,
            DType::U8 | DType::BF16 | DType::F16 | DType::F32
        )
    {
        return Err(FloatQuantError("row shuffle expects numeric [...,M,K]"));
    }
    let rows = input.meta.shape()[rank - 2];
    let cols = input.meta.shape()[rank - 1];
    let block = if epilogue_tile_m % 128 == 0 { 32 } else { 16 };
    if rows == 0 || cols == 0 || rows % block != 0 {
        return Err(FloatQuantError(
            "row shuffle requires complete 16/32-row blocks",
        ));
    }
    let out = alloc(input, input.meta.shape().clone(), input.dtype);
    let n = input.meta.num_elements();
    if n > 0 {
        let dim = RudaDim::new(input.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&input.client, n, dim);
        row_shuffle::launch::<R>(
            &input.client,
            count,
            dim,
            input.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            rows as u32,
            cols as u32,
            block as u32,
            inverse,
            input.dtype.into(),
        );
    }
    Ok(out)
}
/// Checkpoint preparation for the documented braided MegaMoE formats. NVFP4
/// stores 80-byte K128 rows; MXFP4 stores chunk-major N128 tiles of 8704 bytes.
/// This is a byte-layout conversion only, with no foreign numerical kernel.
pub fn mega_moe_weights<R: Runtime>(
    packed: &RudaTensor<R>,
    linear_scales: &RudaTensor<R>,
    mxfp4: bool,
    interleave_gate_up: bool,
) -> Result<RudaTensor<R>, FloatQuantError> {
    check(packed, packed)?;
    check(linear_scales, packed)?;
    if packed.dtype != DType::U8
        || linear_scales.dtype != DType::U8
        || packed.meta.num_dims() != 3
        || linear_scales.meta.num_dims() != 3
    {
        return Err(FloatQuantError(
            "MegaMoE preparation requires rank-three U8 weights/scales",
        ));
    }
    let s = packed.meta.shape();
    let (experts, rows, half_width) = (s[0], s[1], s[2]);
    let width = half_width
        .checked_mul(2)
        .ok_or(FloatQuantError("MegaMoE K overflow"))?;
    let group = if mxfp4 { 32 } else { 16 };
    if rows == 0
        || rows % 128 != 0
        || width == 0
        || width % 128 != 0
        || linear_scales.meta.shape()[..] != [experts, rows, width / group]
    {
        return Err(FloatQuantError(
            "MegaMoE weights need complete N128/K128 tiles",
        ));
    }
    let kb = width / 128;
    let shape = if mxfp4 {
        Shape::from([experts, rows / 128 * kb, 8704])
    } else {
        Shape::from([experts, rows, kb * 80])
    };
    let n = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|n| *n <= u32::MAX as usize)
        .ok_or(FloatQuantError("MegaMoE packed weights exceed U32"))?;
    let out = alloc(packed, shape, DType::U8);
    if n > 0 {
        let dim = RudaDim::new(packed.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&packed.client, n, dim);
        mega_layout::launch::<R>(
            &packed.client,
            count,
            dim,
            packed.clone().into_array_arg(),
            linear_scales.clone().into_array_arg(),
            out.clone().into_array_arg(),
            n as u32,
            rows as u32,
            kb as u32,
            mxfp4,
            interleave_gate_up,
        );
    }
    Ok(out)
}
#[ruda]
fn braided_byte(input: &Array<u8>, base: usize, col: usize) -> u8 {
    let group = col / 4;
    let pair = col % 4;
    let a = pair * 2;
    let b = a + 1;
    let mut shift: u32 = 0u32;
    if a < 4 {
        shift = 4;
    }
    let first = (u32::cast_from(input[base + group * 4 + a % 4]) >> shift) & 7u32;
    let second = (u32::cast_from(input[base + group * 4 + b % 4]) >> shift) & 7u32;
    let sign_source = u32::cast_from(input[base + group * 4 + pair]);
    u8::cast_from(first | (sign_source & 8u32) | (second << 4) | (sign_source & 128u32))
}
#[ruda(launch)]
pub fn mega_layout(
    input: &Array<u8>,
    scales: &Array<u8>,
    out: &mut Array<u8>,
    n: u32,
    rows: u32,
    kb: u32,
    #[comptime] mx: bool,
    #[comptime] gated: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let mut expert: usize = 0usize;
        let mut row: usize = 0usize;
        let mut kblock: usize = 0usize;
        let mut column: usize = 0usize;
        let mut scale = false;
        let mut padding = false;
        if comptime!(mx) {
            let tile = i / 8704;
            let local = i % 8704;
            expert = tile / (rows as usize / 128 * kb as usize);
            let nt = tile / kb as usize % (rows as usize / 128);
            kblock = tile % kb as usize;
            if local < 8192 {
                row = nt * 128 + (local / 16) % 128;
                column = local / (128 * 16) * 16 + local % 16;
            } else {
                scale = true;
                row = nt * 128 + (local - 8192) / 4;
                column = (local - 8192) % 4;
            }
        } else {
            let per = kb as usize * 80;
            expert = i / (rows as usize * per);
            row = i / per % rows as usize;
            kblock = i / 80 % kb as usize;
            let local = i % 80;
            if local < 64 {
                column = local;
            } else {
                scale = true;
                column = local - 64;
                if column >= 8 {
                    padding = true;
                }
            }
        }
        let mut source_row = row;
        if comptime!(gated) {
            source_row = row / 16 * 8 + row % 8 + (row % 16 / 8) * (rows as usize / 2);
        }
        let mut value = 0u8;
        if !padding {
            if scale {
                let mut per_block: usize = 8usize;
                if comptime!(mx) {
                    per_block = 4;
                }
                value = scales[(expert * rows as usize + source_row) * kb as usize * per_block
                    + kblock * per_block
                    + column];
            } else {
                let base = ((expert * rows as usize + source_row) * kb as usize + kblock) * 64;
                value = braided_byte(input, base, column);
            }
        }
        out[i] = value;
    }
}
#[ruda(launch)]
pub fn row_shuffle<I: Numeric>(
    x: &Array<I>,
    y: &mut Array<I>,
    n: u32,
    rows: u32,
    cols: u32,
    block: u32,
    #[comptime] inverse: bool,
    #[define(I)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let col = i % cols as usize;
        let row = i / cols as usize % rows as usize;
        let group = i / (rows as usize * cols as usize);
        let local = row % block as usize;
        let stride = block as usize / 8;
        let mapped = row / block as usize * block as usize + (local % stride) * 8 + local / stride;
        let a = (group * rows as usize + row) * cols as usize + col;
        let b = (group * rows as usize + mapped) * cols as usize + col;
        if comptime!(inverse) {
            y[a] = x[b];
        } else {
            y[b] = x[a];
        }
    }
}
