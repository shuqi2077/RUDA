//! Native expert-parallel FP4/FP8 MoE using explicitly owned peer mappings.
//! All dispatch, expert arithmetic and top-k reduction are Rust kernels. Only
//! addresses are exchanged on the host; no activations are host-staged.
use super::{InferenceError, alloc, check, linear::Matrix};
use ruda_core::tensor::DType;
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
#[derive(Clone, Copy, Debug)]
pub struct PeerLayout {
    pub capacity: usize,
    pub width: usize,
    pub top_k: usize,
    pub world: usize,
    pub scale_offset: usize,
    pub ids_offset: usize,
    pub weights_offset: usize,
    pub contributions_offset: usize,
    pub control_offset: usize,
    pub bytes: usize,
}
impl PeerLayout {
    pub fn new(
        world: usize,
        capacity: usize,
        width: usize,
        top_k: usize,
    ) -> Result<Self, InferenceError> {
        if world == 0 || capacity == 0 || width == 0 || width % 128 != 0 || top_k == 0 {
            return Err(InferenceError(
                "peer MoE requires positive capacities and K divisible by 128",
            ));
        }
        let align = |n: usize| n.checked_add(15).map(|n| n / 16 * 16);
        let x = capacity
            .checked_mul(width)
            .ok_or(InferenceError("peer payload overflow"))?;
        let scale_offset = align(x).ok_or(InferenceError("peer alignment overflow"))?;
        let sf = capacity
            .checked_mul(width / 128)
            .and_then(|n| n.checked_mul(4))
            .ok_or(InferenceError("peer scales overflow"))?;
        let route = capacity
            .checked_mul(top_k)
            .and_then(|n| n.checked_mul(4))
            .ok_or(InferenceError("peer routes overflow"))?;
        let ids_offset = align(
            scale_offset
                .checked_add(sf)
                .ok_or(InferenceError("peer layout overflow"))?,
        )
        .ok_or(InferenceError("peer layout overflow"))?;
        let weights_offset = align(
            ids_offset
                .checked_add(route)
                .ok_or(InferenceError("peer layout overflow"))?,
        )
        .ok_or(InferenceError("peer layout overflow"))?;
        let contributions_offset = align(
            weights_offset
                .checked_add(route)
                .ok_or(InferenceError("peer layout overflow"))?,
        )
        .ok_or(InferenceError("peer layout overflow"))?;
        let contributions = x
            .checked_mul(world)
            .and_then(|n| n.checked_mul(4))
            .ok_or(InferenceError("peer contributions overflow"))?;
        let control_offset = align(
            contributions_offset
                .checked_add(contributions)
                .ok_or(InferenceError("peer layout overflow"))?,
        )
        .ok_or(InferenceError("peer layout overflow"))?;
        let bytes = control_offset
            .checked_add(32)
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(InferenceError("peer layout exceeds U32"))?;
        Ok(Self {
            capacity,
            width,
            top_k,
            world,
            scale_offset,
            ids_offset,
            weights_offset,
            contributions_offset,
            control_offset,
            bytes,
        })
    }
}
pub struct PeerMoe<R: Runtime> {
    pub addresses: RudaTensor<R>,
    pub output: RudaTensor<R>,
    hidden: RudaTensor<R>,
    hidden_quantizer: super::linear::ScaledFp8<R>,
    w1: RudaTensor<R>,
    w2: RudaTensor<R>,
    layout: PeerLayout,
    rank: usize,
    local_experts: usize,
    inner: usize,
    activation_clamp: f32,
}
impl<R: Runtime> PeerMoe<R> {
    /// Every address points to a PeerLayout-sized GPU allocation mapped into
    /// this device's address space. Retain all owners/imports until completion.
    /// All ranks must use identical geometry and non-overlapping local experts.
    pub unsafe fn prepare(
        addresses: &RudaTensor<R>,
        layout: PeerLayout,
        rank: usize,
        w1: Matrix<'_, R>,
        w2: Matrix<'_, R>,
        activation_clamp: f32,
        output_dtype: DType,
    ) -> Result<Self, InferenceError> {
        check(addresses, addresses)?;
        if addresses.dtype != DType::U64
            || addresses.meta.shape()[..] != [layout.world]
            || rank >= layout.world
            || !activation_clamp.is_finite()
            || activation_clamp <= 0.0
        {
            return Err(InferenceError("invalid peer MoE address table/rank/clamp"));
        }
        let w1 = w1.decode()?;
        let w2 = w2.decode()?;
        for t in [&w1, &w2] {
            check(t, addresses)?;
        }
        if w1.meta.num_dims() != 3
            || w2.meta.num_dims() != 3
            || w1.meta.shape()[0] != w2.meta.shape()[0]
            || w1.meta.shape()[2] != layout.width
            || w2.meta.shape()[1] != layout.width
            || w1.meta.shape()[1] != w2.meta.shape()[2] * 2
        {
            return Err(InferenceError(
                "peer expert weights need local [E,2I,K] and [E,K,I]",
            ));
        }
        let local_experts = w1.meta.shape()[0];
        let inner = w2.meta.shape()[2];
        if local_experts == 0 || inner == 0 {
            return Err(InferenceError("peer expert axes must be nonempty"));
        }
        let n = layout
            .world
            .checked_mul(layout.capacity)
            .and_then(|n| n.checked_mul(layout.top_k))
            .and_then(|n| n.checked_mul(inner))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(InferenceError("peer expert workspace exceeds U32"))?;
        let hidden = alloc(
            addresses,
            [layout.world * layout.capacity * layout.top_k, inner],
            DType::F32,
        );
        let hidden_quantizer =
            super::linear::ScaledFp8::allocate(&hidden, 1, 128, super::linear::FloatFormat::E4m3)?;
        let output = alloc(addresses, [layout.capacity, layout.width], output_dtype);
        super::float(&output)?;
        initialize::launch::<R>(
            &addresses.client,
            RudaCount::new_single(),
            RudaDim::new_single(),
            addresses.clone().into_array_arg(),
            rank as u32,
            layout.control_offset as u64,
        );
        Ok(Self {
            addresses: addresses.clone(),
            output,
            hidden,
            hidden_quantizer,
            w1,
            w2,
            layout,
            rank,
            local_experts,
            inner,
            activation_clamp,
        })
    }
    /// Stage one rank's producer tensors. Epochs start at one and increase by exactly one; they are
    /// nonzero; all ranks use the same epoch; live row counts may differ. Do not reuse an
    /// epoch or overwrite the next step before the previous combine completed.
    pub unsafe fn stage(
        &self,
        x_fp8: &RudaTensor<R>,
        scales: &RudaTensor<R>,
        ids: &RudaTensor<R>,
        weights: &RudaTensor<R>,
        rows: usize,
        epoch: u32,
    ) -> Result<(), InferenceError> {
        let l = self.layout;
        for t in [x_fp8, scales, ids, weights] {
            check(t, &self.addresses)?;
        }
        if epoch == 0
            || rows > l.capacity
            || x_fp8.dtype != DType::U8
            || x_fp8.meta.shape()[..] != [rows, l.width]
            || scales.dtype != DType::F32
            || scales.meta.shape()[..] != [rows, l.width / 128]
            || ids.dtype != DType::I32
            || ids.meta.shape()[..] != [rows, l.top_k]
            || weights.dtype != DType::F32
            || weights.meta.shape() != ids.meta.shape()
        {
            return Err(InferenceError(
                "peer MoE stage dimensions/dtypes/epoch mismatch",
            ));
        }
        reset::launch::<R>(
            &self.addresses.client,
            RudaCount::new_single(),
            RudaDim::new_single(),
            self.addresses.clone().into_array_arg(),
            self.rank as u32,
            l.world as u32,
            epoch,
            l.control_offset as u64,
        );
        let n = (rows * l.width)
            .max(rows * l.top_k)
            .max(rows * l.width / 128)
            .max(1);
        let dim = RudaDim::new(x_fp8.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&x_fp8.client, n, dim);
        stage_inputs::launch::<R>(
            &x_fp8.client,
            count,
            dim,
            x_fp8.clone().into_array_arg(),
            scales.clone().into_array_arg(),
            ids.clone().into_array_arg(),
            weights.clone().into_array_arg(),
            self.addresses.clone().into_array_arg(),
            n as u32,
            rows as u32,
            l.width as u32,
            l.top_k as u32,
            self.rank as u32,
            epoch,
            l.scale_offset as u64,
            l.ids_offset as u64,
            l.weights_offset as u64,
            l.control_offset as u64,
        );
        Ok(())
    }
    /// Enqueue local expert computation and publish one contribution plane to
    /// each origin rank. Peer mappings must support native remote atomics.
    pub unsafe fn compute(&self, rows: usize, epoch: u32) -> Result<(), InferenceError> {
        let l = self.layout;
        if rows > l.capacity || epoch == 0 {
            return Err(InferenceError("invalid peer computation epoch/rows"));
        }
        let rows = l.capacity;
        let n = l.world * rows * l.top_k * self.inner;
        let dim = RudaDim::new(self.addresses.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&self.addresses.client, n, dim);
        first::launch::<R>(
            &self.addresses.client,
            count,
            dim,
            self.addresses.clone().into_array_arg(),
            self.w1.clone().into_array_arg(),
            self.hidden.clone().into_array_arg(),
            n as u32,
            rows as u32,
            l.width as u32,
            l.top_k as u32,
            self.inner as u32,
            self.local_experts as u32,
            self.rank as u32,
            epoch,
            self.activation_clamp,
            l.scale_offset as u64,
            l.ids_offset as u64,
            l.control_offset as u64,
        );
        unsafe {
            self.hidden_quantizer.quantize_into(&self.hidden)?;
            self.hidden_quantizer.decode_into(&self.hidden)?;
        }
        let n = l.world * rows * l.width;
        let dim = RudaDim::new(self.addresses.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&self.addresses.client, n, dim);
        second::launch::<R>(
            &self.addresses.client,
            count,
            dim,
            self.addresses.clone().into_array_arg(),
            self.w2.clone().into_array_arg(),
            self.hidden.clone().into_array_arg(),
            n as u32,
            rows as u32,
            l.width as u32,
            l.top_k as u32,
            self.inner as u32,
            self.local_experts as u32,
            self.rank as u32,
            l.capacity as u32,
            epoch,
            l.ids_offset as u64,
            l.weights_offset as u64,
            l.contributions_offset as u64,
            l.control_offset as u64,
        );
        Ok(())
    }
    /// GPU-side rendezvous and deterministic rank-order top-k reduction. No
    /// host polling, tensor transfer or host wait is performed.
    pub unsafe fn combine(&self, rows: usize, epoch: u32) -> Result<RudaTensor<R>, InferenceError> {
        let l = self.layout;
        if rows > l.capacity || epoch == 0 {
            return Err(InferenceError("invalid peer combine epoch/rows"));
        }
        if rows == 0 {
            combine_empty::launch::<R>(
                &self.addresses.client,
                RudaCount::new_single(),
                RudaDim::new_single(),
                self.addresses.clone().into_array_arg(),
                l.world as u32,
                self.rank as u32,
                epoch,
                l.control_offset as u64,
            );
            return Ok(self.output.clone());
        }
        let n = rows * l.width;
        let dim = RudaDim::new(self.addresses.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&self.addresses.client, n, dim);
        combine::launch::<R>(
            &self.addresses.client,
            count,
            dim,
            self.addresses.clone().into_array_arg(),
            self.output.clone().into_array_arg(),
            n as u32,
            l.width as u32,
            l.world as u32,
            self.rank as u32,
            l.capacity as u32,
            epoch,
            l.contributions_offset as u64,
            l.control_offset as u64,
            self.output.dtype.into(),
        );
        Ok(self.output.clone())
    }
}
#[ruda]
fn await_epoch(address: u64, control: u64, index: usize, epoch: u32) {
    let flag = ValuePtr::<Atomic<u32>, GlobalSpace>::from_native(address + control).offset(index);
    while flag.atomic_load() != epoch {}
    fence_system();
}
#[ruda(launch)]
pub fn reset(addresses: &Array<u64>, rank: u32, world: u32, epoch: u32, control: u64) {
    for peer in 0..world as usize {
        await_epoch(addresses[peer], control, 5, epoch - 1);
    }
    let ptr = ValuePtr::<Atomic<u32>, GlobalSpace, ReadWrite>::from_native(
        addresses[rank as usize] + control,
    );
    ptr.atomic_store(0);
    ptr.offset(2).atomic_store(0);
    ptr.offset(4).atomic_store(0);
}
#[ruda(launch)]
pub fn stage_inputs(
    x: &Array<u8>,
    scales: &Array<f32>,
    ids: &Array<i32>,
    weights: &Array<f32>,
    addresses: &Array<u64>,
    n: u32,
    rows: u32,
    width: u32,
    k: u32,
    rank: u32,
    epoch: u32,
    sf_offset: u64,
    ids_offset: u64,
    weights_offset: u64,
    control: u64,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let base = addresses[rank as usize];
        if i < rows as usize * width as usize {
            ValuePtr::<u8, GlobalSpace, ReadWrite>::from_native(base)
                .offset(i)
                .store(x[i]);
        }
        if i < rows as usize * (width / 128) as usize {
            ValuePtr::<f32, GlobalSpace, ReadWrite>::from_native(base + sf_offset)
                .offset(i)
                .store(scales[i]);
        }
        if i < rows as usize * k as usize {
            ValuePtr::<i32, GlobalSpace, ReadWrite>::from_native(base + ids_offset)
                .offset(i)
                .store(ids[i]);
            ValuePtr::<f32, GlobalSpace, ReadWrite>::from_native(base + weights_offset)
                .offset(i)
                .store(weights[i]);
        }
        fence_system();
        let flags = ValuePtr::<Atomic<u32>, GlobalSpace, ReadWrite>::from_native(base + control);
        if i == 0 {
            flags.offset(6).atomic_store(rows);
        }
        let old = flags.fetch_add(1);
        if old == n - 1 {
            fence_system();
            flags.offset(1).atomic_store(epoch);
        }
    }
}
#[ruda(launch)]
pub fn first(
    addresses: &Array<u64>,
    w: &Array<f32>,
    hidden: &mut Array<f32>,
    n: u32,
    rows: u32,
    width: u32,
    k: u32,
    inner: u32,
    experts: u32,
    rank: u32,
    epoch: u32,
    limit: f32,
    sf_offset: u64,
    ids_offset: u64,
    control: u64,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let col = i % inner as usize;
        let route = i / inner as usize;
        let origin = route / (rows as usize * k as usize);
        let token = route / k as usize % rows as usize;
        let selected = route % k as usize;
        let base = addresses[origin];
        await_epoch(base, control, 1, epoch);
        let live = ValuePtr::<Atomic<u32>, GlobalSpace>::from_native(base + control)
            .offset(6)
            .atomic_load();
        let mut id: i32 = -1i32;
        if token < live as usize {
            id = ValuePtr::<i32, GlobalSpace>::from_native(base + ids_offset)
                .offset(token * k as usize + selected)
                .load();
        }
        let e = id - rank as i32 * experts as i32;
        let mut result: f32 = 0.0f32;
        if e >= 0 && e < experts as i32 {
            let x = ValuePtr::<u8, GlobalSpace>::from_native(base);
            let scales = ValuePtr::<f32, GlobalSpace>::from_native(base + sf_offset);
            let mut gate: f32 = 0.0f32;
            let mut value: f32 = 0.0f32;
            for d in 0..width as usize {
                let code = x.offset(token * width as usize + d).load();
                let xv = rublas::tensor_float_quant::fp8_decode(u32::cast_from(code), false)
                    * scales
                        .offset(token * (width as usize / 128) + d / 128)
                        .load();
                gate = fma(
                    xv,
                    w[(e as usize * inner as usize * 2 + col) * width as usize + d],
                    gate,
                );
                value = fma(
                    xv,
                    w[(e as usize * inner as usize * 2 + inner as usize + col) * width as usize
                        + d],
                    value,
                );
            }
            gate = f32::min(gate, limit);
            value = f32::min(f32::max(value, -limit), limit);
            result = gate / (1.0 + (-gate).exp()) * value;
        }
        hidden[i] = result;
    }
}
#[ruda(launch)]
pub fn second(
    addresses: &Array<u64>,
    w: &Array<f32>,
    hidden: &Array<f32>,
    n: u32,
    rows: u32,
    width: u32,
    k: u32,
    inner: u32,
    experts: u32,
    rank: u32,
    capacity: u32,
    epoch: u32,
    ids_offset: u64,
    weights_offset: u64,
    out_offset: u64,
    control: u64,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        let col = i % width as usize;
        let token = i / width as usize % rows as usize;
        let origin = i / (width as usize * rows as usize);
        let base = addresses[origin];
        let ids = ValuePtr::<i32, GlobalSpace>::from_native(base + ids_offset);
        let weights = ValuePtr::<f32, GlobalSpace>::from_native(base + weights_offset);
        let mut sum: f32 = 0.0f32;
        for r in 0..k as usize {
            let route = token * k as usize + r;
            let live = ValuePtr::<Atomic<u32>, GlobalSpace>::from_native(base + control)
                .offset(6)
                .atomic_load();
            let mut e: i32 = -1i32;
            if token < live as usize {
                e = ids.offset(route).load() - rank as i32 * experts as i32;
            }
            if e >= 0 && e < experts as i32 {
                let mut value: f32 = 0.0f32;
                for d in 0..inner as usize {
                    value = fma(
                        hidden[((origin * rows as usize + token) * k as usize + r)
                            * inner as usize
                            + d],
                        w[(e as usize * width as usize + col) * inner as usize + d],
                        value,
                    );
                }
                sum = fma(value, weights.offset(route).load(), sum);
            }
        }
        ValuePtr::<f32, GlobalSpace, ReadWrite>::from_native(base + out_offset)
            .offset((rank as usize * capacity as usize + token) * width as usize + col)
            .store(sum);
        fence_system();
        let flags = ValuePtr::<Atomic<u32>, GlobalSpace, ReadWrite>::from_native(
            addresses[rank as usize] + control,
        );
        let old = flags.offset(2).fetch_add(1);
        if old == n - 1 {
            fence_system();
            flags.offset(3).atomic_store(epoch);
        }
    }
}
#[ruda(launch)]
pub fn combine<O: Float>(
    addresses: &Array<u64>,
    out: &mut Array<O>,
    n: u32,
    width: u32,
    world: u32,
    rank: u32,
    capacity: u32,
    epoch: u32,
    out_offset: u64,
    control: u64,
    #[define(O)] _dtype: StorageType,
) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        for peer in 0..world as usize {
            await_epoch(addresses[peer], control, 3, epoch);
        }
        let values =
            ValuePtr::<f32, GlobalSpace>::from_native(addresses[rank as usize] + out_offset);
        let token = i / width as usize;
        let col = i % width as usize;
        let mut sum: f32 = 0.0f32;
        for peer in 0..world as usize {
            sum += values
                .offset((peer * capacity as usize + token) * width as usize + col)
                .load();
        }
        out[i] = O::cast_from(sum);
        fence_system();
        let flags = ValuePtr::<Atomic<u32>, GlobalSpace, ReadWrite>::from_native(
            addresses[rank as usize] + control,
        );
        let previous = flags.offset(4).fetch_add(1);
        if previous == n - 1 {
            fence_system();
            flags.offset(5).atomic_store(epoch);
        }
    }
}

#[ruda(launch)]
pub fn initialize(addresses: &Array<u64>, rank: u32, control: u64) {
    let flags = ValuePtr::<Atomic<u32>, GlobalSpace, ReadWrite>::from_native(
        addresses[rank as usize] + control,
    );
    for j in 0usize..8 {
        flags.offset(j).atomic_store(0);
    }
    fence_system();
}

#[ruda(launch)]
pub fn combine_empty(addresses: &Array<u64>, world: u32, rank: u32, epoch: u32, control: u64) {
    for peer in 0..world as usize {
        await_epoch(addresses[peer], control, 3, epoch);
    }
    fence_system();
    ValuePtr::<Atomic<u32>, GlobalSpace, ReadWrite>::from_native(
        addresses[rank as usize] + control,
    )
    .offset(5)
    .atomic_store(epoch);
}
