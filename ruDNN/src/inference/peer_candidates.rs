//! GPU-resident ICP carrier publication and owner-side receive/merge.
//! Tagged words carry a generation with every payload word; legacy rows use a
//! free/ready handshake. Peer allocation owners remain an explicit Rust contract.
use super::{InferenceError, alloc, check};
use ruda_core::tensor::{DType, Shape};
use ruda_kernel::{dsl as kernel_dsl, dsl::prelude::*, tensor::RudaTensor};
#[derive(Clone, Copy, Debug)]
pub struct CandidateLayout {
    pub world: usize,
    pub heads_per_rank: usize,
    pub capacity: usize,
    pub tagged_offset: usize,
    pub bytes: usize,
}
impl CandidateLayout {
    pub fn new(
        world: usize,
        heads_per_rank: usize,
        capacity: usize,
    ) -> Result<Self, InferenceError> {
        if world == 0 || heads_per_rank == 0 || capacity == 0 {
            return Err(InferenceError(
                "candidate communicator geometry must be positive",
            ));
        }
        let rows = 3usize
            .checked_mul(world)
            .and_then(|n| n.checked_mul(heads_per_rank))
            .and_then(|n| n.checked_mul(capacity))
            .ok_or(InferenceError("candidate row capacity overflow"))?;
        let legacy = rows
            .checked_mul(33 * 4)
            .ok_or(InferenceError("candidate legacy size overflow"))?;
        let tagged_offset = legacy
            .checked_add(15)
            .map(|n| n / 16 * 16)
            .ok_or(InferenceError("candidate alignment overflow"))?;
        let bytes = rows
            .checked_mul(32 * 8)
            .and_then(|n| n.checked_add(tagged_offset))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or(InferenceError(
                "candidate communication storage exceeds U32",
            ))?;
        Ok(Self {
            world,
            heads_per_rank,
            capacity,
            tagged_offset,
            bytes,
        })
    }
}
pub struct CandidateComm<R: Runtime> {
    pub addresses: RudaTensor<R>,
    pub generations: RudaTensor<R>,
    pub expected: RudaTensor<R>,
    pub errors: RudaTensor<R>,
    pub layout: CandidateLayout,
    rank: usize,
}
impl<R: Runtime> CandidateComm<R> {
    /// Pointers name IPC-mapped allocations of at least layout.bytes. Mappings
    /// and their exporters must remain live until all kernels finish. Prepare
    /// exactly once per allocation, on every participating rank, before use.
    pub unsafe fn prepare(
        addresses: &RudaTensor<R>,
        layout: CandidateLayout,
        rank: usize,
    ) -> Result<Self, InferenceError> {
        check(addresses, addresses)?;
        if addresses.dtype != DType::U64
            || addresses.meta.shape()[..] != [layout.world]
            || rank >= layout.world
        {
            return Err(InferenceError("candidate address table/rank mismatch"));
        }
        let controls = 3 * layout.world * layout.heads_per_rank * layout.capacity;
        let expected_count = 3 * layout.heads_per_rank * layout.capacity;
        let upload = |n: usize| {
            RudaTensor::new_contiguous(
                addresses.client.clone(),
                addresses.device.clone(),
                Shape::from([n]),
                addresses
                    .client
                    .create_from_slice(bytemuck::cast_slice(&vec![0u32; n])),
                DType::U32,
            )
        };
        let generations = upload(controls);
        let expected = upload(expected_count);
        let errors = upload(1);
        let n = layout.bytes / 4;
        let dim = RudaDim::new(addresses.client.properties(), n);
        let count = kernel_dsl::calculate_ruda_count_elemwise(&addresses.client, n, dim);
        initialize::launch::<R>(
            &addresses.client,
            count,
            dim,
            addresses.clone().into_array_arg(),
            rank as u32,
            n as u32,
        );
        Ok(Self {
            addresses: addresses.clone(),
            generations,
            expected,
            errors,
            layout,
            rank,
        })
    }
    /// Carrier [global_heads,T,33]. Every row is published, including padding.
    /// Tagged reuse needs at least two sparse layers between uses of a slot.
    pub unsafe fn publish(
        &self,
        carrier: &RudaTensor<R>,
        slot: usize,
        tagged: bool,
        spin_limit: u32,
    ) -> Result<(), InferenceError> {
        check(carrier, &self.addresses)?;
        let l = self.layout;
        if slot >= 3
            || carrier.dtype != DType::I32
            || carrier.meta.num_dims() != 3
            || carrier.meta.shape()[0] != l.world * l.heads_per_rank
            || carrier.meta.shape()[1] > l.capacity
            || carrier.meta.shape()[2] != 33
        {
            return Err(InferenceError(
                "candidate publication carrier/slot mismatch",
            ));
        }
        let tokens = carrier.meta.shape()[1];
        let n = tokens * l.world * l.heads_per_rank;
        if n > 0 {
            let dim = RudaDim::new(carrier.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&carrier.client, n, dim);
            publish::launch::<R>(
                &carrier.client,
                count,
                dim,
                carrier.clone().into_array_arg(),
                self.addresses.clone().into_array_arg(),
                self.generations.clone().into_array_arg(),
                self.errors.clone().into_array_arg(),
                tokens as u32,
                l.world as u32,
                l.heads_per_rank as u32,
                l.capacity as u32,
                self.rank as u32,
                slot as u32,
                l.tagged_offset as u64,
                spin_limit,
                tagged,
            );
        }
        Ok(())
    }
    /// Snapshot all source carriers into an owned device tensor. Consuming the
    /// legacy ready flag releases that row only after its payload has been read.
    pub unsafe fn receive(
        &self,
        tokens: usize,
        slot: usize,
        tagged: bool,
        spin_limit: u32,
    ) -> Result<RudaTensor<R>, InferenceError> {
        let out = alloc(
            &self.addresses,
            [self.layout.world, self.layout.heads_per_rank, tokens, 33],
            DType::I32,
        );
        unsafe {
            self.receive_into(tokens, slot, tagged, spin_limit, &out)?;
        }
        Ok(out)
    }
    pub unsafe fn receive_into(
        &self,
        tokens: usize,
        slot: usize,
        tagged: bool,
        spin_limit: u32,
        out: &RudaTensor<R>,
    ) -> Result<(), InferenceError> {
        let l = self.layout;
        if tokens > l.capacity || slot >= 3 {
            return Err(InferenceError("candidate receive capacity/slot mismatch"));
        }
        check(out, &self.addresses)?;
        if out.dtype != DType::I32
            || out.meta.shape()[..] != [l.world, l.heads_per_rank, tokens, 33]
        {
            return Err(InferenceError(
                "retained carrier snapshot shape/dtype mismatch",
            ));
        }
        let n = tokens * l.heads_per_rank;
        if n > 0 {
            let dim = RudaDim::new(self.addresses.client.properties(), n);
            let count = kernel_dsl::calculate_ruda_count_elemwise(&self.addresses.client, n, dim);
            receive::launch::<R>(
                &self.addresses.client,
                count,
                dim,
                self.addresses.clone().into_array_arg(),
                self.expected.clone().into_array_arg(),
                self.errors.clone().into_array_arg(),
                out.clone().into_array_arg(),
                tokens as u32,
                l.world as u32,
                l.heads_per_rank as u32,
                l.capacity as u32,
                self.rank as u32,
                slot as u32,
                l.tagged_offset as u64,
                spin_limit,
                tagged,
            );
        }
        Ok(())
    }
}
#[ruda(launch)]
pub fn initialize(addresses: &Array<u64>, rank: u32, n: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < n as usize {
        ValuePtr::<u32, GlobalSpace, ReadWrite>::from_native(addresses[rank as usize])
            .offset(i)
            .store(0);
    }
}
#[ruda(launch)]
pub fn publish(
    carrier: &Array<i32>,
    addresses: &Array<u64>,
    generations: &mut Array<Atomic<u32>>,
    errors: &mut Array<Atomic<u32>>,
    tokens: u32,
    world: u32,
    hpr: u32,
    capacity: u32,
    rank: u32,
    slot: u32,
    tagged_offset: u64,
    spin_limit: u32,
    #[comptime] tagged: bool,
) {
    let i = ABSOLUTE_POS as usize;
    let heads = world as usize * hpr as usize;
    if i < tokens as usize * heads {
        let token = i % tokens as usize;
        let head = i / tokens as usize;
        let peer = head / hpr as usize;
        let local_head = head % hpr as usize;
        let row = (((slot as usize * world as usize + rank as usize) * hpr as usize + local_head)
            * capacity as usize)
            + token;
        let input = (head * tokens as usize + token) * 33;
        let base = addresses[peer];
        if comptime!(tagged) {
            let generation = generations
                [(slot as usize * heads + head) * capacity as usize + token]
                .fetch_add(1)
                + 1;
            let output =
                ValuePtr::<Atomic<u64>, GlobalSpace, ReadWrite>::from_native(base + tagged_offset)
                    .offset(row * 32);
            for j in 0usize..32 {
                let word = carrier[input + j] as u32 as u64;
                output
                    .offset(j)
                    .atomic_store(((generation as u64) << 32) | word);
            }
            fence_system();
        } else {
            let output =
                ValuePtr::<i32, GlobalSpace, ReadWrite>::from_native(base).offset(row * 33);
            let flag = ValuePtr::<Atomic<u32>, GlobalSpace, ReadWrite>::from_native(base)
                .offset(row * 33 + 32);
            let mut spins: u32 = 0u32;
            let mut free = flag.atomic_load() == 0;
            while !free && spins < spin_limit {
                spins += 1;
                free = flag.atomic_load() == 0;
            }
            if free {
                for j in 0usize..32 {
                    output.offset(j).store(carrier[input + j]);
                }
                fence_system();
                flag.atomic_store(1);
            } else {
                errors[0].fetch_add(1);
            }
        }
    }
}
#[ruda(launch)]
pub fn receive(
    addresses: &Array<u64>,
    expected: &mut Array<u32>,
    errors: &mut Array<Atomic<u32>>,
    out: &mut Array<i32>,
    tokens: u32,
    world: u32,
    hpr: u32,
    capacity: u32,
    rank: u32,
    slot: u32,
    tagged_offset: u64,
    spin_limit: u32,
    #[comptime] tagged: bool,
) {
    let i = ABSOLUTE_POS as usize;
    if i < tokens as usize * hpr as usize {
        let token = i % tokens as usize;
        let head = i / tokens as usize;
        let base = addresses[rank as usize];
        let expected_index = (slot as usize * hpr as usize + head) * capacity as usize + token;
        let generation = expected[expected_index] + 1;
        let mut all_ready = true;
        for peer in 0..world as usize {
            let row = (((slot as usize * world as usize + peer) * hpr as usize + head)
                * capacity as usize)
                + token;
            let dst = ((peer * hpr as usize + head) * tokens as usize + token) * 33;
            let mut ready = true;
            if comptime!(tagged) {
                let ptr = ValuePtr::<Atomic<u64>, GlobalSpace>::from_native(base + tagged_offset)
                    .offset(row * 32);
                for j in 0usize..32 {
                    let mut word = ptr.offset(j).atomic_load();
                    let mut spins: u32 = 0u32;
                    while word >> 32 != generation as u64 && spins < spin_limit {
                        spins += 1;
                        word = ptr.offset(j).atomic_load();
                    }
                    if word >> 32 != generation as u64 {
                        ready = false;
                    }
                    out[dst + j] = word as u32 as i32;
                }
                fence_system();
            } else {
                let flag = ValuePtr::<Atomic<u32>, GlobalSpace, ReadWrite>::from_native(base)
                    .offset(row * 33 + 32);
                let mut spins: u32 = 0u32;
                ready = flag.atomic_load() != 0;
                while !ready && spins < spin_limit {
                    spins += 1;
                    ready = flag.atomic_load() != 0;
                }
                if ready {
                    fence_system();
                    let ptr = ValuePtr::<i32, GlobalSpace>::from_native(base).offset(row * 33);
                    for j in 0usize..32 {
                        out[dst + j] = ptr.offset(j).load();
                    }
                    fence_system();
                    flag.atomic_store(0);
                }
            }
            if !ready {
                errors[0].fetch_add(1);
                all_ready = false;
                for j in 0usize..16 {
                    out[dst + j] = i32::reinterpret(f32::NEG_INFINITY);
                    out[dst + 16 + j] = -1;
                }
            }
            let mut flag: i32 = 0i32;
            if ready {
                flag = 1;
            }
            out[dst + 32] = flag;
        }
        if comptime!(tagged) {
            if all_ready {
                expected[expected_index] = generation;
            }
        }
    }
}
