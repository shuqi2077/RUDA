use core::marker::PhantomData;
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

type ArrayExpand<T> = NativeExpand<Array<T>>;

/// A device value with an explicit byte layout. Composite implementations
/// lower field operations to scalar IR, retaining the record's native layout.
#[ruda]
pub trait RudaRecord: Copy + RudaType<ExpandType: Assign> {
    const SIZE: usize;
    const ALIGN: usize;
    fn load(address: u64) -> Self;
    fn store(address: u64, value: Self);
    fn shuffle(value: Self, lane: u32) -> Self;
}

macro_rules! scalar_record {
    ($($ty:ty),* $(,)?) => {$(
        #[ruda]
        impl RudaRecord for $ty {
            const SIZE: usize = core::mem::size_of::<Self>();
            const ALIGN: usize = core::mem::align_of::<Self>();
            fn load(address: u64) -> Self { native_load::<Self>(address) }
            fn store(address: u64, value: Self) { native_store(address, value); }
            fn shuffle(value: Self, lane: u32) -> Self { plane_shuffle(value, lane) }
        }
    )*};
}
scalar_record!(bool, u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, half::f16, half::bf16);

/// Define a C-layout record, including nested records, for primitive operators.
#[macro_export]
macro_rules! ruda_record {
    ($(#[$meta:meta])* $vis:vis struct $name:ident { $($field_vis:vis $field:ident : $ty:ty),* $(,)? }) => {
        $(#[$meta])*
        #[repr(C)]
        #[derive(Clone, Copy, ruda_kernel::dsl::RudaType, ruda_kernel::dsl::prelude::RudaTypeMut, ruda_kernel::dsl::RudaLaunch)]
        $vis struct $name { $($field_vis $field: $ty),* }
        #[ruda_kernel::dsl::ruda]
        impl $crate::collective::record::RudaRecord for $name {
            const SIZE: usize = core::mem::size_of::<Self>();
            const ALIGN: usize = core::mem::align_of::<Self>();
            fn load(address: u64) -> Self {
                $name { $($field: <$ty as $crate::collective::record::RudaRecord>::load(
                    address + ruda_kernel::dsl::comptime![core::mem::offset_of!(Self, $field) as u64])),* }
            }
            fn store(address: u64, value: Self) {
                $(<$ty as $crate::collective::record::RudaRecord>::store(
                    address + ruda_kernel::dsl::comptime![core::mem::offset_of!(Self, $field) as u64], value.$field);)*
            }
            fn shuffle(value: Self, lane: u32) -> Self {
                $name { $($field: <$ty as $crate::collective::record::RudaRecord>::shuffle(value.$field, lane)),* }
            }
        }
    };
}

#[ruda]
pub trait RudaRead<T: RudaType>: RudaType { fn read(&self, index: usize) -> T; }

#[ruda]
pub trait RudaWrite<T: RudaType>: RudaRead<T> { fn write(&mut self, index: usize, value: T); }

#[ruda]
impl<T: RudaPrimitive> RudaRead<T> for Array<T> {
    fn read(&self, index: usize) -> T { self[index] }
}
#[ruda]
impl<T: RudaPrimitive> RudaWrite<T> for Array<T> {
    fn write(&mut self, index: usize, value: T) { self[index] = value; }
}
#[ruda]
impl<T: RudaPrimitive> RudaRead<T> for SharedMemory<T> {
    fn read(&self, index: usize) -> T { self[index] }
}
#[ruda]
impl<T: RudaPrimitive> RudaWrite<T> for SharedMemory<T> {
    fn write(&mut self, index: usize, value: T) { self[index] = value; }
}

#[derive(RudaType)]
pub struct RudaRecordArray<T: RudaRecord> {
    bytes: Array<u8>,
    #[ruda(comptime)]
    marker: PhantomData<T>,
}

#[ruda]
impl<T: RudaRecord> RudaRecordArray<T> {
    pub fn new(#[comptime] length: usize) -> Self {
        let bytes = Array::<u8>::new(comptime![(length * T::SIZE + T::ALIGN).max(1)]);
        RudaRecordArray::<T> { bytes, marker: PhantomData }
    }
    pub fn address(&self, index: usize) -> u64 {
        let base = native_address(&self.bytes.to_slice(), 0);
        let align = comptime![T::ALIGN as u64];
        (base + align - 1) / align * align + index as u64 * comptime![T::SIZE as u64]
    }
}
#[ruda]
impl<T: RudaRecord> RudaRead<T> for RudaRecordArray<T> {
    fn read(&self, index: usize) -> T { T::load(self.address(index)) }
}
#[ruda]
impl<T: RudaRecord> RudaWrite<T> for RudaRecordArray<T> {
    fn write(&mut self, index: usize, value: T) { T::store(self.address(index), value); }
}

#[derive(RudaType)]
pub struct RudaRecordShared<T: RudaRecord> {
    bytes: SharedMemory<u8>,
    #[ruda(comptime)]
    marker: PhantomData<T>,
}

#[ruda]
impl<T: RudaRecord> RudaRecordShared<T> {
    pub fn new(#[comptime] length: usize) -> Self {
        let bytes = SharedMemory::<u8>::new_aligned(comptime![(length * T::SIZE).max(1)], comptime![T::ALIGN]);
        RudaRecordShared::<T> { bytes, marker: PhantomData }
    }
    pub fn address(&self, index: usize) -> u64 {
        native_address(&self.bytes.to_slice(), index * comptime![T::SIZE])
    }
}
#[ruda]
impl<T: RudaRecord> RudaRead<T> for RudaRecordShared<T> {
    fn read(&self, index: usize) -> T { T::load(self.address(index)) }
}
#[ruda]
impl<T: RudaRecord> RudaWrite<T> for RudaRecordShared<T> {
    fn write(&mut self, index: usize, value: T) { T::store(self.address(index), value); }
}

/// Stable reference into an existing allocation, not a reference to a local copy.
#[derive(Clone, RudaType, RudaTypeMut)]
pub struct RudaReference<T: RudaRecord> {
    pub address: u64,
    #[ruda(comptime)]
    marker: PhantomData<T>,
}

#[ruda]
impl<T: RudaRecord> RudaReference<T> {
    pub fn new(address: u64) -> Self { RudaReference::<T> { address, marker: PhantomData } }
    pub fn read(&self) -> T { T::load(self.address) }
    pub fn write(&self, value: T) { T::store(self.address, value); }
}

/// A device-native strided record range. Addresses and strides are in bytes.
/// The caller retains the allocation and guarantees alignment and bounds for
/// every access, including any references retained by an operation.
#[derive(Clone, Copy, RudaType, RudaLaunch)]
pub struct RudaNativeRecords {
    pub address: u64,
    pub stride: u64,
}

#[ruda]
impl<T: RudaRecord> RudaRead<T> for RudaNativeRecords {
    fn read(&self, index: usize) -> T { T::load(self.address + index as u64 * self.stride) }
}
#[ruda]
impl<T: RudaRecord> RudaWrite<T> for RudaNativeRecords {
    fn write(&mut self, index: usize, value: T) { T::store(self.address + index as u64 * self.stride, value); }
}

#[ruda]
pub trait RudaAddress<T: RudaRecord>: RudaRead<T> {
    fn reference(&self, index: usize) -> RudaReference<T>;
}

#[ruda]
impl<T: RudaRecord> RudaAddress<T> for RudaNativeRecords {
    fn reference(&self, index: usize) -> RudaReference<T> {
        RudaReference::<T>::new(self.address + index as u64 * self.stride)
    }
}

/// Recursive product: nesting has no fixed arity limit.
#[derive(Clone, RudaType)]
pub struct RudaPair<A: RudaType, B: RudaType> {
    pub first: A,
    pub second: B,
}

#[derive(Clone, RudaType, RudaLaunch)]
pub struct RudaZip<I: RudaType, J: RudaType> {
    pub first: I,
    pub second: J,
}

#[ruda]
impl<A: RudaType, B: RudaType, I: RudaRead<A>, J: RudaRead<B>> RudaRead<RudaPair<A, B>> for RudaZip<I, J> {
    fn read(&self, index: usize) -> RudaPair<A, B> {
        RudaPair::<A, B> { first: self.first.read(index), second: self.second.read(index) }
    }
}

#[derive(Clone, RudaType, RudaLaunch)]
pub struct RudaReferences<T: RudaRecord, I: RudaAddress<T>> {
    pub input: I,
    #[ruda(comptime)]
    pub marker: PhantomData<T>,
}

#[ruda]
impl<T: RudaRecord, I: RudaAddress<T>> RudaRead<RudaReference<T>> for RudaReferences<T, I> {
    fn read(&self, index: usize) -> RudaReference<T> { self.input.reference(index) }
}
