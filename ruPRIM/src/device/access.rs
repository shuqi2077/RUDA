use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::{prelude::*, calculate_ruda_count_elemwise};
use ruda_kernel::library::tensor::layout::linear::LinearView;
use crate::collective::record::{RudaRead, RudaReadExpand, RudaWrite, RudaWriteExpand};
use crate::collective::record::{RudaRecord, RudaAddress, RudaAddressExpand, RudaReference};
use super::transform::{RudaUnaryOp, RudaUnaryOpExpand, RudaGenerator, RudaGeneratorExpand};
use super::select::{RudaPredicate, RudaPredicateExpand};

type LinearViewExpand<T, IO = ReadOnly> = ruda_kernel::library::tensor::ViewExpand<T, usize, IO>;

#[ruda]
impl<T: Numeric> RudaRead<T> for LinearView<T> {
    fn read(&self, index: usize) -> T { self[index] }
}
#[ruda]
impl<T: Numeric> RudaRead<T> for LinearView<T, ReadWrite> {
    fn read(&self, index: usize) -> T { self[index] }
}
#[ruda]
impl<T: Numeric> RudaWrite<T> for LinearView<T, ReadWrite> {
    fn write(&mut self, index: usize, value: T) { self[index] = value; }
}

#[ruda]
impl<T: Numeric + RudaRecord> RudaAddress<T> for LinearView<T, ReadWrite> {
    fn reference(&self, index: usize) -> RudaReference<T> {
        let element = self.slice(index, 1);
        RudaReference::<T>::new(native_address(&element.to_linear_slice(), 0))
    }
}

#[ruda]
pub trait RudaVisit<T: RudaType + 'static>: RudaType {
    fn visit(&self, value: T, index: usize);
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn transform_kernel<T: RudaType + 'static, U: RudaType + 'static, I: RudaRead<T> + LaunchArg,
    W: RudaWrite<U> + LaunchArg, O: RudaUnaryOp<T, U> + LaunchArg>(
    input: &I, output: &mut W, op: &O, count: usize,
) {
    if ABSOLUTE_POS < count { output.write(ABSOLUTE_POS, op.apply(input.read(ABSOLUTE_POS))); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn transform_if_kernel<T: RudaType + 'static, U: RudaType + 'static, S: RudaType + 'static, I: RudaRead<T> + LaunchArg,
    W: RudaWrite<U> + LaunchArg, J: RudaRead<S> + LaunchArg,
    O: RudaUnaryOp<T, U> + LaunchArg, P: RudaPredicate<S> + LaunchArg>(
    input: &I, stencil: &J, output: &mut W, op: &O, predicate: &P, count: usize,
) {
    let index = ABSOLUTE_POS;
    if index < count {
        if predicate.test(stencil.read(index)) { output.write(index, op.apply(input.read(index))); }
    }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn visit_kernel<T: RudaType + 'static, I: RudaRead<T> + LaunchArg, O: RudaVisit<T> + LaunchArg>(input: &I, op: &O, count: usize) {
    if ABSOLUTE_POS < count { op.visit(input.read(ABSOLUTE_POS), ABSOLUTE_POS); }
}

#[ruda(launch_unchecked, explicit_define, address_type = "dynamic")]
fn generate_kernel<T: RudaType + 'static, W: RudaWrite<T> + LaunchArg, G: RudaGenerator<T> + LaunchArg>(
    output: &mut W, generator: &mut G, count: usize,
) {
    if ABSOLUTE_POS < count { output.write(ABSOLUTE_POS, generator.generate()); }
}

/// Random-access transform. Recursive RudaZip inputs carry any number of
/// heterogeneous values. RudaReferences readers instead pass stable references
/// into the original allocations; they never materialize local-copy addresses.
/// Every reader/writer must support count accesses. Exact elementwise aliasing
/// is allowed; accesses from different invocations must not race.
pub fn transform<R, T, U, I, W, O>(
    client: &ComputeClient<R>, count: usize, input: I::RuntimeArg<R>, output: W::RuntimeArg<R>,
    op: O::RuntimeArg<R>, address_type: AddressType,
) where R: Runtime, T: RudaType + 'static, U: RudaType + 'static, I: RudaRead<T> + LaunchArg,
    W: RudaWrite<U> + LaunchArg, O: RudaUnaryOp<T, U> + LaunchArg,
{
    if count == 0 { return; }
    let dim = RudaDim::new(client.properties(), count);
    let grid = calculate_ruda_count_elemwise(client, count, dim);
    unsafe {
        transform_kernel::launch_unchecked::<T, U, I, W, O, R>(client, grid, dim,
            address_type.max(AddressType::from_len(count)), input, output, op, count);
    }
}

/// Conditional transform with an independently typed stencil. Unselected
/// output elements are untouched; the transform is not invoked for them.
pub fn transform_if<R, T, U, S, I, W, J, O, P>(
    client: &ComputeClient<R>, count: usize, input: I::RuntimeArg<R>, stencil: J::RuntimeArg<R>,
    output: W::RuntimeArg<R>, op: O::RuntimeArg<R>, predicate: P::RuntimeArg<R>, address_type: AddressType,
) where R: Runtime, T: RudaType + 'static, U: RudaType + 'static, S: RudaType + 'static,
    I: RudaRead<T> + LaunchArg, W: RudaWrite<U> + LaunchArg, J: RudaRead<S> + LaunchArg,
    O: RudaUnaryOp<T, U> + LaunchArg, P: RudaPredicate<S> + LaunchArg,
{
    if count == 0 { return; }
    let dim = RudaDim::new(client.properties(), count);
    let grid = calculate_ruda_count_elemwise(client, count, dim);
    unsafe {
        transform_if_kernel::launch_unchecked::<T, U, S, I, W, J, O, P, R>(client, grid, dim,
            address_type.max(AddressType::from_len(count)), input, stencil, output, op, predicate, count);
    }
}

/// ForEach/ForEachN through value or stable-reference readers. A value reader
/// supplies copy semantics; a reference reader allows original-storage edits.
pub fn for_each<R, T, I, O>(
    client: &ComputeClient<R>, count: usize, input: I::RuntimeArg<R>, op: O::RuntimeArg<R>, address_type: AddressType,
) where R: Runtime, T: RudaType + 'static, I: RudaRead<T> + LaunchArg, O: RudaVisit<T> + LaunchArg,
{
    if count == 0 { return; }
    let dim = RudaDim::new(client.properties(), count);
    let grid = calculate_ruda_count_elemwise(client, count, dim);
    unsafe {
        visit_kernel::launch_unchecked::<T, I, O, R>(client, grid, dim,
            address_type.max(AddressType::from_len(count)), input, op, count);
    }
}

pub fn generate<R, T, W, G>(
    client: &ComputeClient<R>, count: usize, output: W::RuntimeArg<R>, generator: G::RuntimeArg<R>, address_type: AddressType,
) where R: Runtime, T: RudaType + 'static, W: RudaWrite<T> + LaunchArg, G: RudaGenerator<T> + LaunchArg,
{
    if count == 0 { return; }
    let dim = RudaDim::new(client.properties(), count);
    let grid = calculate_ruda_count_elemwise(client, count, dim);
    unsafe {
        generate_kernel::launch_unchecked::<T, W, G, R>(client, grid, dim,
            address_type.max(AddressType::from_len(count)), output, generator, count);
    }
}
