use ruda_driver_cuda::CudaRuntime;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch, address_type = "dynamic")]
fn integer_ops<I: Int + RudaElement>(
    input: &Array<I>,
    rhs: &Array<I>,
    shifts: &Array<I>,
    output: &mut Array<I>,
    counts: &mut Array<u32>,
) {
    let i = ABSOLUTE_POS as usize;
    if i < input.len() {
        let n = input.len();
        let a = input[i];
        let b = rhs[i];
        let shift = shifts[i];
        output[i] = a & b;
        output[n + i] = a | b;
        output[n * 2 + i] = a ^ b;
        output[n * 3 + i] = !a;
        output[n * 4 + i] = a.reverse_bits();
        output[n * 5 + i] = a << shift;
        output[n * 6 + i] = a >> shift;
        let choose = select(a != I::new(0), a < b, a == b);
        output[n * 7 + i] = select(choose, a, b);
        counts[i] = u32::cast_from(a.count_ones());
        counts[n + i] = u32::cast_from(a.leading_zeros());
        counts[n * 2 + i] = u32::cast_from(a.trailing_zeros());
        counts[n * 3 + i] = u32::cast_from(a.find_first_set());
    }
}

trait IntegerTest: Int + RudaElement {
    fn fixture(index: usize) -> Self;
    fn shift(index: usize) -> Self;
    fn expected(a: Self, b: Self, shift: Self) -> ([Self; 8], [u32; 4]);
}

macro_rules! integer_test {
    ($ty:ty) => {
        impl IntegerTest for $ty {
            fn fixture(index: usize) -> Self {
                match index % 8 {
                    0 => 0,
                    1 => <$ty>::MAX,
                    2 => <$ty>::MIN,
                    3 => (1_u64 << (index as u32 % <$ty>::BITS)) as $ty,
                    4 => 0xaaaa_aaaa_aaaa_aaaa_u64 as $ty,
                    5 => 0x5555_5555_5555_5555_u64 as $ty,
                    _ => (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) as $ty,
                }
            }
            fn shift(index: usize) -> Self {
                (index as u32 % <$ty>::BITS) as $ty
            }
            fn expected(a: Self, b: Self, shift: Self) -> ([Self; 8], [u32; 4]) {
                let choose = if a != 0 { a < b } else { a == b };
                ([a & b, a | b, a ^ b, !a, a.reverse_bits(), a << shift, a >> shift,
                  if choose { a } else { b }],
                 [a.count_ones(), a.leading_zeros(), a.trailing_zeros(),
                  if a == 0 { 0 } else { a.trailing_zeros() + 1 }])
            }
        }
    };
}
integer_test!(u32);
integer_test!(i32);
integer_test!(u64);
integer_test!(i64);

fn check<I: IntegerTest>(client: &ComputeClient<CudaRuntime>, backend: &str, address: AddressType, count: usize) {
    let input: Vec<I> = (0..count).map(I::fixture).collect();
    let rhs: Vec<I> = (0..count).map(|index| I::fixture(index + 11)).collect();
    let shifts: Vec<I> = (0..count).map(I::shift).collect();
    let a = client.create_from_slice(I::as_bytes(&input));
    let b = client.create_from_slice(I::as_bytes(&rhs));
    let shift = client.create_from_slice(I::as_bytes(&shifts));
    let sentinel = I::new(117);
    let output = client.create_from_slice(I::as_bytes(&vec![sentinel; count * 8 + 16]));
    let counts = client.create_from_slice(u32::as_bytes(&vec![u32::MAX; count * 4 + 16]));
    // SAFETY: Bindings cover the declared element counts. Checked accesses and
    // the in-kernel input length guard protect tail threads and output sentinels.
    unsafe {
        integer_ops::launch::<I, CudaRuntime>(
            client, RudaCount::Static(count.div_ceil(64) as u32, 1, 1), RudaDim::new_1d(64), address,
            ArrayArg::from_raw_parts(a, count),
            ArrayArg::from_raw_parts(b, count),
            ArrayArg::from_raw_parts(shift, count),
            ArrayArg::from_raw_parts(output.clone(), count * 8),
            ArrayArg::from_raw_parts(counts.clone(), count * 4),
        );
    }
    let bytes = client.read_one(output).unwrap();
    let actual = I::from_bytes(&bytes);
    let count_bytes = client.read_one(counts).unwrap();
    let actual_counts = u32::from_bytes(&count_bytes);
    for index in 0..count {
        let (values, counts) = I::expected(input[index], rhs[index], shifts[index]);
        for (column, expected) in values.into_iter().enumerate() {
            assert_eq!(actual[column * count + index], expected,
                "{backend} {} {address:?} column={column} index={index}", std::any::type_name::<I>());
        }
        for (column, expected) in counts.into_iter().enumerate() {
            assert_eq!(actual_counts[column * count + index], expected,
                "{backend} {} {address:?} count column={column} index={index}", std::any::type_name::<I>());
        }
    }
    assert!(actual[count * 8..].iter().all(|&value| value == sentinel));
    assert!(actual_counts[count * 4..].iter().all(|&value| value == u32::MAX));
    println!("PASS {backend} integer bits {} {address:?} count={count}", std::any::type_name::<I>());
}

pub fn run(client: &ComputeClient<CudaRuntime>, backend: &str) {
    for address in [AddressType::U32, AddressType::U64] {
        for count in [1, 63, 64, 65, 257] {
            check::<u32>(client, backend, address, count);
            check::<i32>(client, backend, address, count);
            check::<u64>(client, backend, address, count);
            check::<i64>(client, backend, address, count);
        }
    }
}
