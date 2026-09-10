use half::{bf16, f16};
use ruda_driver_cuda::CudaRuntime;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch, address_type = "dynamic")]
fn wmma<F: Float + RudaElement>(a: &Array<F>, b: &Array<F>, output: &mut Array<f32>,
    #[comptime] m: usize, #[comptime] n: usize) {
    let a = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::A, m, n, 16usize,
        cmma::MatrixLayout::RowMajor, &a.to_slice(), 16);
    let b = cmma::Matrix::<F>::from_slice(cmma::MatrixIdent::B, m, n, 16usize,
        cmma::MatrixLayout::ColMajor, &b.to_slice(), 16);
    let c = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, m, n, 16usize,
        cmma::MatrixLayout::Undefined, 0.25f32);
    cmma::execute::<F,F,f32,f32>(&a, &b, &c, &c);
    cmma::store(&mut output.to_slice_mut(), &c, n as u32, cmma::MatrixLayout::RowMajor);
}

trait MatrixTest: Float + RudaElement { fn from_float(x: f32) -> Self; }
impl MatrixTest for f16 { fn from_float(x: f32) -> Self { Self::from_f32(x) } }
impl MatrixTest for bf16 { fn from_float(x: f32) -> Self { Self::from_f32(x) } }

fn check<F: MatrixTest>(client: &ComputeClient<CudaRuntime>, backend: &str, m:usize, n:usize, k:usize, manual:bool, address:AddressType) {
    let a:Vec<f32>=(0..m*k).map(|i| ((i*7%19) as f32-9.0)*0.25).collect();
    let b:Vec<f32>=(0..k*n).map(|i| ((i*11%23) as f32-11.0)*0.25).collect();
    let a_half:Vec<F>=a.iter().copied().map(F::from_float).collect();
    let b_half:Vec<F>=if manual { b.iter().copied().map(F::from_float).collect() }
        else { (0..n).flat_map(|col| (0..k).map(move |row| (row,col))).map(|(row,col)| F::from_float(b[row*n+col])).collect() };
    let lhs=client.create_from_slice(F::as_bytes(&a_half));
    let rhs=client.create_from_slice(F::as_bytes(&b_half));
    let out=client.create_from_slice(f32::as_bytes(&vec![247.0;m*n+16]));
    // SAFETY: A complete warp operates on aligned, fully allocated matrices and output.
    unsafe {
        if manual {
            let c=client.create_from_slice(f32::as_bytes(&vec![0.25;m*n]));
            ruda_kernel::dsl::runtime_tests::cmma::kernel_manual::launch::<F,F,f32,CudaRuntime>(
                client,RudaCount::Static(1,1,1),RudaDim::new_1d(32),
                TensorArg::from_raw_parts(lhs,vec![k,1].into(),vec![m,k].into()),
                TensorArg::from_raw_parts(rhs,vec![n,1].into(),vec![k,n].into()),
                TensorArg::from_raw_parts(c,vec![n,1].into(),vec![m,n].into()),
                TensorArg::from_raw_parts(out.clone(),vec![n,1].into(),vec![m,n].into()),m,n,k);
        } else {
            wmma::launch::<F,CudaRuntime>(client,RudaCount::Static(1,1,1),RudaDim::new_1d(32),address,
                ArrayArg::from_raw_parts(lhs,m*k),ArrayArg::from_raw_parts(rhs,k*n),ArrayArg::from_raw_parts(out.clone(),m*n),m,n);
        }
    }
    let bytes=client.read_one(out).unwrap();
    let actual=f32::from_bytes(&bytes);
    for row in 0..m { for col in 0..n {
        let expected=0.25+(0..k).map(|inner| a[row*k+inner]*b[inner*n+col]).sum::<f32>();
        assert_eq!(actual[row*n+col],expected,"{backend} {} manual={manual} {m}x{n}x{k} ({row},{col})",core::any::type_name::<F>());
    }}
    assert!(actual[m*n..].iter().all(|&x| x==247.0));
    println!("PASS {backend} matrix {} manual={manual} {m}x{n}x{k} {address:?}",core::any::type_name::<F>());
}

pub fn run(client:&ComputeClient<CudaRuntime>,backend:&str) {
    for address in [AddressType::U32,AddressType::U64] {
        for (m,n) in [(16,16),(8,32),(32,8)] {
            check::<bf16>(client,backend,m,n,16,false,address);
            check::<f16>(client,backend,m,n,16,false,address);
        }
    }
    for k in [8,16] {
        check::<bf16>(client,backend,16,8,k,true,AddressType::U32);
        check::<f16>(client,backend,16,8,k,true,AddressType::U32);
    }
}
