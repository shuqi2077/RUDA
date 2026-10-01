//! Real NPU validation: no mock library, no skip-on-missing-device behavior.
//! cargo run -p ruda-driver-cann --release --example deepgemm_validate -- ARTIFACT_DIR
use ruda_driver_cann::{CannDevice,CannError,tensor::{CannSession,CannTensor,DType,deepgemm::{DeepGemm,Transpose,GroupEnds}}};
use std::rc::Rc;
fn bf16(x:f32)->u16 { let bits=x.to_bits(); ((bits.wrapping_add(0x7fff+((bits>>16)&1)))>>16) as u16 }
fn host(n:usize,seed:usize)->Vec<f32>{(0..n).map(|i| ((i*7+seed)%17) as f32/8.0-1.0).collect()}
fn upload(s:&Rc<CannSession>,shape:&[i64],v:&[f32])->Result<CannTensor,CannError>{
 let bytes:Vec<u8>=v.iter().flat_map(|&x|bf16(x).to_ne_bytes()).collect();s.from_bytes(shape,DType::BF16,&bytes)
}
fn read(t:&CannTensor)->Result<Vec<f32>,CannError>{
 let b=t.to_bytes()?; Ok(if t.layout().dtype()==DType::F32 {b.chunks_exact(4).map(|x|f32::from_ne_bytes(x.try_into().unwrap())).collect()}
 else {b.chunks_exact(2).map(|x|f32::from_bits((u16::from_ne_bytes(x.try_into().unwrap()) as u32)<<16)).collect()})
}
fn check(actual:&[f32],expected:&[f32],label:&str)->Result<(),Box<dyn std::error::Error>>{
 if actual.len()!=expected.len(){return Err(format!("{label}: length mismatch").into());}
 for (i,(&a,&b)) in actual.iter().zip(expected).enumerate(){if !a.is_finite() || (a-b).abs()>0.05+0.01*b.abs(){return Err(format!("{label}: index {i}: actual={a}, expected={b}").into());}}
 Ok(())
}
fn reference(a:&[f32],b:&[f32],m:usize,n:usize,k:usize,ta:bool,tb:bool)->Vec<f32>{
 (0..m*n).map(|ij|{let i=ij/n;let j=ij%n;(0..k).map(|p|a[if ta{p*m+i}else{i*k+p}]*b[if tb{j*k+p}else{p*n+j}]).sum()}).collect()
}
fn main()->Result<(),Box<dyn std::error::Error>>{
 let root=std::env::args().nth(1).ok_or("usage: deepgemm_validate ARTIFACT_DIR")?;
 let acl=std::env::var("RUDA_CANN_LIBRARY").unwrap_or_else(|_|"libascendcl.so".into());
 let op=std::env::var("RUDA_CANN_OPAPI").unwrap_or_else(|_|"libopapi.so".into());
 // SAFETY: this is an isolated standalone test process with trusted SDK/artifacts.
 let s=unsafe{CannSession::open_exclusive(CannDevice::new(0)?,acl,op)?};
 let native=unsafe{DeepGemm::load(&s,root)?};
 let mut cases=0;
 for batch in [1usize,3] {for ta in [false,true]{for tb in [false,true]{for dt in [DType::BF16,DType::F32]{
     let(m,n,k)=(32usize,64usize,48usize);let a=host(batch*m*k,1);let b=host(batch*k*n,4);
     let mut ash=if ta{vec![k as i64,m as i64]}else{vec![m as i64,k as i64]};let mut bsh=if tb{vec![n as i64,k as i64]}else{vec![k as i64,n as i64]};
     if batch!=1{ash.insert(0,batch as i64);bsh.insert(0,batch as i64);}
     let at=upload(&s,&ash,&a)?;let bt=upload(&s,&bsh,&b)?;
     let tr=|x|if x{Transpose::Yes}else{Transpose::No};
     let mut output=native.gemm(&at,&bt,tr(ta),tr(tb),dt)?;
     let expected:Vec<f32>=(0..batch).flat_map(|i|reference(&a[i*m*k..(i+1)*m*k],&b[i*k*n..(i+1)*k*n],m,n,k,ta,tb)).collect();
     check(&read(&output)?,&expected,"GEMM")?;
     let modules=native.stats().modules_loaded;
     native.gemm_into(&at,&bt,tr(ta),tr(tb),&mut output)?;
     if modules!=native.stats().modules_loaded{return Err("module cache did not reuse the binary".into());}
     check(&read(&output)?,&expected,"GEMM output reuse")?;cases+=1;
 }}}}
 // M-grouped aligned physical rows, including an empty expert.
 for dt in [DType::BF16,DType::F32]{
   let(m,n,k)=(512usize,64usize,48usize);let ends=[256,256,512];let a=host(m*k,2);let b=host(3*n*k,5);
   let at=upload(&s,&[m as i64,k as i64],&a)?;let bt=upload(&s,&[3,n as i64,k as i64],&b)?;
   let plan=native.upload_group_ends(GroupEnds::new(&ends)?)?;let output=native.grouped_nt(&at,&bt,&plan,dt)?;
   let mut expected=reference(&a[..256*k],&b[..n*k],256,n,k,false,true);expected.extend(reference(&a[256*k..],&b[2*n*k..],256,n,k,false,true));
   check(&read(&output)?,&expected,"grouped with empty expert")?;cases+=1;
 }
 let(m,n,k)=(32usize,64usize,48usize);let x=host(m*k,3);let w=host(n*k,2);let dy=host(m*n,7);
 let xt=upload(&s,&[m as i64,k as i64],&x)?;let wt=upload(&s,&[n as i64,k as i64],&w)?;let dyt=upload(&s,&[m as i64,n as i64],&dy)?;
 let(dx,dw)=native.linear_nt_backward(&xt,&wt,&dyt)?;
 check(&read(&dx)?,&reference(&dy,&w,m,k,n,false,false),"linear dX")?;
 check(&read(&dw)?,&reference(&dy,&x,n,k,m,true,false),"linear dW")?;cases+=2;
 if cases!=20 || native.stats().launches!=36 {return Err("validation count mismatch".into());}
 println!("RUDA_ASCEND_REAL_DEVICE_OK cases={} launches={} modules={}",cases,native.stats().launches,native.stats().modules_loaded);
 Ok(())
}
