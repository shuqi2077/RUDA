//! Standalone Rust CLI. No Python/PyTorch/DeepJIT dependencies.
use std::{env,fs,path::PathBuf,process};
use ruda_ascend_kernels::{Spec,emit,kernel,ascend};
fn run()->Result<(),String>{
    let mut out=None;let mut only=Vec::new();let mut args=env::args().skip(1);
    while let Some(a)=args.next(){match a.as_str(){
        "--list"=>{for s in Spec::all(){println!("{}",s.key())}return Ok(())},
        "--out"=>out=Some(PathBuf::from(args.next().ok_or("--out needs a directory")?)),
        "--only"=>only.push(args.next().ok_or("--only needs a key")?),
        _=>return Err(format!("unknown option {a}")),
    }}
    let out=out.ok_or("usage: ruda-ascend-emit --out NEW_DIR [--only KEY] | --list")?;
    if out.exists(){return Err("output directory exists; use a fresh path".into())}
    let all=Spec::all();
    for k in &only{if !all.iter().any(|s|s.key()==*k){return Err(format!("unknown key {k}"))}}
    fs::create_dir_all(&out).map_err(|e|e.to_string())?;
    fs::write(out.join("ruda_kernel_abi.h"),ascend::abi_header()).map_err(|e|e.to_string())?;
    let mut count=0;
    for s in all{if !only.is_empty() && !only.contains(&s.key()){continue}
        let dir=out.join(s.key());fs::create_dir(&dir).map_err(|e|e.to_string())?;
        let program=kernel::build(s)?;
        fs::write(dir.join("kernel.asc"),emit(s)?).map_err(|e|e.to_string())?;
        fs::write(dir.join("kernel.rust-ir.txt"),format!("{program:#?}\n")).map_err(|e|e.to_string())?;
        count+=1;
    }
    fs::write(out.join("SOURCE_ONLY"),"Rust-authored device IR lowered to CCE. Not an executable; NPU validation is required.\n").map_err(|e|e.to_string())?;
    println!("RUDA_RUST_DEVICE_EMIT_OK kernels={count}");Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("{e}");process::exit(1)}}
