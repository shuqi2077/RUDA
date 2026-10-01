//! Real NPU smoke/numerical tests of production common-IR compiler artifacts.
//! Host math is an independent reference only; no device fallback or mocks.
use ruda_driver_cann::{CannDevice,tensor::{CannSession,common_ir::CannProgram}};
use ruda_compiler::ascend::{AscendCompiler,AscendOptions,AscendTarget,programs::{MapProgram,definition}};
use ruda_core::{compiler::Compiler,ir::UIntKind,launch::ExecutionMode};
fn main()->Result<(),Box<dyn std::error::Error>>{
    let root=std::path::PathBuf::from(std::env::args().nth(1).ok_or("usage: common_ir_validate ARTIFACT_ROOT")?);
    let acl=std::env::var("RUDA_CANN_LIBRARY").unwrap_or_else(|_|"libascendcl.so".into());
    let opapi=std::env::var("RUDA_CANN_OPAPI").unwrap_or_else(|_|"libopapi.so".into());
    // SAFETY: standalone isolated process, trusted SDK/artifacts; no torch_npu.
    let session=unsafe{CannSession::open_exclusive(CannDevice::new(0)?,acl,opapi)?};
    let mut cases=0;let mut launches=0;
    for op in [MapProgram::Copy,MapProgram::Add,MapProgram::Mul,MapProgram::Silu,MapProgram::SiluMul,MapProgram::SiluBackward,MapProgram::SiluMulBackward]{for n in [1usize,7,256,257,1025]{
        let options=AscendOptions{target:Some(AscendTarget::Ascend950DT),elements:n as u64,..Default::default()};
        let compiled=AscendCompiler.compile(definition(op),&options,ExecutionMode::Checked,UIntKind::U64.into())?;
        let program=unsafe{CannProgram::load(&session,compiled,root.join(format!("{}-{n}",op.name())))?};
        let x:Vec<f32>=(0..n).map(|i|(i%61)as f32/7.0-4.0).collect();
        let up:Vec<f32>=(0..n).map(|i|(i%19)as f32/9.0-1.0).collect();
        let dy:Vec<f32>=(0..n).map(|i|(i%11)as f32/8.0-0.5).collect();
        let host=[&x,&up,&dy];let device:Vec<_>=host[..op.input_count()].iter().map(|v|session.from_f32(&[n as i64],v)).collect::<Result<_,_>>()?;
        let refs:Vec<_>=device.iter().collect();let mut outputs=program.run(&refs)?;
        let expected:Vec<Vec<f32>>=(0..op.output_count()).map(|j|(0..n).map(|i|{
            let a=x[i];let b=up[i];let g=dy[i];let s=1.0/(1.0+(-a).exp());let h=a*s;let d=s*(1.0+a*(1.0-s));
            match op{MapProgram::Copy=>a,MapProgram::Add=>a+b,MapProgram::Mul=>a*b,MapProgram::Silu=>h,MapProgram::SiluMul=>h*b,MapProgram::SiluBackward=>b*d,MapProgram::SiluMulBackward=>if j==0{g*b*d}else{g*h}}
        }).collect()).collect();
        for repeat in 0..2{
            if repeat==1{let mut refs:Vec<_>=outputs.iter_mut().collect();program.run_into(&device.iter().collect::<Vec<_>>(),&mut refs)?;}
            for(out,want)in outputs.iter().zip(&expected){let actual=out.to_f32()?;for(i,(&a,&b))in actual.iter().zip(want).enumerate(){if !a.is_finite()||(a-b).abs()>5e-4+5e-4*b.abs(){return Err(format!("{} n={n} index={i} repeat={repeat}: actual={a}, reference={b}",op.name()).into());}}}
        }
        if program.stats().launches!=2{return Err("native launch count mismatch".into());}
        launches+=program.stats().launches;cases+=1;
        println!("RUDA_ASCEND_COMMON_IR_CASE op={} elements={n} passed=true",op.name());
    }}
    if cases!=35||launches!=70{return Err("unexpected case count".into());}
    println!("RUDA_ASCEND_COMMON_IR_DEVICE_OK cases={cases} launches={launches}");Ok(())
}
