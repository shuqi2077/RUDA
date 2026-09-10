//! Actual AMD adapter example. Requires a working HIP installation and model.
use rullm::backends::{Amd, AmdDevice};
use rullm::{GreedyGenerationConfig, generate_greedy_packed_with_mode, load_huggingface_qwen2_pipeline};
use rullm::runtime::KernelMode;
use ruda_tensor::api::backend::Backend;
use std::{error::Error,io};
type B = Amd<f32, i32>;
fn main() -> Result<(),Box<dyn Error>> {
    let mut args=std::env::args().skip(1);
    let directory=args.next().ok_or_else(||io::Error::other("usage: qwen2_amd <model-directory> <prompt> [device-index]"))?;
    let prompt=args.next().ok_or_else(||io::Error::other("prompt required"))?;
    let index=args.next().map(|n|n.parse()).transpose()?.unwrap_or(0usize);
    if args.next().is_some() {return Err(io::Error::other("too many arguments").into());}
    u16::try_from(index).map_err(|_|io::Error::other("device index exceeds Ruda device identity range"))?;
    let device=AmdDevice::new(index);
    let pipeline=load_huggingface_qwen2_pipeline::<B>(&directory,&device)?;
    let tokens=pipeline.encode(&prompt,false)?;
    let model=pipeline.loaded.model.into_packed_inference();
    let result=generate_greedy_packed_with_mode(&model,&pipeline.loaded.config,&tokens,
        &GreedyGenerationConfig {max_new_tokens:32,eos_token_ids:pipeline.loaded.default_eos_token_ids},
        &device,KernelMode::Automatic);
    B::sync(&device)?;
    let output=result?;
    let ids=output.generated_token_ids.into_iter().map(u32::try_from).collect::<Result<Vec<_>,_>>()?;
    println!("{}",pipeline.tokenizer.decode(&ids,true).map_err(io::Error::other)?);
    Ok(())
}
