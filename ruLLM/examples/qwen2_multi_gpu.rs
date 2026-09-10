//! One complete model per NVIDIA device. No tensor/model sharding is performed.
use rullm::backends::{Nvidia,NvidiaDevice,cuda_driver_api_version};
use rullm::runtime::{BackendKind,BoundedWorker,DeviceKey,KernelMode,ReplicaPool,RuntimeError,RuntimeErrorKind};
use rullm::{GreedyGenerationConfig,generate_greedy_packed_with_mode,load_huggingface_qwen2_pipeline};
use ruda_tensor::api::backend::Backend;
use std::{error::Error,io};
type B=Nvidia<f32,i32>;
fn main()->Result<(),Box<dyn Error>> {
    let mut args=std::env::args().skip(1);
    let directory=args.next().ok_or_else(||io::Error::other("usage: qwen2_multi_gpu <model-directory> <prompt> <device-indices e.g. 0,1>"))?;
    let prompt=args.next().ok_or_else(||io::Error::other("prompt required"))?;
    let indices=args.next().ok_or_else(||io::Error::other("explicit device indices required"))?
        .split(',').map(str::parse::<usize>).collect::<Result<Vec<_>,_>>()?;
    if args.next().is_some() {return Err(io::Error::other("too many arguments").into());}
    // Reject aliases BEFORE starting any device threads.
    if indices.iter().any(|&i|u16::try_from(i).is_err()) {return Err(io::Error::other("device index exceeds Ruda device identity range").into());}
    let unique=indices.iter().copied().collect::<std::collections::BTreeSet<_>>();
    if unique.len()!=indices.len() {return Err(io::Error::other("duplicate device indices").into());}
    eprintln!("driver-supported CUDA API: {} (not a PTX compatibility proof)",cuda_driver_api_version()?.encoded());
    let mut workers=Vec::new();
    for &index in &indices {
        let directory=directory.clone();
        let ordinal=u32::try_from(index)?;
        workers.push(BoundedWorker::spawn(DeviceKey {backend:BackendKind::Nvidia,ordinal},2,move || {
            let device=NvidiaDevice::new(index);
            let pipeline=load_huggingface_qwen2_pipeline::<B>(&directory,&device)
                .map_err(|e|RuntimeError::new(RuntimeErrorKind::InvalidInput,e.to_string()))?;
            let model=pipeline.loaded.model.into_packed_inference();
            B::sync(&device).map_err(|e|RuntimeError::new(RuntimeErrorKind::Synchronization,e.to_string()))?;
            Ok((device,model,pipeline.loaded.config,pipeline.loaded.default_eos_token_ids,pipeline.tokenizer))
        })?);
    }
    let mut pool=ReplicaPool::new(workers)?;
    let mut jobs=Vec::new();
    // Pin one request per explicitly selected device, to exercise every replica.
    for &index in &indices {
        let device=DeviceKey {backend:BackendKind::Nvidia,ordinal:u32::try_from(index)?};
        let prompt=prompt.clone();
        let handle=pool.worker_for(&device).expect("configured worker").try_submit(move |state,cancel| {
            let (device,model,config,eos,tokenizer)=state;
            if cancel.is_cancelled() {return Err(RuntimeError::new(RuntimeErrorKind::Cancelled,"cancelled before inference"));}
            let encoded=tokenizer.encode(prompt.as_str(),false)
                .map_err(|e|RuntimeError::new(RuntimeErrorKind::InvalidInput,e.to_string()))?;
            let ids=encoded.get_ids().iter().map(|&id|i32::try_from(id))
                .collect::<Result<Vec<_>,_>>().map_err(|e|RuntimeError::new(RuntimeErrorKind::InvalidInput,e.to_string()))?;
            let result=generate_greedy_packed_with_mode(model,config,&ids,
                &GreedyGenerationConfig {max_new_tokens:32,eos_token_ids:eos.clone()},device,KernelMode::Automatic);
            // Fence even when generation returned a recoverable host error.
            B::sync(device).map_err(|e|RuntimeError::new(RuntimeErrorKind::Synchronization,e.to_string()))?;
            let output=result.map_err(|e|RuntimeError::new(RuntimeErrorKind::InvalidInput,e.to_string()))?;
            let ids=output.generated_token_ids.into_iter().map(u32::try_from).collect::<Result<Vec<_>,_>>()
                .map_err(|e|RuntimeError::new(RuntimeErrorKind::InvalidInput,e.to_string()))?;
            tokenizer.decode(&ids,true).map_err(|e|RuntimeError::new(RuntimeErrorKind::InvalidInput,e.to_string()))
        })?;
        jobs.push((device,handle));
    }
    // JobHandle also implements Future; wait() keeps this example executor-free.
    // This example only checks cancellation before inference, not per token.
    for (device,handle) in jobs {println!("{device:?}: {}",handle.wait()?);}
    pool.shutdown()?;
    Ok(())
}
