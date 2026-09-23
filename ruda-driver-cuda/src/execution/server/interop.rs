use super::*;
use crate::interop::StreamCommand;
use cudarc::driver::sys::*;
use ruda::runtime::stream::GcTask;

#[derive(Debug)]
pub(super) struct Event { raw:usize, timing:bool }
#[derive(Debug)]
pub(super) struct InteropState {
    context:usize,
    next_stream:u64,
    stream_capacity:u64,
    next_event:u64,
    events:HashMap<u64,Event>,
}
impl InteropState {
    pub(super) fn new(context:usize,capacity:u8)->Self {
        Self { context,next_stream:1,stream_capacity:capacity as u64,next_event:1,events:HashMap::new() }
    }
    fn stream(&self,id:u64)->Result<StreamId,ServerError> {
        if id>=self.next_stream || id>=self.stream_capacity {
            return Err(error("unknown RUDA native stream"));
        }
        Ok(StreamId { value:id })
    }
    fn event(&self,id:u64)->Result<&Event,ServerError> { self.events.get(&id).ok_or_else(||error("unknown/destroyed RUDA event")) }
}
impl Drop for InteropState {
    fn drop(&mut self) {
        // Destructors cannot report errors. Normal event destruction is checked.
        // Restore the prior context on the service thread after cleanup.
        unsafe {
            let mut old=std::ptr::null_mut();
            if cuCtxGetCurrent(&mut old)!=CUresult::CUDA_SUCCESS { return; }
            if cuCtxSetCurrent(self.context as CUcontext)!=CUresult::CUDA_SUCCESS { return; }
            for event in self.events.values() { let _=cuEventDestroy_v2(event.raw as CUevent); }
            let _=cuCtxSetCurrent(old);
        }
    }
}
fn error(reason:&str)->ServerError { ServerError::Generic { reason:reason.into(),backtrace:BackTrace::capture() } }
fn status(value:CUresult)->Result<(),ServerError> {
    if value==CUresult::CUDA_SUCCESS { Ok(()) } else { Err(error(&format!("GPU interop error: {value:?}"))) }
}
fn ready(value:CUresult)->Result<u64,ServerError> {
    if value==CUresult::CUDA_ERROR_NOT_READY { Ok(0) } else { status(value)?;Ok(1) }
}
impl CudaServer {
    fn interop_stream(&mut self,id:u64)->Result<CUstream,ServerError> {
        let stream_id=self.interop.stream(id)?;
        let mut command=self.command_no_inputs(stream_id,StreamErrorMode { ignore:false,flush:true })?;
        Ok(command.streams.current().sys)
    }
    pub(crate) fn interop_command(&mut self,command:StreamCommand)->Result<u64,ServerError> {
        self.ctx.unsafe_set_current().map_err(|e|error(&format!("GPU context error: {e:?}")))?;
        match command {
            StreamCommand::Create=>{
                if self.interop.next_stream>=self.interop.stream_capacity {
                    return Err(error("native stream pool exhausted; increase streaming.max_streams (no ID aliasing)"));
                }
                let id=self.interop.next_stream;
                self.interop.next_stream+=1;
                self.interop_stream(id)?;
                Ok(id)
            }
            StreamCommand::Validate(id)=>{self.interop.stream(id)?;Ok(id)}
            StreamCommand::Query(id)=>{let stream=self.interop_stream(id)?;ready(unsafe{cuStreamQuery(stream)})}
            StreamCommand::Synchronize(id)=>{let stream=self.interop_stream(id)?;status(unsafe{cuStreamSynchronize(stream)})?;Ok(0)}
            StreamCommand::Record {stream,event,timing}=>{
                let stream=self.interop_stream(stream)?;
                let mut id=event;
                let raw=if id==0 {
                    id=self.interop.next_event;
                    let next=id.checked_add(1).ok_or_else(||error("event identifier overflow"))?;
                    let mut raw=std::ptr::null_mut();
                    status(unsafe{cuEventCreate(&mut raw,if timing {0} else {CUevent_flags::CU_EVENT_DISABLE_TIMING as u32})})?;
                    self.interop.next_event=next;
                    self.interop.events.insert(id,Event{raw:raw as usize,timing});
                    raw
                } else {
                    let e=self.interop.event(id)?;
                    if e.timing!=timing { return Err(error("cannot change timing policy on an existing event")); }
                    e.raw as CUevent
                };
                if let Err(err)=status(unsafe{cuEventRecord(raw,stream)}) {
                    if event==0 { self.interop.events.remove(&id);unsafe{let _=cuEventDestroy_v2(raw);} }
                    return Err(err);
                }
                Ok(id)
            }
            StreamCommand::Wait {stream,event}=>{
                let stream=self.interop_stream(stream)?;
                if event!=0 { let raw=self.interop.event(event)?.raw as CUevent;
                    status(unsafe{cuStreamWaitEvent(stream,raw,0)})?; }
                Ok(0)
            }
            StreamCommand::EventQuery(id)=>{
                if id==0 { return Ok(1); }
                ready(unsafe{cuEventQuery(self.interop.event(id)?.raw as CUevent)})
            }
            StreamCommand::EventSynchronize(id)=>{
                if id!=0 {status(unsafe{cuEventSynchronize(self.interop.event(id)?.raw as CUevent)})?;}
                Ok(0)
            }
            StreamCommand::EventDestroy(id)=>{
                if id!=0 {
                    let raw=self.interop.event(id)?.raw as CUevent;
                    status(unsafe{cuEventDestroy_v2(raw)})?;self.interop.events.remove(&id);
                } Ok(0)
            }
            StreamCommand::Elapsed {start,end}=>{
                let a=self.interop.event(start)?;let b=self.interop.event(end)?;
                if !a.timing || !b.timing { return Err(error("elapsed_time requires two timing-enabled events")); }
                let ms=unsafe{cudarc::driver::result::event::elapsed(a.raw as CUevent,b.raw as CUevent)}
                    .map_err(|e|error(&format!("GPU interop error: {:?}",e.0)))?;
                Ok(ms.to_bits() as u64)
            }
            StreamCommand::DeviceSynchronize=>{
                // Report deferred RUDA launch errors as well as driver errors.
                for id in 0..self.interop.next_stream { self.interop_stream(id)?; }
                status(unsafe{cuCtxSynchronize()})?;Ok(0)
            }
        }
    }
    pub(crate) fn interop_record_allocation(&mut self,id:u64,handle:Handle)->Result<(),ServerError> {
        let stream_id=self.interop.stream(id)?;
        let binding=handle.clone().binding();
        let mut command=self.command(stream_id,std::iter::once(&binding),StreamErrorMode {ignore:false,flush:true})?;
        let event=Fence::new(command.streams.current().sys);
        command.streams.gc(GcTask::new(handle,event));
        Ok(())
    }
}
