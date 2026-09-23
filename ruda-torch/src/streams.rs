//! Thread-local stream selection, backed by real RUDA physical streams/events.
use super::*;
use ruda_core::stream_id::StreamId;
use ruda_driver_cuda::interop::{self,StreamCommand};
thread_local! {static CURRENT:std::cell::Cell<u64>=const { std::cell::Cell::new(0) };}
pub(super) fn current()->u64 {CURRENT.get()}
pub(super) fn bind(client:&mut ComputeClient<CudaRuntime>) {
    // IDs are allocated/validated by the CUDA service and do not alias within
    // its configured pool. Tensor bindings retain their allocation-origin stream.
    unsafe {client.set_stream(StreamId{value:current()});}
}
pub(super) fn device_sync() {interop::command(client().device_id(),StreamCommand::DeviceSynchronize).expect("RUDA device sync failed");}

/// Private in-process ABI. All calls return status; errors stay on the caller's
/// thread. `object` is an event ID except for record-allocation (op 12).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ruda_torch_stream(op:u32,stream:u64,object:u64,flags:u32,result:*mut u64)->i32 {
    checked(|| {
        assert!(!result.is_null());
        let device=client().device_id();
        let value=match op {
            0=>current(),
            1=>interop::command(device,StreamCommand::Create).expect("stream creation failed"),
            2=>{interop::command(device,StreamCommand::Validate(stream)).expect("invalid stream");CURRENT.replace(stream)},
            3=>interop::command(device,StreamCommand::Query(stream)).expect("stream query failed"),
            4=>interop::command(device,StreamCommand::Synchronize(stream)).expect("stream synchronize failed"),
            5=>{assert!(flags<=1);interop::command(device,StreamCommand::Record {stream,event:object,timing:flags==1}).expect("event record failed")},
            6=>interop::command(device,StreamCommand::Wait {stream,event:object}).expect("event wait failed"),
            7=>interop::command(device,StreamCommand::EventQuery(object)).expect("event query failed"),
            8=>interop::command(device,StreamCommand::EventSynchronize(object)).expect("event synchronize failed"),
            9=>interop::command(device,StreamCommand::EventDestroy(object)).expect("event destroy failed"),
            10=>interop::command(device,StreamCommand::Elapsed {start:stream,end:object}).expect("event elapsed time failed"),
            11=>{device_sync();0},
            12=>{
                let allocation=object as *const Allocation;
                assert!(!allocation.is_null());
                let handle=unsafe{(*allocation).handle.clone()};
                interop::record_allocation(device,stream,handle).expect("record_stream allocation retention failed");0
            }
            _=>panic!("unknown RUDA stream command"),
        };
        unsafe{*result=value;}
    })
}
