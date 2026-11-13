use super::*;
use crate::interop::StreamCommand;
use cudarc::driver::sys::*;
use ruda::runtime::stream::GcTask;

#[derive(Debug)]
pub(super) struct Event {
    raw: usize,
    timing: bool,
    capture: Option<StreamId>,
    warmup: Option<StreamId>,
}
#[derive(Debug)]
pub(super) struct InteropState {
    context: usize,
    next_stream: u64,
    stream_capacity: u64,
    next_event: u64,
    events: HashMap<u64, Event>,
}
impl InteropState {
    pub(super) fn new(context: usize, capacity: u8) -> Self {
        Self {
            context,
            next_stream: 1,
            stream_capacity: capacity as u64,
            next_event: 1,
            events: HashMap::new(),
        }
    }
    fn stream(&self, id: u64) -> Result<StreamId, ServerError> {
        if id >= self.next_stream || id >= self.stream_capacity {
            return Err(error("unknown RUDA native stream"));
        }
        Ok(StreamId { value: id })
    }
    fn event(&self, id: u64) -> Result<&Event, ServerError> {
        self.events
            .get(&id)
            .ok_or_else(|| error("unknown/destroyed RUDA event"))
    }
    pub(super) fn end_capture(&mut self, root: StreamId) {
        for event in self.events.values_mut() {
            if event.capture == Some(root) {
                event.capture = None;
            }
        }
    }
    pub(super) fn end_warmup(&mut self, root: StreamId) {
        for event in self.events.values_mut() {
            if event.warmup == Some(root) {
                event.warmup = None;
            }
        }
    }
}
impl Drop for InteropState {
    fn drop(&mut self) {
        // Destructors cannot report errors. Normal event destruction is checked.
        // Restore the prior context on the service thread after cleanup.
        unsafe {
            let mut old = std::ptr::null_mut();
            if cuCtxGetCurrent(&mut old) != CUresult::CUDA_SUCCESS {
                return;
            }
            if cuCtxSetCurrent(self.context as CUcontext) != CUresult::CUDA_SUCCESS {
                return;
            }
            for event in self.events.values() {
                let _ = cuEventDestroy_v2(event.raw as CUevent);
            }
            let _ = cuCtxSetCurrent(old);
        }
    }
}
fn error(reason: &str) -> ServerError {
    ServerError::Generic {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    }
}
fn status(value: CUresult) -> Result<(), ServerError> {
    if value == CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(error(&format!("GPU interop error: {value:?}")))
    }
}
fn ready(value: CUresult) -> Result<u64, ServerError> {
    if value == CUresult::CUDA_ERROR_NOT_READY {
        Ok(0)
    } else {
        status(value)?;
        Ok(1)
    }
}
impl CudaServer {
    pub(crate) fn interop_enqueue<T, F>(&mut self, id: u64, operation: F) -> Result<T, ServerError>
    where
        F: FnOnce(crate::interop::NativeScope) -> Result<T, String>,
    {
        let stream = self.interop.stream(id)?;
        self.interop_execution_enqueue(stream, operation)
    }
    pub(crate) fn interop_execution_enqueue<T, F>(
        &mut self,
        stream: StreamId,
        operation: F,
    ) -> Result<T, ServerError>
    where
        F: FnOnce(crate::interop::NativeScope) -> Result<T, String>,
    {
        self.set_current_checked()?;
        let stream = self.execution_native_stream(stream)?;
        let scope = crate::interop::NativeScope {
            context: self.ctx.context as usize,
            stream: stream as usize,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(scope)));
        match result {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(reason)) => Err(error(&reason)),
            Err(_) => Err(error("native enqueue callback panicked")),
        }
    }
    pub(crate) fn interop_native_scope<T, F>(
        &mut self,
        id: u64,
        operation: F,
    ) -> Result<T, ServerError>
    where
        F: FnOnce(crate::interop::NativeScope) -> Result<T, String>,
    {
        let stream = self.interop.stream(id)?;
        self.interop_execution_scope(stream, operation)
    }
    pub(crate) fn interop_execution_scope<T, F>(
        &mut self,
        stream_id: StreamId,
        operation: F,
    ) -> Result<T, ServerError>
    where
        F: FnOnce(crate::interop::NativeScope) -> Result<T, String>,
    {
        self.set_current_checked()?;
        let stream = self.execution_native_stream(stream_id)?;
        if self.capture_group(stream_id).is_some() {
            let failure = error("Host synchronization/query cannot observe a recording stream");
            self.capture_failure(stream_id, &failure);
            return Err(failure);
        }
        status(unsafe { cuStreamSynchronize(stream) })?;
        let scope = crate::interop::NativeScope {
            context: self.ctx.context as usize,
            stream: stream as usize,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(scope)));
        // Drain even on a callback error/panic before releasing callback-owned
        // host data. Report both failures when CUDA also reports an error.
        let sync = status(unsafe { cuStreamSynchronize(stream) });
        match (result, sync) {
            (Ok(Ok(value)), Ok(())) => Ok(value),
            (Ok(Err(reason)), Ok(())) => Err(error(&reason)),
            (Ok(Ok(_)), Err(cuda)) => Err(cuda),
            (Ok(Err(reason)), Err(cuda)) => Err(error(&format!("{reason}; drain: {cuda:?}"))),
            (Err(_), sync) => Err(error(&format!(
                "external CUDA callback panicked; drain: {sync:?}"
            ))),
        }
    }

    fn interop_stream(&mut self, id: u64) -> Result<CUstream, ServerError> {
        let stream_id = self.interop.stream(id)?;
        self.execution_native_stream(stream_id)
    }
    fn execution_native_stream(&mut self, stream_id: StreamId) -> Result<CUstream, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        )?;
        Ok(command.streams.current().sys)
    }
    fn interop_host_stream(&mut self, id: u64) -> Result<CUstream, ServerError> {
        let stream = self.interop_stream(id)?;
        let stream_id = self.interop.stream(id)?;
        if self.capture_group(stream_id).is_some() {
            let failure = error("Host synchronization/query cannot observe a recording stream");
            self.capture_failure(stream_id, &failure);
            return Err(failure);
        }
        Ok(stream)
    }
    pub(crate) fn interop_command(&mut self, command: StreamCommand) -> Result<u64, ServerError> {
        self.ctx
            .unsafe_set_current()
            .map_err(|e| error(&format!("GPU context error: {e:?}")))?;
        let (stream, events) = match command {
            StreamCommand::Record { stream, .. } => (Some(stream), [0, 0]),
            StreamCommand::Wait { stream, event } => (Some(stream), [event, 0]),
            StreamCommand::Query(stream) | StreamCommand::Synchronize(stream) => {
                (Some(stream), [0, 0])
            }
            StreamCommand::EventQuery(event)
            | StreamCommand::EventSynchronize(event)
            | StreamCommand::EventDestroy(event) => (None, [event, 0]),
            StreamCommand::Elapsed { start, end } => (None, [start, end]),
            _ => (None, [0, 0]),
        };
        let mut affected = Vec::new();
        if let Some(stream) = stream {
            let stream = self.interop.stream(stream)?;
            if let Some(owner) = self.capture_owner(stream) {
                if owner != stream {
                    return Err(error(
                        "The physical stream belongs to another logical stream's capture",
                    ));
                }
                if let Some(root) = self.capture_group(stream) {
                    affected.push(root);
                }
            }
        }
        for event in events {
            if let Some(root) = self
                .interop
                .events
                .get(&event)
                .and_then(|event| event.capture)
            {
                if !affected.contains(&root) {
                    affected.push(root);
                }
            }
        }
        let result = (|| {
            match command {
                StreamCommand::Create | StreamCommand::CreatePriority(_) => {
                    if self.interop.next_stream >= self.interop.stream_capacity {
                        return Err(error(
                            "native stream pool exhausted; increase streaming.max_streams (no ID aliasing)",
                        ));
                    }
                    let id = self.interop.next_stream;
                    if let StreamCommand::CreatePriority(priority) = command {
                        let initialized = self
                            .streams
                            .try_initialize_with(&StreamId { value: id }, |backend| {
                                backend.create_stream_with_priority(priority)
                            })?;
                        if !initialized {
                            return Err(error(
                                "priority stream slot is already in use; no pending stream is replaced",
                            ));
                        }
                    }
                    self.interop.next_stream += 1;
                    self.interop_stream(id)?;
                    Ok(id)
                }
                StreamCommand::Validate(id) => {
                    self.interop.stream(id)?;
                    Ok(id)
                }
                StreamCommand::Priority(id) => {
                    let stream = self.interop_stream(id)?;
                    let mut priority = 0i32;
                    status(unsafe { cuStreamGetPriority(stream, &mut priority) })?;
                    Ok(priority as u32 as u64)
                }
                StreamCommand::PriorityRange => {
                    let mut least = 0i32;
                    let mut greatest = 0i32;
                    status(unsafe { cuCtxGetStreamPriorityRange(&mut least, &mut greatest) })?;
                    Ok((least as u32 as u64) | ((greatest as u32 as u64) << 32))
                }
                StreamCommand::Query(id) => {
                    let stream = self.interop_host_stream(id)?;
                    ready(unsafe { cuStreamQuery(stream) })
                }
                StreamCommand::Synchronize(id) => {
                    let stream = self.interop_host_stream(id)?;
                    status(unsafe { cuStreamSynchronize(stream) })?;
                    Ok(0)
                }
                StreamCommand::Record {
                    stream,
                    event,
                    timing,
                } => {
                    let stream_id = self.interop.stream(stream)?;
                    let stream = self.interop_stream(stream)?;
                    let capture = self.capture_group(stream_id);
                    let warmup = self.capture_warmup_root(stream_id);
                    let mut id = event;
                    let raw = if id == 0 {
                        id = self.interop.next_event;
                        let next = id
                            .checked_add(1)
                            .ok_or_else(|| error("event identifier overflow"))?;
                        let mut raw = std::ptr::null_mut();
                        status(unsafe {
                            cuEventCreate(
                                &mut raw,
                                if timing {
                                    0
                                } else {
                                    CUevent_flags::CU_EVENT_DISABLE_TIMING as u32
                                },
                            )
                        })?;
                        self.interop.next_event = next;
                        self.interop.events.insert(
                            id,
                            Event {
                                raw: raw as usize,
                                timing,
                                capture: None,
                                warmup: None,
                            },
                        );
                        raw
                    } else {
                        let e = self.interop.event(id)?;
                        if e.timing != timing {
                            return Err(error("cannot change timing policy on an existing event"));
                        }
                        e.raw as CUevent
                    };
                    if let Err(err) = status(unsafe { cuEventRecord(raw, stream) }) {
                        if let Some(root) =
                            self.interop.events.get(&id).and_then(|event| event.capture)
                        {
                            self.capture_failure(root, &err);
                        }
                        if event == 0 {
                            self.interop.events.remove(&id);
                            unsafe {
                                let _ = cuEventDestroy_v2(raw);
                            }
                        }
                        return Err(err);
                    }
                    let event = self.interop.events.get_mut(&id).expect("recorded event");
                    event.capture = capture;
                    event.warmup = warmup;
                    Ok(id)
                }
                StreamCommand::Wait { stream, event } => {
                    let stream_id = self.interop.stream(stream)?;
                    if event == 0 {
                        self.interop_stream(stream)?;
                        return Ok(0);
                    }
                    let event = self.interop.event(event)?;
                    let (raw, capture, warmup) =
                        (event.raw as CUevent, event.capture, event.warmup);
                    if let Some(root) = capture {
                        self.capture_join_stream(stream_id, root, raw)?;
                    } else {
                        if let Some(root) = warmup {
                            self.capture_prepare_branch(stream_id, root)?;
                        }
                        let stream = self.interop_stream(stream)?;
                        status(unsafe { cuStreamWaitEvent(stream, raw, 0) })?;
                    }
                    Ok(0)
                }
                StreamCommand::EventQuery(id) => {
                    if id == 0 {
                        return Ok(1);
                    }
                    let event = self.interop.event(id)?;
                    if event.capture.is_some() {
                        return Err(error("Cannot query an event in an active capture graph"));
                    }
                    ready(unsafe { cuEventQuery(event.raw as CUevent) })
                }
                StreamCommand::EventSynchronize(id) => {
                    if id != 0 {
                        let event = self.interop.event(id)?;
                        if event.capture.is_some() {
                            return Err(error(
                                "Cannot synchronize an event in an active capture graph",
                            ));
                        }
                        status(unsafe { cuEventSynchronize(event.raw as CUevent) })?;
                    }
                    Ok(0)
                }
                StreamCommand::EventDestroy(id) => {
                    if id != 0 {
                        let raw = self.interop.event(id)?.raw as CUevent;
                        status(unsafe { cuEventDestroy_v2(raw) })?;
                        self.interop.events.remove(&id);
                    }
                    Ok(0)
                }
                StreamCommand::Elapsed { start, end } => {
                    let a = self.interop.event(start)?;
                    let b = self.interop.event(end)?;
                    if a.capture.is_some() || b.capture.is_some() {
                        return Err(error("Cannot time events in an active capture graph"));
                    }
                    if !a.timing || !b.timing {
                        return Err(error("elapsed_time requires two timing-enabled events"));
                    }
                    let ms = unsafe {
                        cudarc::driver::result::event::elapsed(a.raw as CUevent, b.raw as CUevent)
                    }
                    .map_err(|e| error(&format!("GPU interop error: {:?}", e.0)))?;
                    Ok(ms.to_bits() as u64)
                }
                StreamCommand::NativeStream(id) => Ok(self.interop_stream(id)? as usize as u64),
                StreamCommand::DeviceSynchronize => {
                    if !self.captures.active.is_empty() {
                        let failure =
                            error("Device synchronization cannot observe active recordings");
                        for owner in self.captures.active.clone() {
                            self.capture_failure(owner, &failure);
                        }
                        return Err(failure);
                    }
                    // Report deferred RUDA launch errors as well as driver errors.
                    for id in 0..self.interop.next_stream {
                        self.interop_stream(id)?;
                    }
                    status(unsafe { cuCtxSynchronize() })?;
                    Ok(0)
                }
            }
        })();
        if let Err(failure) = &result {
            for root in affected {
                self.capture_failure(root, failure);
            }
        }
        result
    }
    pub(crate) fn interop_record_allocation(
        &mut self,
        id: u64,
        handle: Handle,
    ) -> Result<(), ServerError> {
        let stream_id = self.interop.stream(id)?;
        let binding = handle.clone().binding();
        let mut command = self.command(
            stream_id,
            std::iter::once(&binding),
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        )?;
        if command.streams.current().capture.recording {
            if let Err(failure) = command.resource(binding) {
                let failure: ServerError = failure.into();
                command.error(failure.clone());
                return Err(failure);
            }
            command.streams.current().capture.retained.push(handle);
            return Ok(());
        }
        let event = Fence::new(command.streams.current().sys);
        command.streams.gc(GcTask::new(handle, event));
        Ok(())
    }
}
