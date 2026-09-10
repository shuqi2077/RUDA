use super::{DeviceKey, RuntimeError, RuntimeErrorKind};
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}, mpsc};
use std::task::{Context, Poll, Waker};
use std::thread::{self, JoinHandle};

#[derive(Debug, Clone, Default)]
pub struct CancellationFlag(Arc<AtomicBool>);
impl CancellationFlag {
    pub fn cancel(&self) { self.0.store(true, Ordering::Release); }
    pub fn is_cancelled(&self) -> bool { self.0.load(Ordering::Acquire) }
}

struct ResultState<T> { value: Option<Result<T, RuntimeError>>, completed: bool, waker: Option<Waker> }
struct ResultCell<T> { state: Mutex<ResultState<T>>, ready: Condvar }
impl<T> ResultCell<T> {
    fn new() -> Self { Self { state: Mutex::new(ResultState { value: None, completed: false, waker: None }), ready: Condvar::new() } }
    fn complete(&self, value: Result<T, RuntimeError>) {
        let wake = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.completed { return; }
            state.value = Some(value);
            state.completed = true;
            state.waker.take()
        };
        self.ready.notify_all();
        // Never call executor-provided code while holding the result lock.
        if let Some(waker) = wake { waker.wake(); }
    }
}

/// A runtime-independent Future, also usable from synchronous Rust with wait().
/// Dropping it requests cooperative cancellation, never a GPU abort.
#[must_use = "dropping a job handle requests cancellation"]
pub struct JobHandle<T> { cell: Arc<ResultCell<T>>, cancellation: CancellationFlag }
impl<T> JobHandle<T> {
    pub fn cancel(&self) { self.cancellation.cancel(); }
    pub fn cancellation(&self) -> CancellationFlag { self.cancellation.clone() }
    pub fn wait(self) -> Result<T, RuntimeError> {
        let mut state = self.cell.state.lock().map_err(|_| RuntimeError::internal("job result lock poisoned"))?;
        while !state.completed {
            state = self.cell.ready.wait(state).map_err(|_| RuntimeError::internal("job result lock poisoned"))?;
        }
        state.value.take().unwrap_or_else(|| Err(RuntimeError::internal("job result was already consumed")))
    }
}
impl<T> Future for JobHandle<T> {
    type Output = Result<T, RuntimeError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = match self.cell.state.lock() {
            Ok(state) => state,
            Err(_) => return Poll::Ready(Err(RuntimeError::internal("job result lock poisoned"))),
        };
        if state.completed {
            Poll::Ready(state.value.take().unwrap_or_else(|| Err(RuntimeError::internal("job result was already consumed"))))
        } else {
            if state.waker.as_ref().is_none_or(|w| !w.will_wake(cx.waker())) { state.waker = Some(cx.waker().clone()); }
            Poll::Pending
        }
    }
}
impl<T> Drop for JobHandle<T> { fn drop(&mut self) { self.cancellation.cancel(); } }

struct SlotPermit(Arc<AtomicUsize>);
impl Drop for SlotPermit { fn drop(&mut self) { self.0.fetch_sub(1, Ordering::AcqRel); } }

type Operation<S> = Box<dyn FnOnce(&mut S, &CancellationFlag) -> Option<RuntimeError> + Send>;
struct Job<S> {
    operation: Operation<S>,
    reject: Box<dyn FnOnce(RuntimeError) + Send>,
    cancellation: CancellationFlag,
    _permit: SlotPermit,
}

fn fatal(error: &RuntimeError) -> bool {
    matches!(error.kind(), RuntimeErrorKind::DeviceLost | RuntimeErrorKind::Synchronization
        | RuntimeErrorKind::WorkerPanicked | RuntimeErrorKind::Internal)
}

/// One dedicated host thread owns one model replica / device context. The
/// initializer runs on that SAME thread. No concurrent mutable access to S.
///
/// Contract for GPU operations: bind the right context, then synchronize ALL
/// streams touching S before returning success or a recoverable error. Return
/// DeviceLost/Synchronization when this cannot be established. The worker then
/// refuses every later job; it never treats catch_unwind as GPU recovery.
/// This is host asynchronous execution, not an implementation of GPU streams.
///
/// Failed/quarantined state is deliberately retained (leaked) on thread exit,
/// since destructors might otherwise recycle still-in-flight device buffers.
/// Release it by terminating/restarting the isolated worker process, not by
/// blindly reusing the device. Healthy state is dropped normally.
pub struct BoundedWorker<S> {
    device: DeviceKey,
    capacity: usize,
    sender: Option<mpsc::SyncSender<Job<S>>>,
    outstanding: Arc<AtomicUsize>,
    quarantined: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}
impl<S: Send + 'static> BoundedWorker<S> {
    pub fn spawn(
        device: DeviceKey,
        max_outstanding: usize,
        initialize: impl FnOnce() -> Result<S, RuntimeError> + Send + 'static,
    ) -> Result<Self, RuntimeError> {
        if max_outstanding == 0 { return Err(RuntimeError::invalid("worker capacity must be positive")); }
        let (sender, receiver) = mpsc::sync_channel::<Job<S>>(max_outstanding);
        let outstanding = Arc::new(AtomicUsize::new(0));
        let quarantined = Arc::new(AtomicBool::new(false));
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_quarantined = quarantined.clone();
        let worker_stopping = stopping.clone();
        let name = format!("rullm-{:?}-{}", device.backend, device.ordinal);
        let thread = thread::Builder::new().name(name).spawn(move || {
            let initialized = catch_unwind(AssertUnwindSafe(initialize));
            let (mut state, mut failed) = match initialized {
                Ok(Ok(state)) => (Some(state), None),
                Ok(Err(error)) => (None, Some(error)),
                Err(_) => (None, Some(RuntimeError::new(RuntimeErrorKind::WorkerPanicked, "replica initialization panicked"))),
            };
            if failed.is_some() { worker_quarantined.store(true, Ordering::Release); }
            while let Ok(job) = receiver.recv() {
                if let Some(error) = &failed { (job.reject)(error.clone()); continue; }
                if worker_stopping.load(Ordering::Acquire) || job.cancellation.is_cancelled() {
                    (job.reject)(RuntimeError::new(RuntimeErrorKind::Cancelled, "job cancelled before execution"));
                    continue;
                }
                let result = catch_unwind(AssertUnwindSafe(|| {
                    (job.operation)(state.as_mut().expect("healthy worker has state"), &job.cancellation)
                }));
                match result {
                    Ok(Some(error)) => {
                        worker_quarantined.store(true, Ordering::Release);
                        failed = Some(error);
                    }
                    Ok(None) => {}
                    Err(_) => {
                        let error = RuntimeError::new(RuntimeErrorKind::WorkerPanicked,
                            "replica operation panicked; worker quarantined, no device reset attempted");
                        worker_quarantined.store(true, Ordering::Release);
                        (job.reject)(error.clone());
                        failed = Some(error);
                    }
                }
            }
            if failed.is_some() {
                // Not a recovery path. Avoid running device-state destructors
                // without proof that outstanding work has stopped.
                if let Some(state) = state.take() { std::mem::forget(state); }
            }
        }).map_err(|e| RuntimeError::new(RuntimeErrorKind::Closed, format!("cannot spawn replica thread: {e}")))?;
        Ok(Self { device, capacity: max_outstanding, sender: Some(sender), outstanding,
            quarantined, stopping, thread: Some(thread) })
    }
    pub fn device(&self) -> &DeviceKey { &self.device }
    pub fn outstanding(&self) -> usize { self.outstanding.load(Ordering::Acquire) }
    pub fn is_quarantined(&self) -> bool { self.quarantined.load(Ordering::Acquire) }
    fn reserve_slot(&self) -> Result<SlotPermit, RuntimeError> {
        if self.is_quarantined() { return Err(RuntimeError::new(RuntimeErrorKind::DeviceLost, "replica is quarantined")); }
        if self.stopping.load(Ordering::Acquire) || self.sender.is_none() {
            return Err(RuntimeError::new(RuntimeErrorKind::Closed, "replica is closed"));
        }
        self.outstanding.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |n| (n < self.capacity).then(|| n + 1))
            .map_err(|_| RuntimeError::new(RuntimeErrorKind::QueueFull, "replica outstanding-job limit reached"))?;
        Ok(SlotPermit(self.outstanding.clone()))
    }
    fn submit_reserved<T: Send + 'static>(
        &self, permit: SlotPermit,
        operation: impl FnOnce(&mut S, &CancellationFlag) -> Result<T, RuntimeError> + Send + 'static,
    ) -> Result<JobHandle<T>, RuntimeError> {
        let cell = Arc::new(ResultCell::new());
        let result_cell = cell.clone();
        let rejected_cell = cell.clone();
        let cancellation = CancellationFlag::default();
        let job = Job {
            operation: Box::new(move |state, cancellation| {
                let result = operation(state, cancellation);
                let failure = result.as_ref().err().filter(|e| fatal(e)).cloned();
                result_cell.complete(result);
                failure
            }),
            reject: Box::new(move |error| rejected_cell.complete(Err(error))),
            cancellation: cancellation.clone(), _permit: permit,
        };
        match self.sender.as_ref().ok_or_else(|| RuntimeError::new(RuntimeErrorKind::Closed, "replica closed"))?.try_send(job) {
            Ok(()) => Ok(JobHandle { cell, cancellation }),
            Err(mpsc::TrySendError::Full(_)) => Err(RuntimeError::new(RuntimeErrorKind::QueueFull, "replica queue is full")),
            Err(mpsc::TrySendError::Disconnected(_)) => Err(RuntimeError::new(RuntimeErrorKind::Closed, "replica thread disconnected")),
        }
    }
    pub fn try_submit<T: Send + 'static>(
        &self, operation: impl FnOnce(&mut S, &CancellationFlag) -> Result<T, RuntimeError> + Send + 'static,
    ) -> Result<JobHandle<T>, RuntimeError> {
        self.submit_reserved(self.reserve_slot()?, operation)
    }
    /// Cancels queued work and waits for the current cooperative job to finish.
    /// No timeout is interpreted as a stopped GPU. Do not invoke from this worker.
    pub fn shutdown(&mut self) -> Result<(), RuntimeError> {
        self.stopping.store(true, Ordering::Release);
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            thread.join().map_err(|_| RuntimeError::new(RuntimeErrorKind::WorkerPanicked, "replica thread panicked during shutdown"))?;
        }
        Ok(())
    }
}
impl<S> Drop for BoundedWorker<S> {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Release);
        self.sender.take();
        // Nonblocking drop: the existing thread finishes its current job and
        // rejects queued jobs. Explicit shutdown() is the join boundary.
    }
}

/// Routes independent requests to healthy replicas. This is data parallelism,
/// not tensor parallelism: each S must already own a complete model. Different
/// backends can share a pool through a user-defined enum / trait-object S.
/// KV state never migrates between workers. Multi-step sessions must pin to a
/// worker via worker_for(); each pool submission should otherwise be a complete
/// request. Reservations are atomic even under concurrent producer threads.
pub struct ReplicaPool<S> { workers: Vec<BoundedWorker<S>>, cursor: AtomicUsize }
impl<S: Send + 'static> ReplicaPool<S> {
    pub fn new(workers: Vec<BoundedWorker<S>>) -> Result<Self, RuntimeError> {
        if workers.is_empty() { return Err(RuntimeError::invalid("replica pool must not be empty")); }
        let mut keys = std::collections::BTreeSet::new();
        if workers.iter().any(|w| !keys.insert(w.device().clone())) {
            return Err(RuntimeError::invalid("duplicate backend/device identity in replica pool"));
        }
        Ok(Self { workers, cursor: AtomicUsize::new(0) })
    }
    pub fn worker_for(&self, device: &DeviceKey) -> Option<&BoundedWorker<S>> {
        self.workers.iter().find(|w| w.device() == device)
    }
    pub fn try_submit<T: Send + 'static>(
        &self, operation: impl FnOnce(&mut S, &CancellationFlag) -> Result<T, RuntimeError> + Send + 'static,
    ) -> Result<(DeviceKey, JobHandle<T>), RuntimeError> {
        let start = self.cursor.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        let mut order: Vec<_> = (0..self.workers.len()).map(|n| (start + n) % self.workers.len()).collect();
        order.sort_by_key(|&i| self.workers[i].outstanding());
        let mut last = RuntimeError::new(RuntimeErrorKind::QueueFull, "all replicas are busy");
        for index in order {
            let worker = &self.workers[index];
            match worker.reserve_slot() {
                Ok(permit) => return worker.submit_reserved(permit, operation).map(|handle| (worker.device().clone(), handle)),
                Err(error) => last = error,
            }
        }
        Err(last)
    }
    pub fn shutdown(&mut self) -> Result<(), RuntimeError> {
        let mut first_error = None;
        for worker in &mut self.workers {
            if let Err(error) = worker.shutdown() { if first_error.is_none() { first_error = Some(error); } }
        }
        first_error.map_or(Ok(()), Err)
    }
}
