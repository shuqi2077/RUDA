use std::error::Error;
use std::fmt::{Display, Formatter};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub enum WorkError {
    WorkerStart(String),
    WorkerQueueClosed,
    WorkerPanic,
    Poisoned,
}

impl Display for WorkError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorkerStart(error) => {
                write!(formatter, "cannot start collective worker: {error}")
            }
            Self::WorkerQueueClosed => write!(formatter, "collective worker queue is closed"),
            Self::WorkerPanic => write!(formatter, "collective worker panicked"),
            Self::Poisoned => write!(formatter, "collective synchronization state is poisoned"),
        }
    }
}

impl Error for WorkError {}

#[derive(Debug)]
struct WorkCompletion<T, E> {
    result: Mutex<Option<Result<T, E>>>,
    changed: Condvar,
}

/// Asynchronous collective completion handle. Dropping it detaches the host
/// worker but does not cancel or reorder an already submitted collective.
#[derive(Debug)]
pub struct CollectiveWork<T, E = WorkError> {
    completion: Arc<WorkCompletion<T, E>>,
    worker: Option<JoinHandle<()>>,
}

type OrderedTask = Box<dyn FnOnce() + Send + 'static>;

#[derive(Clone)]
pub struct OrderedWorkQueue {
    sender: mpsc::Sender<OrderedTask>,
}

impl std::fmt::Debug for OrderedWorkQueue {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("OrderedWorkQueue")
    }
}

impl OrderedWorkQueue {
    pub fn new(name: &'static str) -> Result<Self, WorkError> {
        let (sender, receiver) = mpsc::channel::<OrderedTask>();
        thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                while let Ok(task) = receiver.recv() {
                    task();
                }
            })
            .map_err(|error| WorkError::WorkerStart(error.to_string()))?;
        Ok(Self { sender })
    }

    pub fn submit<T: Send + 'static, E: From<WorkError> + Send + 'static>(
        &self,
        task: impl FnOnce() -> Result<T, E> + Send + 'static,
    ) -> Result<CollectiveWork<T, E>, E> {
        let completion = Arc::new(WorkCompletion {
            result: Mutex::new(None),
            changed: Condvar::new(),
        });
        let target = Arc::clone(&completion);
        self.sender
            .send(Box::new(move || {
                let result = catch_unwind(AssertUnwindSafe(task))
                    .map_err(|_| E::from(WorkError::WorkerPanic))
                    .and_then(|result| result);
                if let Ok(mut state) = target.result.lock() {
                    *state = Some(result);
                    target.changed.notify_all();
                }
            }))
            .map_err(|_| E::from(WorkError::WorkerQueueClosed))?;
        Ok(CollectiveWork {
            completion,
            worker: None,
        })
    }
}

impl<T: Send + 'static, E: From<WorkError> + Send + 'static> CollectiveWork<T, E> {
    pub fn spawn(
        name: &'static str,
        task: impl FnOnce() -> Result<T, E> + Send + 'static,
    ) -> Result<Self, E> {
        let completion = Arc::new(WorkCompletion {
            result: Mutex::new(None),
            changed: Condvar::new(),
        });
        let target = Arc::clone(&completion);
        let worker = thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let result = catch_unwind(AssertUnwindSafe(task))
                    .map_err(|_| E::from(WorkError::WorkerPanic))
                    .and_then(|result| result);
                if let Ok(mut state) = target.result.lock() {
                    *state = Some(result);
                    target.changed.notify_all();
                }
            })
            .map_err(|error| E::from(WorkError::WorkerStart(error.to_string())))?;
        Ok(Self {
            completion,
            worker: Some(worker),
        })
    }

    pub fn is_complete(&self) -> bool {
        self.completion
            .result
            .lock()
            .map(|result| result.is_some())
            .unwrap_or(true)
    }

    pub fn wait_for(&self, timeout: Duration) -> Result<bool, E> {
        let result = self
            .completion
            .result
            .lock()
            .map_err(|_| E::from(WorkError::Poisoned))?;
        let (result, _) = self
            .completion
            .changed
            .wait_timeout_while(result, timeout, |result| result.is_none())
            .map_err(|_| E::from(WorkError::Poisoned))?;
        Ok(result.is_some())
    }

    pub fn wait(mut self) -> Result<T, E> {
        let result = {
            let mut state = self
                .completion
                .result
                .lock()
                .map_err(|_| E::from(WorkError::Poisoned))?;
            while state.is_none() {
                state = self
                    .completion
                    .changed
                    .wait(state)
                    .map_err(|_| E::from(WorkError::Poisoned))?;
            }
            state.take().expect("collective wait exits only when ready")
        };
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| E::from(WorkError::WorkerPanic))?;
        }
        result
    }
}
