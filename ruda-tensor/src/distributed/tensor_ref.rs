use std::sync::{Arc, Mutex};

use crate::{Backend, tensor::FloatTensor};

trait TensorStorage<B: Backend>: Send + Sync {
    fn access(&self, callback: &mut dyn FnMut(&mut FloatTensor<B>));
}

struct OwnedStorage<B: Backend>(Mutex<FloatTensor<B>>);

impl<B: Backend> TensorStorage<B> for OwnedStorage<B> {
    fn access(&self, callback: &mut dyn FnMut(&mut FloatTensor<B>)) {
        let mut tensor = self.0.lock().expect("gradient storage lock poisoned");
        callback(&mut tensor);
    }
}

struct ProjectedStorage<B: Backend, T: Backend> {
    parent: TensorRef<B>,
    project: fn(&mut FloatTensor<B>) -> &mut FloatTensor<T>,
}

impl<B: Backend, T: Backend> TensorStorage<T> for ProjectedStorage<B, T> {
    fn access(&self, callback: &mut dyn FnMut(&mut FloatTensor<T>)) {
        self.parent.with_mut(|tensor| callback((self.project)(tensor)));
    }
}

/// Shared ownership of a gradient handle awaiting collective synchronization.
/// Clones share handle replacement; they do not borrow the gradient container.
#[derive(Clone)]
pub struct TensorRef<B: Backend> {
    storage: Arc<dyn TensorStorage<B>>,
}

impl<B: Backend> TensorRef<B> {
    /// Own a gradient handle for asynchronous synchronization.
    pub fn new(tensor: FloatTensor<B>) -> Self {
        Self {
            storage: Arc::new(OwnedStorage::<B>(Mutex::new(tensor))),
        }
    }

    /// Project a wrapper's inner handle while retaining ownership and its lock.
    pub fn map<T: Backend>(
        self,
        project: fn(&mut FloatTensor<B>) -> &mut FloatTensor<T>,
    ) -> TensorRef<T> {
        TensorRef {
            storage: Arc::new(ProjectedStorage::<B, T> { parent: self, project }),
        }
    }

    pub(crate) fn with_mut<R>(&self, operation: impl FnOnce(&mut FloatTensor<B>) -> R) -> R {
        let mut operation = Some(operation);
        let mut result = None;
        self.storage.access(&mut |tensor| {
            result = Some(operation.take().expect("gradient storage accessed twice")(tensor));
        });
        result.expect("gradient storage was not accessed")
    }

    /// Replace the shared handle. This does not synchronize device execution.
    pub fn replace(&self, tensor: FloatTensor<B>) {
        self.with_mut(|current| *current = tensor);
    }
}
