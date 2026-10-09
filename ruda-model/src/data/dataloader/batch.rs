use super::{BatchStrategy, DataLoader, DataLoaderIterator, Progress, batcher::Batcher};
use ruda_dataset::{
    Dataset,
    transform::{PartialDataset, ShuffledDataset},
};
use ruda_tensor::api::backend::Backend;
use rand::SeedableRng;
use std::ops::DerefMut;
use std::sync::Arc;

/// A data loader that can be used to iterate over a dataset in batches.
pub struct BatchDataLoader<B: Backend, I, O> {
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Arc<dyn Batcher<B, I, O>>,
    device: B::Device,
    rng: Option<Arc<spin::Mutex<rand::rngs::StdRng>>>,
    batch_reading: bool,
}

impl<B: Backend, I, O> Clone for BatchDataLoader<B, I, O> {
    fn clone(&self) -> Self {
        Self {
            strategy: self.strategy.clone_dyn(),
            dataset: self.dataset.clone(),
            batcher: self.batcher.clone(),
            device: self.device.clone(),
            rng: self.rng.clone(),
            batch_reading: self.batch_reading,
        }
    }
}

impl<B: Backend, I, O> BatchDataLoader<B, I, O> {
    /// Creates a new batch data loader.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    /// * `device`  - The device to use when loading a batch.
    /// * `rng`     - The rng determining if the dataset is shuffled each time a dataloader
    ///   iterator is created.
    ///
    /// # Returns
    ///
    /// The batch data loader.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Arc<dyn Batcher<B, I, O>>,
        device: B::Device,
        rng: Option<rand::rngs::StdRng>,
    ) -> Self {
        Self {
            strategy,
            dataset,
            batcher,
            device,
            rng: rng.map(|rng| Arc::new(spin::Mutex::new(rng))),
            batch_reading: false,
        }
    }

    /// Enables `Dataset::get_many` reads using the strategy's batch size (default: false).
    ///
    /// Unknown or zero batch sizes read one item at a time. Items are still passed
    /// to the strategy individually; unread items remain buffered after an early
    /// batch. Missing items or a mismatched result length panic without rereading.
    pub fn batch_reading(mut self, enabled: bool) -> Self {
        self.set_batch_reading(enabled);
        self
    }

    pub(super) fn set_batch_reading(&mut self, enabled: bool) {
        self.batch_reading = enabled;
    }
}

/// A data loader iterator that can be used to iterate over a data loader.
struct BatchDataloaderIterator<B: Backend, I, O> {
    current_index: usize,
    strategy: Box<dyn BatchStrategy<I>>,
    dataset: Arc<dyn Dataset<I>>,
    batcher: Arc<dyn Batcher<B, I, O>>,
    device: B::Device,
    batch_reading: bool,
    pending: std::vec::IntoIter<I>,
    read_indices: Vec<usize>,
    read_failed: bool,
}

impl<B, I, O> DataLoader<B, O> for BatchDataLoader<B, I, O>
where
    B: Backend,
    I: Send + Sync + Clone + 'static,
    O: Send + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a> {
        // When starting a new iteration, we first check if the dataloader was created with an rng,
        // implying that we should shuffle the dataset beforehand, while advancing the current
        // rng to ensure that each new iteration shuffles the dataset differently.
        let dataset = match &self.rng {
            Some(rng) => Arc::new(ShuffledDataset::new(
                self.dataset.clone(),
                rng.lock().deref_mut(),
            )),
            None => self.dataset.clone(),
        };
        Box::new(BatchDataloaderIterator::new(
            self.strategy.clone_dyn(),
            dataset,
            self.batcher.clone(),
            self.device.clone(),
            self.batch_reading,
        ))
    }

    fn num_items(&self) -> usize {
        self.dataset.len()
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, O>> {
        let rng = self.rng.as_ref().map(|rng| {
            let mut rng = rng.lock();
            rng.fork()
        });
        Arc::new(
            Self::new(
                self.strategy.clone_dyn(),
                self.dataset.clone(),
                self.batcher.clone(),
                device.clone(),
                rng,
            )
            .batch_reading(self.batch_reading),
        )
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, O>> {
        let rng = self.rng.as_ref().map(|rng| {
            let mut rng = rng.lock();
            rng.fork()
        });
        let dataloader = Self::new(
            self.strategy.clone_dyn(),
            Arc::new(PartialDataset::new(self.dataset.clone(), start, end)),
            self.batcher.clone(),
            self.device.clone(),
            rng,
        )
        .batch_reading(self.batch_reading);
        Arc::new(dataloader)
    }
}

impl<B: Backend, I, O> BatchDataloaderIterator<B, I, O> {
    /// Creates a new batch data loader iterator.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The batch strategy.
    /// * `dataset` - The dataset.
    /// * `batcher` - The batcher.
    /// * `device`  - The device to use when loading a batch.
    ///
    /// # Returns
    ///
    /// The batch data loader iterator.
    pub fn new(
        strategy: Box<dyn BatchStrategy<I>>,
        dataset: Arc<dyn Dataset<I>>,
        batcher: Arc<dyn Batcher<B, I, O>>,
        device: B::Device,
        batch_reading: bool,
    ) -> Self {
        BatchDataloaderIterator {
            current_index: 0,
            strategy,
            dataset,
            batcher,
            device,
            batch_reading,
            pending: Vec::new().into_iter(),
            read_indices: Vec::new(),
            read_failed: false,
        }
    }

    fn next_item(&mut self) -> Option<I> {
        if !self.batch_reading {
            return self.dataset.get(self.current_index);
        }

        assert!(
            !self.read_failed,
            "cannot resume a failed batch dataset read"
        );
        if let Some(item) = self.pending.next() {
            return Some(item);
        }

        let len = self.dataset.len();
        if self.current_index >= len {
            return None;
        }
        let count = self.strategy.batch_size().unwrap_or(1).max(1)
            .min(len - self.current_index);
        self.read_indices.clear();
        self.read_indices
            .extend(self.current_index..self.current_index + count);

        self.read_failed = true;
        let items = self.dataset.get_many(&self.read_indices).unwrap_or_else(|| {
            panic!(
                "batch dataset read is missing an item in indices {:?}",
                self.read_indices
            )
        });
        assert_eq!(
            items.len(),
            count,
            "batch dataset read returned the wrong number of items"
        );
        self.pending = items.into_iter();
        self.read_failed = false;
        self.pending.next()
    }
}

impl<B: Backend, I, O> Iterator for BatchDataloaderIterator<B, I, O> {
    type Item = O;

    fn next(&mut self) -> Option<O> {
        while let Some(item) = self.next_item() {
            self.current_index += 1;
            self.strategy.add(item);

            if let Some(items) = self.strategy.batch(false) {
                return Some(self.batcher.batch(items, &self.device));
            }
        }

        if let Some(items) = self.strategy.batch(true) {
            return Some(self.batcher.batch(items, &self.device));
        }

        None
    }
}

impl<B: Backend, I, O> DataLoaderIterator<O> for BatchDataloaderIterator<B, I, O> {
    fn progress(&self) -> Progress {
        Progress::new(self.current_index, self.dataset.len())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::data::dataloader::FixBatchStrategy;
    use crate::data::dataloader::batcher::TestBatcher;
    use crate::data::dataset::FakeDataset;

    #[test]
    fn test_batch_dataloader() {
        let batcher = Arc::new(TestBatcher::new());
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher,
            Default::default(),
            None,
        );

        let mut items_dataset = HashSet::new();
        let mut items_dataloader = HashSet::new();

        for item in dataset.iter() {
            items_dataset.insert(item);
        }

        for items in dataloader.iter() {
            for item in items {
                items_dataloader.insert(item);
            }
        }

        assert_eq!(items_dataset, items_dataloader);
    }

    #[test]
    fn test_batch_dataloader_slice() {
        let batcher = Arc::new(TestBatcher::new());
        let dataset = Arc::new(FakeDataset::<String>::new(27));
        let dataloader = BatchDataLoader::new(
            Box::new(FixBatchStrategy::new(5)),
            dataset.clone(),
            batcher,
            Default::default(),
            None,
        );
        let dataloader_slice = dataloader.slice(5, 15);

        let mut items_dataloader = HashSet::new();
        let mut items_dataloader_slice = HashSet::new();

        let mut idx = 0;
        for items in dataloader.iter() {
            for item in items {
                if (5..15).contains(&idx) {
                    items_dataloader.insert(item);
                }
                idx += 1;
            }
        }

        for items in dataloader_slice.iter() {
            for item in items {
                items_dataloader_slice.insert(item);
            }
        }

        assert_eq!(items_dataloader, items_dataloader_slice);
    }
}
