use std::sync::Arc;

use crate::DatasetIterator;

/// The dataset trait defines a basic collection of items with a predefined size.
pub trait Dataset<I>: Send + Sync {
    /// Gets the item at the given index.
    fn get(&self, index: usize) -> Option<I>;

    /// Fetch actual items in requested order, preserving repeated indices.
    /// None means at least one requested item is absent; no filler is returned.
    /// Storage-backed implementations may combine I/O without changing order.
    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        indices.iter().map(|&index| self.get(index)).collect()
    }

    /// Gets the number of items in the dataset.
    fn len(&self) -> usize;

    /// Checks if the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator over the dataset.
    fn iter(&self) -> DatasetIterator<'_, I>
    where
        Self: Sized,
    {
        DatasetIterator::new(self)
    }
}

impl<D, I> Dataset<I> for Arc<D>
where
    D: Dataset<I>,
{
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        self.as_ref().get_many(indices)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I> Dataset<I> for Arc<dyn Dataset<I>> {
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        self.as_ref().get_many(indices)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<D, I> Dataset<I> for Box<D>
where
    D: Dataset<I>,
{
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        self.as_ref().get_many(indices)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

impl<I> Dataset<I> for Box<dyn Dataset<I>> {
    fn get(&self, index: usize) -> Option<I> {
        self.as_ref().get(index)
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        self.as_ref().get_many(indices)
    }

    fn len(&self) -> usize {
        self.as_ref().len()
    }
}
