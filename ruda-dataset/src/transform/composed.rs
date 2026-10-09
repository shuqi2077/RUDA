use crate::Dataset;

/// Compose multiple datasets together to create a bigger one.
#[derive(new)]
pub struct ComposedDataset<D> {
    datasets: Vec<D>,
}

impl<D, I> Dataset<I> for ComposedDataset<D>
where
    D: Dataset<I>,
    I: Clone,
{
    fn get(&self, index: usize) -> Option<I> {
        let mut current_index = 0;
        for dataset in self.datasets.iter() {
            if index < dataset.len() + current_index {
                return dataset.get(index - current_index);
            }
            current_index += dataset.len();
        }
        None
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        if indices.is_empty() {
            return Some(Vec::new());
        }

        let mut offset = 0;
        let ends: Vec<_> = self.datasets
            .iter()
            .map(|dataset| {
                offset += dataset.len();
                offset
            })
            .collect();
        let translated: Vec<_> = indices
            .iter()
            .map(|&index| {
                let dataset = ends.partition_point(|&end| end <= index);
                if dataset == self.datasets.len() {
                    return None;
                }
                let start = if dataset == 0 { 0 } else { ends[dataset - 1] };
                Some((dataset, index - start))
            })
            .collect::<Option<_>>()?;

        let mut items = Vec::with_capacity(indices.len());
        let mut local_indices = Vec::new();
        let mut start = 0;
        while start < translated.len() {
            let dataset = translated[start].0;
            let mut end = start + 1;
            while end < translated.len() && translated[end].0 == dataset {
                end += 1;
            }
            local_indices.clear();
            local_indices.extend(translated[start..end].iter().map(|&(_, index)| index));
            let batch = self.datasets[dataset].get_many(&local_indices)?;
            if batch.len() != local_indices.len() {
                return None;
            }
            items.extend(batch);
            start = end;
        }
        Some(items)
    }

    fn len(&self) -> usize {
        let mut total = 0;
        for dataset in self.datasets.iter() {
            total += dataset.len();
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FakeDataset;

    #[test]
    fn test_composed_dataset() {
        let dataset1 = FakeDataset::<String>::new(10);
        let dataset2 = FakeDataset::<String>::new(5);

        let items1 = dataset1.iter().collect::<Vec<_>>();
        let items2 = dataset2.iter().collect::<Vec<_>>();

        let composed = ComposedDataset::new(vec![dataset1, dataset2]);

        assert_eq!(composed.len(), 15);

        let expected_items: Vec<String> = items1.iter().chain(items2.iter()).cloned().collect();

        let items = composed.iter().collect::<Vec<_>>();

        assert_eq!(items, expected_items);
    }
}
