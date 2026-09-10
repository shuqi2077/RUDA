use alloc::vec::Vec;
use super::data::TensorData;

pub enum ReadbackOrder {
    Float(usize),
    QFloat(usize),
    Int(usize),
    Bool(usize),
}

#[derive(Default)]
/// Contains all [data](TensorData) related to a transaction.
pub struct TransactionData {
    /// Float tensor data.
    pub read_floats: Vec<TensorData>,
    /// Quantized tensor data.
    pub read_qfloats: Vec<TensorData>,
    /// Int tensor data.
    pub read_ints: Vec<TensorData>,
    /// Bool tensor data.
    pub read_bools: Vec<TensorData>,
}


impl TransactionData {
    pub fn into_ordered(self, orders: Vec<ReadbackOrder>) -> Vec<TensorData> {
        let mut floats: Vec<_> = self.read_floats.into_iter().map(Some).collect();
        let mut qfloats: Vec<_> = self.read_qfloats.into_iter().map(Some).collect();
        let mut ints: Vec<_> = self.read_ints.into_iter().map(Some).collect();
        let mut bools: Vec<_> = self.read_bools.into_iter().map(Some).collect();

        orders
            .into_iter()
            .map(|order| match order {
                ReadbackOrder::Float(index) => floats.get_mut(index).unwrap().take().unwrap(),
                ReadbackOrder::QFloat(index) => qfloats.get_mut(index).unwrap().take().unwrap(),
                ReadbackOrder::Int(index) => ints.get_mut(index).unwrap().take().unwrap(),
                ReadbackOrder::Bool(index) => bools.get_mut(index).unwrap().take().unwrap(),
            })
            .collect::<Vec<_>>()
    }
}
