use super::TensorData;
use crate::tensor::element::Element;
use alloc::vec::Vec;

impl<E: Element, const A: usize> From<[E; A]> for TensorData {
    fn from(elems: [E; A]) -> Self {
        TensorData::new(elems.to_vec(), [A])
    }
}

impl<const A: usize> From<[usize; A]> for TensorData {
    fn from(elems: [usize; A]) -> Self {
        TensorData::new(elems.iter().map(|&e| e as i64).collect(), [A])
    }
}

impl From<&[usize]> for TensorData {
    fn from(elems: &[usize]) -> Self {
        let mut data = Vec::with_capacity(elems.len());
        for elem in elems.iter() {
            data.push(*elem as i64);
        }

        TensorData::new(data, [elems.len()])
    }
}

impl<E: Element> From<&[E]> for TensorData {
    fn from(elems: &[E]) -> Self {
        let mut data = Vec::with_capacity(elems.len());
        for elem in elems.iter() {
            data.push(*elem);
        }

        TensorData::new(data, [elems.len()])
    }
}

impl<E: Element, const A: usize, const B: usize> From<[[E; B]; A]> for TensorData {
    fn from(elems: [[E; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B);
        for elem in elems.into_iter().take(A) {
            for elem in elem.into_iter().take(B) {
                data.push(elem);
            }
        }

        TensorData::new(data, [A, B])
    }
}

impl<E: Element, const A: usize, const B: usize, const C: usize> From<[[[E; C]; B]; A]>
    for TensorData
{
    fn from(elems: [[[E; C]; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B * C);

        for elem in elems.into_iter().take(A) {
            for elem in elem.into_iter().take(B) {
                for elem in elem.into_iter().take(C) {
                    data.push(elem);
                }
            }
        }

        TensorData::new(data, [A, B, C])
    }
}

impl<E: Element, const A: usize, const B: usize, const C: usize, const D: usize>
    From<[[[[E; D]; C]; B]; A]> for TensorData
{
    fn from(elems: [[[[E; D]; C]; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B * C * D);

        for elem in elems.into_iter().take(A) {
            for elem in elem.into_iter().take(B) {
                for elem in elem.into_iter().take(C) {
                    for elem in elem.into_iter().take(D) {
                        data.push(elem);
                    }
                }
            }
        }

        TensorData::new(data, [A, B, C, D])
    }
}

impl<Elem: Element, const A: usize, const B: usize, const C: usize, const D: usize, const E: usize>
    From<[[[[[Elem; E]; D]; C]; B]; A]> for TensorData
{
    fn from(elems: [[[[[Elem; E]; D]; C]; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B * C * D * E);

        for elem in elems.into_iter().take(A) {
            for elem in elem.into_iter().take(B) {
                for elem in elem.into_iter().take(C) {
                    for elem in elem.into_iter().take(D) {
                        for elem in elem.into_iter().take(E) {
                            data.push(elem);
                        }
                    }
                }
            }
        }

        TensorData::new(data, [A, B, C, D, E])
    }
}
