use ruda_core::tensor::{DType, Metadata, data::TensorData};
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{RudaTensor, readback::into_data_sync, transfer::from_data};
use rusparse::{
    CsrMatrix, CsrMatrixOwned, IndexBase, Operation,
    tensor::{CsrTensor, csrgeam, csrgemm, csrmv},
};
use std::collections::BTreeMap;

type Sparse = CsrTensor<CudaRuntime>;
fn upload(matrix: CsrMatrix<'_>) -> Sparse {
    Sparse::from_csr(matrix, Operation::None, &CudaDevice::default()).unwrap()
}
fn values(tensor: RudaTensor<CudaRuntime>) -> Vec<f32> {
    into_data_sync(tensor).to_vec().unwrap()
}
fn base_value(base: IndexBase) -> u32 {
    if base == IndexBase::One { 1 } else { 0 }
}
fn sample(base: IndexBase, columns: usize, counts: &[usize]) -> CsrMatrixOwned {
    let shift = base_value(base);
    let mut offsets = vec![shift];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for (row, &count) in counts.iter().enumerate() {
        for i in 0..count {
            indices.push(((i * 3 + row) % columns) as u32 + shift);
            values.push(((row * 7 + i) as i32 % 13 - 6) as f32 / 8.);
        }
        offsets.push(indices.len() as u32 + shift);
    }
    CsrMatrixOwned::new(counts.len(), columns, offsets, indices, values, base).unwrap()
}
fn entries(matrix: CsrMatrix<'_>, row: usize) -> Vec<(usize, f64)> {
    let base = base_value(matrix.index_base());
    ((matrix.row_offsets()[row] - base) as usize..(matrix.row_offsets()[row + 1] - base) as usize)
        .map(|i| {
            (
                (matrix.column_indices()[i] - base) as usize,
                matrix.values()[i] as f64,
            )
        })
        .collect()
}
fn reference(
    a: CsrMatrix<'_>,
    b: CsrMatrix<'_>,
    alpha: f64,
    beta: f64,
    product: bool,
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let base = base_value(a.index_base());
    let mut offsets = vec![base];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..a.rows() {
        let mut sums = BTreeMap::<usize, f64>::new();
        if product {
            for (inner, av) in entries(a, row) {
                for (column, bv) in entries(b, inner) {
                    *sums.entry(column).or_default() += av * bv;
                }
            }
            for value in sums.values_mut() {
                *value *= alpha;
            }
        } else {
            for (column, av) in entries(a, row) {
                *sums.entry(column).or_default() += alpha * av;
            }
            for (column, bv) in entries(b, row) {
                *sums.entry(column).or_default() += beta * bv;
            }
        }
        for (column, value) in sums {
            indices.push(column as u32 + base);
            values.push(value);
        }
        offsets.push(indices.len() as u32 + base);
    }
    (offsets, indices, values)
}
fn check(
    output: &Sparse,
    rows: usize,
    columns: usize,
    base: IndexBase,
    expected: &(Vec<u32>, Vec<u32>, Vec<f64>),
) {
    assert_eq!(
        (
            output.rows(),
            output.columns(),
            output.nnz(),
            output.index_base()
        ),
        (rows, columns, expected.2.len(), base)
    );
    assert_eq!(
        into_data_sync(output.row_offsets())
            .to_vec::<u32>()
            .unwrap(),
        expected.0
    );
    assert_eq!(
        into_data_sync(output.column_indices())
            .to_vec::<u32>()
            .unwrap(),
        expected.1
    );
    assert_eq!(output.values().dtype, DType::F32);
    let actual = values(output.values());
    assert_eq!(actual.len(), expected.2.len());
    for (&a, &b) in actual.iter().zip(&expected.2) {
        if b.is_nan() {
            assert!(a.is_nan());
        } else if b.is_infinite() {
            assert_eq!(a as f64, b);
        } else {
            assert!((a as f64 - b).abs() <= 2e-5 * b.abs().max(1.), "{a} != {b}");
        }
    }
}

#[test]
fn device_sparse_binary_bases_transposes_duplicates_tails_and_zeros() {
    for base_a in [IndexBase::Zero, IndexBase::One] {
        for base_b in [IndexBase::Zero, IndexBase::One] {
            let a = sample(base_a, 5, &[0, 1, 31, 33, 70]);
            let b = sample(base_b, 5, &[33, 0, 1, 31, 35]);
            let a_device = upload(a.as_ref());
            let b_device = upload(b.as_ref());
            let at = a.as_ref().transpose().unwrap();
            let bt = b.as_ref().transpose().unwrap();
            for op_a in [
                Operation::None,
                Operation::Transpose,
                Operation::ConjugateTranspose,
            ] {
                let a_host = if op_a == Operation::None {
                    a.as_ref()
                } else {
                    at.as_ref()
                };
                for op_b in [
                    Operation::None,
                    Operation::Transpose,
                    Operation::ConjugateTranspose,
                ] {
                    let b_host = if op_b == Operation::None {
                        b.as_ref()
                    } else {
                        bt.as_ref()
                    };
                    for (alpha, beta) in [(0.5, -0.75), (0., 0.), (-1.25, 1.)] {
                        let sum = csrgeam(op_a, op_b, alpha, &a_device, beta, &b_device).unwrap();
                        check(
                            &sum,
                            5,
                            5,
                            base_a,
                            &reference(a_host, b_host, alpha as f64, beta as f64, false),
                        );
                        let product = csrgemm(op_a, op_b, alpha, &a_device, &b_device).unwrap();
                        check(
                            &product,
                            5,
                            5,
                            base_a,
                            &reference(a_host, b_host, alpha as f64, 0., true),
                        );
                    }
                }
            }
            assert_eq!(values(a_device.values()), a.as_ref().values());
            assert_eq!(values(b_device.values()), b.as_ref().values());
        }
    }
}

#[test]
fn device_sparse_binary_rectangular_chaining_and_current_device_values() {
    let a = sample(IndexBase::One, 3, &[1, 4]);
    let b = sample(IndexBase::Zero, 4, &[3, 0, 5]);
    let mut replacement: RudaTensor<CudaRuntime> = from_data(
        TensorData::new(vec![2f32, 99., 3., 99., 4., 99., 5., 99., 6., 99.], [5, 2]),
        &CudaDevice::default(),
    );
    let stride = replacement.meta.strides()[0];
    replacement.meta = Box::new(Metadata::new([5], [stride]));
    let a_device = upload(a.as_ref()).with_values(replacement).unwrap();
    let a_current = CsrMatrix::new(
        2,
        3,
        a.as_ref().row_offsets(),
        a.as_ref().column_indices(),
        &[2., 3., 4., 5., 6.],
        IndexBase::One,
    )
    .unwrap();
    let b_device = upload(b.as_ref());
    let product = csrgemm(Operation::None, Operation::None, 0.5, &a_device, &b_device).unwrap();
    let expected = reference(a_current, b.as_ref(), 0.5, 0., true);
    check(&product, 2, 4, IndexBase::One, &expected);
    let sum = csrgeam(
        Operation::None,
        Operation::None,
        2.,
        &product,
        -1.,
        &product,
    )
    .unwrap();
    check(&sum, 2, 4, IndexBase::One, &expected);
    let host = CsrMatrixOwned::new(
        2,
        4,
        expected.0.clone(),
        expected.1.clone(),
        expected.2.iter().map(|&x| x as f32).collect(),
        IndexBase::One,
    )
    .unwrap();
    let transposed = host.as_ref().transpose().unwrap();
    let device_transposed = sum.transpose().unwrap();
    let t = transposed.as_ref();
    check(
        &device_transposed,
        4,
        2,
        IndexBase::One,
        &(
            t.row_offsets().to_vec(),
            t.column_indices().to_vec(),
            t.values().iter().map(|&x| x as f64).collect(),
        ),
    );
    let roundtrip = device_transposed.transpose().unwrap();
    check(&roundtrip, 2, 4, IndexBase::One, &expected);
    let x = from_data(TensorData::new(vec![1f32; 4], [4]), &CudaDevice::default());
    let y = from_data(TensorData::new(vec![0f32; 2], [2]), &CudaDevice::default());
    let result = values(csrmv(&roundtrip, 1., x, 0., y).unwrap());
    for row in 0..2 {
        assert_eq!(
            result[row] as f64,
            entries(host.as_ref(), row).iter().map(|x| x.1).sum::<f64>()
        );
    }
    let bad = from_data(TensorData::new(vec![1f32], [1]), &CudaDevice::default());
    assert!(a_device.with_values(bad).is_err());
    let bad = from_data(TensorData::new(vec![1u32; 5], [5]), &CudaDevice::default());
    assert!(a_device.with_values(bad).is_err());
    assert!(
        csrgeam(
            Operation::None,
            Operation::None,
            1.,
            &a_device,
            1.,
            &b_device
        )
        .is_err()
    );
    assert!(csrgemm(Operation::None, Operation::None, 1., &b_device, &a_device).is_err());
}

#[test]
fn device_sparse_binary_empty_shapes_and_ieee_values() {
    let a = CsrMatrix::new(1, 1, &[1, 1], &[], &[], IndexBase::One).unwrap();
    let b = CsrMatrix::new(1, u32::MAX as usize, &[0, 0], &[], &[], IndexBase::Zero).unwrap();
    assert!(matches!(
        csrgemm(Operation::None, Operation::None, 1., &upload(a), &upload(b)),
        Err(rusparse::SparseError::SizeOverflow("CSR column index range"))
    ));
    for (m, k, n) in [(0, 3, 2), (2, 0, 3), (2, 3, 0), (2, 3, 4)] {
        let a = sample(IndexBase::One, k, &vec![0; m]);
        let b = sample(IndexBase::Zero, n, &vec![0; k]);
        let output = csrgemm(
            Operation::None,
            Operation::None,
            1.,
            &upload(a.as_ref()),
            &upload(b.as_ref()),
        )
        .unwrap();
        check(
            &output,
            m,
            n,
            IndexBase::One,
            &(vec![1; m + 1], vec![], vec![]),
        );
        let sum = csrgeam(Operation::None, Operation::None, 1., &output, 1., &output).unwrap();
        check(
            &sum,
            m,
            n,
            IndexBase::One,
            &(vec![1; m + 1], vec![], vec![]),
        );
        let t = sum.transpose().unwrap();
        check(&t, n, m, IndexBase::One, &(vec![1; n + 1], vec![], vec![]));
    }
    let a = CsrMatrix::new(
        1,
        2,
        &[0, 2],
        &[0, 1],
        &[f32::INFINITY, f32::NAN],
        IndexBase::Zero,
    )
    .unwrap();
    let b = CsrMatrix::new(1, 2, &[1, 2], &[1], &[1.], IndexBase::One).unwrap();
    let output = csrgeam(
        Operation::None,
        Operation::None,
        0.,
        &upload(a),
        1.,
        &upload(b),
    )
    .unwrap();
    check(
        &output,
        1,
        2,
        IndexBase::Zero,
        &reference(a, b, 0., 1., false),
    );
    let b = CsrMatrix::new(2, 2, &[0, 1, 2], &[0, 1], &[1., 1.], IndexBase::Zero).unwrap();
    let output = csrgemm(Operation::None, Operation::None, 1., &upload(a), &upload(b)).unwrap();
    check(
        &output,
        1,
        2,
        IndexBase::Zero,
        &reference(a, b, 1., 0., true),
    );
}
