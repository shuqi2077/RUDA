use ruda_core::tensor::{DType, Metadata, data::TensorData};
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{RudaTensor, readback::into_data_sync, transfer::from_data};
use rusparse::{
    CsrMatrix, CsrMatrixOwned, DenseOrder, IndexBase, Operation,
    tensor::{CsrTensor, csrmm, csrmv},
};

fn upload(values: Vec<f32>, shape: Vec<usize>) -> RudaTensor<CudaRuntime> {
    from_data(TensorData::new(values, shape), &CudaDevice::default())
}
fn read(tensor: RudaTensor<CudaRuntime>) -> Vec<f32> {
    into_data_sync(tensor).to_vec().unwrap()
}
fn vector(values: &[f32], strided: bool) -> RudaTensor<CudaRuntime> {
    if !strided {
        return upload(values.to_vec(), vec![values.len()]);
    }
    let mut tensor = upload(
        values.iter().flat_map(|&x| [x, -99.]).collect(),
        vec![values.len(), 2],
    );
    let stride = tensor.meta.strides()[0];
    tensor.meta = Box::new(Metadata::new([values.len()], [stride]));
    tensor
}
fn matrix(base: IndexBase) -> CsrMatrixOwned {
    let start = if base == IndexBase::One { 1 } else { 0 };
    let mut offsets = vec![start];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for (row, count) in [0, 1, 31, 33, 70].into_iter().enumerate() {
        for column in 0..count {
            indices.push(column + start);
            values.push(((row as i32 * 3 + column as i32) % 13 - 6) as f32 / 8.);
        }
        offsets.push(indices.len() as u32 + start);
    }
    CsrMatrixOwned::new(5, 70, offsets, indices, values, base).unwrap()
}
fn dense(matrix: CsrMatrix<'_>) -> Vec<f32> {
    rusparse::conversion::csr_to_dense(matrix, DenseOrder::RowMajor)
        .unwrap()
        .as_ref()
        .values()
        .to_vec()
}
fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!((a as f64 - b).abs() < 2e-5 * b.abs().max(1.), "{a} != {b}");
    }
}

#[test]
fn csr_device_mv_index_bases_transposes_tails_and_input_preservation() {
    for base in [IndexBase::Zero, IndexBase::One] {
        let original = matrix(base);
        for operation in [
            Operation::None,
            Operation::Transpose,
            Operation::ConjugateTranspose,
        ] {
            let transpose = original.as_ref().transpose().unwrap();
            let host = if operation == Operation::None {
                original.as_ref()
            } else {
                transpose.as_ref()
            };
            let matrix = CsrTensor::<CudaRuntime>::from_csr(
                original.as_ref(),
                operation,
                &CudaDevice::default(),
            )
            .unwrap();
            assert_eq!(
                (
                    matrix.rows(),
                    matrix.columns(),
                    matrix.nnz(),
                    matrix.index_base()
                ),
                (host.rows(), host.columns(), host.nnz(), base)
            );
            let a = dense(host);
            let x = (0..host.columns())
                .map(|i| (i as i32 % 11 - 5) as f32 / 4.)
                .collect::<Vec<_>>();
            let y = (0..host.rows()).map(|i| i as f32 / 8.).collect::<Vec<_>>();
            let expected = (0..host.rows())
                .map(|r| {
                    1.25 * (0..host.columns())
                        .map(|k| a[r * host.columns() + k] as f64 * x[k] as f64)
                        .sum::<f64>()
                        - 0.5 * y[r] as f64
                })
                .collect::<Vec<_>>();
            for strided in [false, true] {
                let x_device = vector(&x, strided);
                let y_device = vector(&y, strided);
                assert_eq!(read(x_device.clone()), x);
                assert_eq!(read(y_device.clone()), y);
                for _ in 0..2 {
                    let output =
                        csrmv(&matrix, 1.25, x_device.clone(), -0.5, y_device.clone()).unwrap();
                    assert_eq!(output.dtype, DType::F32);
                    close(&read(output), &expected);
                }
                assert_eq!(read(x_device), x);
                assert_eq!(read(y_device), y);
            }
        }
    }
}

#[test]
fn csr_device_mm_dense_transpose_strides_output_orders_and_beta() {
    for base in [IndexBase::Zero, IndexBase::One] {
        let original = matrix(base);
        for op_a in [Operation::None, Operation::Transpose] {
            let transpose = original.as_ref().transpose().unwrap();
            let host = if op_a == Operation::None {
                original.as_ref()
            } else {
                transpose.as_ref()
            };
            let a = dense(host);
            let matrix =
                CsrTensor::<CudaRuntime>::from_csr(original.as_ref(), op_a, &CudaDevice::default())
                    .unwrap();
            let n = 3;
            let b = (0..host.columns() * n)
                .map(|i| (i as i32 % 17 - 8) as f32 / 8.)
                .collect::<Vec<_>>();
            let c = (0..host.rows() * n)
                .map(|i| i as f32 / 4.)
                .collect::<Vec<_>>();
            for op_b in [
                Operation::None,
                Operation::Transpose,
                Operation::ConjugateTranspose,
            ] {
                let transposed_b = (0..n)
                    .flat_map(|column| {
                        (0..host.columns())
                            .map(|row| b[row * n + column])
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let mut b_device = upload(transposed_b, vec![n, host.columns()]);
                if op_b == Operation::None {
                    b_device.meta.swap(0, 1);
                }
                for order in [DenseOrder::RowMajor, DenseOrder::ColumnMajor] {
                    for has_c in [false, true] {
                        let transposed_c = (0..n)
                            .flat_map(|column| {
                                (0..host.rows())
                                    .map(|row| c[row * n + column])
                                    .collect::<Vec<_>>()
                            })
                            .collect();
                        let mut c_device = upload(transposed_c, vec![n, host.rows()]);
                        c_device.meta.swap(0, 1);
                        let output = csrmm(
                            &matrix,
                            op_b,
                            0.75,
                            b_device.clone(),
                            -0.25,
                            has_c.then(|| c_device.clone()),
                            order,
                        )
                        .unwrap();
                        assert_eq!(output.meta.shape()[..], [host.rows(), n]);
                        assert_eq!(
                            output.meta.strides()[..],
                            if order == DenseOrder::ColumnMajor {
                                [1, host.rows()]
                            } else {
                                [n, 1]
                            }
                        );
                        let expected = (0..host.rows() * n)
                            .map(|i| {
                                0.75 * (0..host.columns())
                                    .map(|k| {
                                        a[i / n * host.columns() + k] as f64
                                            * b[k * n + i % n] as f64
                                    })
                                    .sum::<f64>()
                                    - if has_c { 0.25 * c[i] as f64 } else { 0. }
                            })
                            .collect::<Vec<_>>();
                        close(&read(output), &expected);
                        assert_eq!(read(c_device), c);
                    }
                }
            }
        }
    }
}

#[test]
fn csr_device_validation_empty_shapes_and_special_values() {
    let device = CudaDevice::default();
    let empty = CsrMatrix::new(0, 2, &[1], &[], &[], IndexBase::One).unwrap();
    let empty = CsrTensor::<CudaRuntime>::from_csr(empty, Operation::None, &device).unwrap();
    assert!(
        read(csrmv(&empty, 1., vector(&[1., 2.], false), 1., vector(&[], false)).unwrap())
            .is_empty()
    );
    let special = CsrMatrix::new(
        3,
        2,
        &[0, 0, 1, 2],
        &[0, 1],
        &[f32::INFINITY, 1.],
        IndexBase::Zero,
    )
    .unwrap();
    let special = CsrTensor::<CudaRuntime>::from_csr(special, Operation::None, &device).unwrap();
    let output = read(
        csrmv(
            &special,
            1.,
            vector(&[1., f32::NAN], false),
            0.,
            vector(&[f32::NAN, 0., 0.], false),
        )
        .unwrap(),
    );
    assert!(output[0].is_nan() && output[1].is_infinite() && output[2].is_nan());
    assert!(
        csrmv(
            &special,
            1.,
            vector(&[1.], false),
            0.,
            vector(&[0.; 3], false)
        )
        .is_err()
    );
    let integers = from_data::<CudaRuntime>(TensorData::new(vec![1u32, 2], [2]), &device);
    assert!(csrmv(&special, 1., integers, 0., vector(&[0.; 3], false)).is_err());
    assert!(
        csrmm(
            &special,
            Operation::None,
            1.,
            upload(vec![1.; 4], vec![2, 2]),
            0.,
            Some(upload(vec![0.; 4], vec![2, 2])),
            DenseOrder::RowMajor
        )
        .is_err()
    );
}

#[test]
fn cuda_empty_tensor_transfer_preserves_shape_and_queue() {
    for shape in [vec![0], vec![0, 2], vec![2, 0], vec![2, 0, 3]] {
        for value in [0.0f32, 1.0, -2.5] {
            let tensor = ruda_kernel::tensor::initialization::full::<CudaRuntime, f32>(
                shape.clone().into(), &CudaDevice::default(), value,
            );
            let data = into_data_sync(tensor);
            assert_eq!(data.shape[..], shape[..]);
            assert_eq!(data.dtype, DType::F32);
            assert!(data.to_vec::<f32>().unwrap().is_empty());
        }
        for _ in 0..3 {
            let tensor = upload(Vec::new(), shape.clone());
            let data = into_data_sync(tensor);
            assert_eq!(data.shape[..], shape[..]);
            assert_eq!(data.dtype, DType::F32);
            assert_eq!(data.bytes.as_ptr().align_offset(16), 0);
            assert!(data.to_vec::<f32>().unwrap().is_empty());
            if shape.len() > 1 {
                let mut tensor = upload(Vec::new(), shape.clone());
                tensor.meta.swap(0, 1);
                let expected_shape = tensor.meta.shape().clone();
                let tensor = ruda_kernel::tensor::contiguous::into_contiguous(tensor);
                assert!(tensor.is_contiguous());
                let data = into_data_sync(tensor);
                assert_eq!(data.shape, expected_shape);
                assert!(data.to_vec::<f32>().unwrap().is_empty());
            }
        }
    }
    assert_eq!(read(vector(&[1., -2., 3.], false)), vec![1., -2., 3.]);
}

#[test]
fn csr_device_dense_conversion_and_backward_preserve_duplicate_overwrite_semantics() {
    use rusparse::tensor::{csr_to_dense, csr_to_dense_backward};
    let device = CudaDevice::default();
    for base in [IndexBase::Zero, IndexBase::One] {
        let shift = if base == IndexBase::One { 1 } else { 0 };
        let offsets = [0, 3, 3, 5].map(|x| x + shift);
        let indices = [2, 0, 2, 3, 1].map(|x| x + shift);
        let original = [1., 2., 7., -4., 5.];
        let host = CsrMatrix::new(3, 4, &offsets, &indices, &original, base).unwrap();
        let matrix = CsrTensor::<CudaRuntime>::from_csr(host, Operation::None, &device).unwrap();
        for order in [DenseOrder::RowMajor, DenseOrder::ColumnMajor] {
            let dense = csr_to_dense(&matrix, order).unwrap();
            assert_eq!(dense.meta.shape()[..], [3, 4]);
            assert_eq!(read(dense), [2., 0., 7., 0., 0., 0., 0., 0., 0., 5., 0., -4.], "{base:?} {order:?}");
            let grad = match order {
                DenseOrder::RowMajor => upload(
                    vec![10., 11., 12., 13., 20., 21., 22., 23., 30., 31., 32., 33.], vec![3, 4],
                ),
                DenseOrder::ColumnMajor => {
                    let mut grad = upload(
                        vec![10., 20., 30., 11., 21., 31., 12., 22., 32., 13., 23., 33.], vec![4, 3],
                    );
                    grad.meta.swap(0, 1);
                    grad
                }
            };
            assert_eq!(read(csr_to_dense_backward(&matrix, grad.clone()).unwrap()), [0., 10., 12., 33., 31.]);
            assert_eq!(read(grad), [10., 11., 12., 13., 20., 21., 22., 23., 30., 31., 32., 33.]);
        }
        assert_eq!(read(matrix.values()), original);
        let odd_offsets = [0, 1, 2].map(|x| x + shift);
        let odd_indices = [1, 2].map(|x| x + shift);
        let odd = CsrMatrix::new(2, 3, &odd_offsets, &odd_indices, &[6., 9.], base).unwrap();
        let odd = CsrTensor::<CudaRuntime>::from_csr(odd, Operation::None, &device).unwrap();
        for order in [DenseOrder::RowMajor, DenseOrder::ColumnMajor] {
            assert_eq!(read(csr_to_dense(&odd, order).unwrap()), [0., 6., 0., 0., 0., 9.], "{base:?} {order:?}");
        }
        for (rows, columns) in [(0, 4), (3, 0), (3, 4)] {
            let offsets = vec![shift; rows + 1];
            let empty = CsrMatrix::new(rows, columns, &offsets, &[], &[], base).unwrap();
            let empty = CsrTensor::<CudaRuntime>::from_csr(empty, Operation::None, &device).unwrap();
            for order in [DenseOrder::RowMajor, DenseOrder::ColumnMajor] {
                let dense = csr_to_dense(&empty, order).unwrap();
                assert_eq!(dense.meta.shape()[..], [rows, columns]);
                assert_eq!(read(dense), vec![0.; rows * columns]);
            }
            let grad = upload(vec![1.; rows * columns], vec![rows, columns]);
            assert!(read(csr_to_dense_backward(&empty, grad).unwrap()).is_empty());
        }
    }
}
