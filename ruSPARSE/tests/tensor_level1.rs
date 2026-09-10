use ruda_core::tensor::{DType, Metadata, data::TensorData};
use ruda_driver_cuda::{CudaDevice, CudaRuntime};
use ruda_kernel::tensor::{RudaTensor, readback::into_data_sync, transfer::from_data};
use rusparse::{
    CsrMatrix, IndexBase, Operation, SparseVector,
    tensor::{CsrTensor, SparseVectorTensor, sddmm, spvv},
};

type Tensor = RudaTensor<CudaRuntime>;
fn upload(values: Vec<f32>, shape: impl Into<Vec<usize>>) -> Tensor {
    from_data(
        TensorData::new(values, shape.into()),
        &CudaDevice::default(),
    )
}
fn read(tensor: Tensor) -> Vec<f32> {
    into_data_sync(tensor).to_vec().unwrap()
}
fn close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &b) in actual.iter().zip(expected) {
        assert!((a as f64 - b).abs() <= 2e-5 * b.abs().max(1.), "{a} != {b}");
    }
}

#[test]
fn device_spvv_duplicates_unsorted_tails_strides_and_empty() {
    for base in [IndexBase::Zero, IndexBase::One] {
        let shift = if base == IndexBase::One { 1 } else { 0 };
        for nnz in [0, 1, 31, 32, 33, 129] {
            let indices = (0..nnz)
                .map(|i| (i * 17 % 71) as u32 + shift)
                .collect::<Vec<_>>();
            let values = (0..nnz)
                .map(|i| (i as i32 % 13 - 6) as f32 / 4.)
                .collect::<Vec<_>>();
            let host = SparseVector::new(71, &indices, &values, base).unwrap();
            let x = SparseVectorTensor::<CudaRuntime>::from_sparse(host, &CudaDevice::default())
                .unwrap();
            assert_eq!((x.size(), x.nnz(), x.index_base()), (71, nnz, base));
            let y = (0..71).map(|i| (i % 9 - 4) as f32 / 8.).collect::<Vec<_>>();
            let expected = indices
                .iter()
                .zip(&values)
                .map(|(&i, &v)| v as f64 * y[(i - shift) as usize] as f64)
                .sum::<f64>();
            for strided in [false, true] {
                let dense = if strided {
                    let mut tensor = upload(y.iter().flat_map(|&v| [v, -99.]).collect(), [71, 2]);
                    let stride = tensor.meta.strides()[0];
                    tensor.meta = Box::new(Metadata::new([71], [stride]));
                    tensor
                } else {
                    upload(y.clone(), [71])
                };
                for operation in [
                    Operation::None,
                    Operation::Transpose,
                    Operation::ConjugateTranspose,
                ] {
                    let output = spvv(operation, &x, dense.clone()).unwrap();
                    assert_eq!(output.dtype, DType::F32);
                    assert_eq!(output.meta.shape()[..], [1]);
                    close(&read(output), &[expected]);
                }
                assert_eq!(read(dense), y);
            }
        }
    }
    let empty = SparseVector::new(0, &[], &[], IndexBase::Zero).unwrap();
    let empty =
        SparseVectorTensor::<CudaRuntime>::from_sparse(empty, &CudaDevice::default()).unwrap();
    assert_eq!(
        read(spvv(Operation::None, &empty, upload(vec![], [0])).unwrap()),
        [0.]
    );
    assert!(spvv(Operation::None, &empty, upload(vec![1.], [1])).is_err());
}

fn dense_operand(
    values: &[f32],
    rows: usize,
    columns: usize,
    operation: Operation,
    strided: bool,
) -> Tensor {
    let mut tensor = if strided {
        let transposed = (0..columns)
            .flat_map(|c| (0..rows).map(move |r| values[r * columns + c]))
            .collect();
        let mut tensor = upload(transposed, [columns, rows]);
        tensor.meta.swap(0, 1);
        tensor
    } else {
        upload(values.to_vec(), [rows, columns])
    };
    if operation != Operation::None {
        tensor.meta.swap(0, 1);
    }
    tensor
}

#[test]
fn device_sddmm_preserves_structure_inputs_transposes_and_empty_inner() {
    for base in [IndexBase::Zero, IndexBase::One] {
        let shift = if base == IndexBase::One { 1 } else { 0 };
        let offsets = [0, 0, 2, 2, 5, 6, 6].map(|i| i + shift);
        let indices = [4, 1, 3, 3, 0, 2].map(|i| i + shift);
        let values = [1., -2., 3., 4., -5., 6.];
        let pattern = CsrMatrix::new(6, 5, &offsets, &indices, &values, base).unwrap();
        let c =
            CsrTensor::<CudaRuntime>::from_csr(pattern, Operation::None, &CudaDevice::default())
                .unwrap();
        for inner in [0, 1, 31, 32, 33, 70] {
            let a = (0..6 * inner)
                .map(|i| (i as i32 % 11 - 5) as f32 / 8.)
                .collect::<Vec<_>>();
            let b = (0..inner * 5)
                .map(|i| (i as i32 % 13 - 6) as f32 / 4.)
                .collect::<Vec<_>>();
            let entry_rows = [1, 1, 3, 3, 3, 4];
            let expected = (0..6)
                .map(|e| {
                    0.75 * (0..inner)
                        .map(|k| {
                            a[entry_rows[e] * inner + k] as f64
                                * b[k * 5 + (indices[e] - shift) as usize] as f64
                        })
                        .sum::<f64>()
                        - 0.5 * values[e] as f64
                })
                .collect::<Vec<_>>();
            for op_a in [
                Operation::None,
                Operation::Transpose,
                Operation::ConjugateTranspose,
            ] {
                for op_b in [
                    Operation::None,
                    Operation::Transpose,
                    Operation::ConjugateTranspose,
                ] {
                    for strided in [false, true] {
                        let a_device = dense_operand(&a, 6, inner, op_a, strided);
                        let b_device = dense_operand(&b, inner, 5, op_b, !strided);
                        let before_a = read(a_device.clone());
                        let before_b = read(b_device.clone());
                        let output = sddmm(
                            op_a,
                            op_b,
                            0.75,
                            a_device.clone(),
                            b_device.clone(),
                            -0.5,
                            &c,
                        )
                        .unwrap();
                        assert_eq!(
                            (
                                output.rows(),
                                output.columns(),
                                output.nnz(),
                                output.index_base()
                            ),
                            (6, 5, 6, base)
                        );
                        assert_eq!(
                            into_data_sync(output.row_offsets())
                                .to_vec::<u32>()
                                .unwrap(),
                            offsets
                        );
                        assert_eq!(
                            into_data_sync(output.column_indices())
                                .to_vec::<u32>()
                                .unwrap(),
                            indices
                        );
                        close(&read(output.values()), &expected);
                        assert_eq!(read(c.values()), values);
                        assert_eq!(read(a_device), before_a);
                        assert_eq!(read(b_device), before_b);
                    }
                }
            }
        }
    }
}

#[test]
fn device_sparse_level1_special_values_empty_patterns_and_validation() {
    let device = CudaDevice::default();
    let sparse = SparseVector::new(2, &[1], &[f32::INFINITY], IndexBase::Zero).unwrap();
    let sparse = SparseVectorTensor::<CudaRuntime>::from_sparse(sparse, &device).unwrap();
    assert!(
        read(spvv(Operation::None, &sparse, upload(vec![0., 1.], [2])).unwrap())[0].is_infinite()
    );
    assert!(read(spvv(Operation::None, &sparse, upload(vec![0., 0.], [2])).unwrap())[0].is_nan());
    let integers = from_data::<CudaRuntime>(TensorData::new(vec![1u32, 2], [2]), &device);
    assert!(spvv(Operation::None, &sparse, integers).is_err());
    let pattern = CsrMatrix::new(1, 1, &[0, 1], &[0], &[f32::NAN], IndexBase::Zero).unwrap();
    let c = CsrTensor::<CudaRuntime>::from_csr(pattern, Operation::None, &device).unwrap();
    let output = sddmm(
        Operation::None,
        Operation::None,
        1.,
        upload(vec![1.], [1, 1]),
        upload(vec![2.], [1, 1]),
        0.,
        &c,
    )
    .unwrap();
    assert!(read(output.values())[0].is_nan());
    assert!(
        sddmm(
            Operation::None,
            Operation::None,
            1.,
            upload(vec![1.], [1]),
            upload(vec![1.], [1, 1]),
            0.,
            &c
        )
        .is_err()
    );
    assert!(
        sddmm(
            Operation::None,
            Operation::None,
            1.,
            upload(vec![1., 2.], [1, 2]),
            upload(vec![1.], [1, 1]),
            0.,
            &c
        )
        .is_err()
    );
    let integers = from_data::<CudaRuntime>(TensorData::new(vec![1u32], [1, 1]), &device);
    assert!(
        sddmm(
            Operation::None,
            Operation::None,
            1.,
            integers,
            upload(vec![1.], [1, 1]),
            0.,
            &c
        )
        .is_err()
    );
    let empty = CsrMatrix::new(2, 3, &[1, 1, 1], &[], &[], IndexBase::One).unwrap();
    let empty = CsrTensor::<CudaRuntime>::from_csr(empty, Operation::None, &device).unwrap();
    let output = sddmm(
        Operation::None,
        Operation::None,
        1.,
        upload(vec![1.; 8], [2, 4]),
        upload(vec![1.; 12], [4, 3]),
        0.,
        &empty,
    )
    .unwrap();
    assert_eq!(
        (
            output.rows(),
            output.columns(),
            output.nnz(),
            output.index_base()
        ),
        (2, 3, 0, IndexBase::One)
    );
    assert!(read(output.values()).is_empty());
}
