use rusparse::conversion as sparse;
use rusparse::{DenseMatrix, DenseOrder, IndexBase};

const ROW_MAJOR: [f32; 12] = [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 3.0, 0.0, 4.0];
const COLUMN_MAJOR: [f32; 12] = [1.0, 0.0, -1.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 4.0];

#[test]
fn dense_to_csr_coo_and_csc_follow_rocsparse_ordering_and_index_base() {
    let dense = DenseMatrix::new(&COLUMN_MAJOR, 3, 4, DenseOrder::ColumnMajor).unwrap();

    let csr = sparse::dense_to_csr(dense, IndexBase::One).unwrap();
    assert_eq!(csr.as_ref().row_offsets(), &[1, 3, 3, 6]);
    assert_eq!(csr.as_ref().column_indices(), &[1, 3, 1, 2, 4]);
    assert_eq!(csr.as_ref().values(), &[1.0, 2.0, -1.0, 3.0, 4.0]);

    let coo = sparse::dense_to_coo(dense, IndexBase::One).unwrap();
    assert_eq!(coo.as_ref().row_indices(), &[1, 1, 3, 3, 3]);
    assert_eq!(coo.as_ref().column_indices(), &[1, 3, 1, 2, 4]);
    assert_eq!(coo.as_ref().values(), &[1.0, 2.0, -1.0, 3.0, 4.0]);

    let csc = sparse::dense_to_csc(dense, IndexBase::One).unwrap();
    assert_eq!(csc.as_ref().column_offsets(), &[1, 3, 4, 5, 6]);
    assert_eq!(csc.as_ref().row_indices(), &[1, 3, 3, 1, 3]);
    assert_eq!(csc.as_ref().values(), &[1.0, -1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn csr_coo_and_csc_round_trip_to_both_dense_orders() {
    let dense = DenseMatrix::new(&ROW_MAJOR, 3, 4, DenseOrder::RowMajor).unwrap();
    let csr = sparse::dense_to_csr(dense, IndexBase::Zero).unwrap();
    let coo = sparse::dense_to_coo(dense, IndexBase::Zero).unwrap();
    let csc = sparse::dense_to_csc(dense, IndexBase::Zero).unwrap();

    assert_eq!(
        sparse::csr_to_dense(csr.as_ref(), DenseOrder::RowMajor)
            .unwrap()
            .as_ref()
            .values(),
        &ROW_MAJOR
    );
    assert_eq!(
        sparse::coo_to_dense(coo.as_ref(), DenseOrder::ColumnMajor)
            .unwrap()
            .as_ref()
            .values(),
        &COLUMN_MAJOR
    );
    assert_eq!(
        sparse::csc_to_dense(csc.as_ref(), DenseOrder::ColumnMajor)
            .unwrap()
            .as_ref()
            .values(),
        &COLUMN_MAJOR
    );
}

#[test]
fn conversions_preserve_empty_shapes() {
    let dense = DenseMatrix::new(&[], 0, 3, DenseOrder::RowMajor).unwrap();
    let csr = sparse::dense_to_csr(dense, IndexBase::One).unwrap();
    assert_eq!(csr.as_ref().row_offsets(), &[1]);
    assert!(csr.as_ref().values().is_empty());
    let dense = sparse::csr_to_dense(csr.as_ref(), DenseOrder::ColumnMajor).unwrap();
    assert_eq!((dense.as_ref().rows(), dense.as_ref().columns()), (0, 3));
    assert!(dense.as_ref().values().is_empty());
}
