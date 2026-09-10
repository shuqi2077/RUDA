use crate::Dataset;

use rayon::prelude::*;
use rstest::{fixture, rstest};
use serde::{Deserialize, Serialize};
use tempfile::{NamedTempFile, TempDir, tempdir};

use super::*;

type SqlDs = SqliteDataset<Sample>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sample {
    column_str: String,
    column_bytes: Vec<u8>,
    column_int: i64,
    column_bool: bool,
    column_float: f64,
}

#[fixture]
fn train_dataset() -> SqlDs {
    SqliteDataset::<Sample>::from_db_file("tests/data/sqlite-dataset.db", "train").unwrap()
}

#[rstest]
pub fn len(train_dataset: SqlDs) {
    assert_eq!(train_dataset.len(), 2);
}

#[rstest]
pub fn get_some(train_dataset: SqlDs) {
    let item = train_dataset.get(0).unwrap();
    assert_eq!(item.column_str, "HI1");
    assert_eq!(item.column_bytes, vec![55, 231, 159]);
    assert_eq!(item.column_int, 1);
    assert!(item.column_bool);
    assert_eq!(item.column_float, 1.0);
}

#[rstest]
pub fn get_none(train_dataset: SqlDs) {
    assert_eq!(train_dataset.get(10), None);
}

#[rstest]
pub fn multi_thread(train_dataset: SqlDs) {
    let indices: Vec<usize> = vec![0, 1, 1, 3, 4, 5, 6, 0, 8, 1];
    let results: Vec<Option<Sample>> = indices.par_iter().map(|&i| train_dataset.get(i)).collect();

    let mut match_count = 0;
    for (_index, result) in indices.iter().zip(results.iter()) {
        if let Some(_val) = result {
            match_count += 1
        }
    }

    assert_eq!(match_count, 5);
}

#[test]
fn sqlite_dataset_storage() {
    // Test with non-existing file
    let storage = SqliteDatasetStorage::from_file("non-existing.db");
    assert!(!storage.exists());

    // Test with non-existing name
    let storage = SqliteDatasetStorage::from_name("non-existing.db");
    assert!(!storage.exists());

    // Test with existing file
    let storage = SqliteDatasetStorage::from_file("tests/data/sqlite-dataset.db");
    assert!(storage.exists());
    let result = storage.reader::<Sample>("train");
    assert!(result.is_ok());
    let train = result.unwrap();
    assert_eq!(train.len(), 2);

    // Test get writer
    let temp_file = NamedTempFile::new().unwrap();
    let storage = SqliteDatasetStorage::from_file(temp_file.path());
    assert!(storage.exists());
    let result = storage.writer::<Sample>(true);
    assert!(result.is_ok());
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Complex {
    column_str: String,
    column_bytes: Vec<u8>,
    column_int: i64,
    column_bool: bool,
    column_float: f64,
    column_complex: Vec<Vec<Vec<[u8; 3]>>>,
}

/// Create a temporary directory.
#[fixture]
fn tmp_dir() -> TempDir {
    // Create a TempDir. This object will be automatically
    // deleted when it goes out of scope.
    tempdir().unwrap()
}
type Writer = SqliteDatasetWriter<Complex>;

/// Create a SqliteDatasetWriter with a temporary directory.
/// Make sure to return the temporary directory so that it is not deleted.
#[fixture]
fn writer_fixture(tmp_dir: TempDir) -> (Writer, TempDir) {
    let temp_dir_str = tmp_dir.path();
    let storage = SqliteDatasetStorage::from_name("preprocessed").with_base_dir(temp_dir_str);
    let overwrite = true;
    let result = storage.writer::<Complex>(overwrite);
    assert!(result.is_ok());
    let writer = result.unwrap();
    (writer, tmp_dir)
}

#[test]
fn test_new() {
    // Test that the constructor works with overwrite = true
    let test_path = NamedTempFile::new().unwrap();
    let _writer = SqliteDatasetWriter::<Complex>::new(&test_path, true).unwrap();
    assert!(!test_path.path().exists());

    // Test that the constructor works with overwrite = false
    let test_path = NamedTempFile::new().unwrap();
    let result = SqliteDatasetWriter::<Complex>::new(&test_path, false);
    assert!(result.is_err());

    // Test that the constructor works with no existing file
    let temp = NamedTempFile::new().unwrap();
    let test_path = temp.path().to_path_buf();
    assert!(temp.close().is_ok());
    assert!(!test_path.exists());
    let _writer = SqliteDatasetWriter::<Complex>::new(&test_path, true).unwrap();
    assert!(!test_path.exists());
}

#[rstest]
pub fn sqlite_writer_write(writer_fixture: (Writer, TempDir)) {
    // Get the dataset_saver from the fixture and tmp_dir (will be deleted after scope)
    let (writer, _tmp_dir) = writer_fixture;

    assert!(writer.overwrite);
    assert!(!writer.db_file.exists());

    let new_item = Complex {
        column_str: "HI1".to_string(),
        column_bytes: vec![1_u8, 2, 3],
        column_int: 0,
        column_bool: true,
        column_float: 1.0,
        column_complex: vec![vec![vec![[1, 23_u8, 3]]]],
    };

    let index = writer.write("train", &new_item).unwrap();
    assert_eq!(index, 0);

    let mut writer = writer;

    writer.set_completed().expect("Failed to set completed");

    assert!(writer.db_file.exists());
    assert!(writer.db_file_tmp.is_none());

    let result = writer.write("train", &new_item);

    // Should fail because the writer is completed
    assert!(result.is_err());

    let dataset = SqliteDataset::<Complex>::from_db_file(writer.db_file, "train").unwrap();

    let fetched_item = dataset.get(0).unwrap();
    assert_eq!(fetched_item, new_item);
    assert_eq!(dataset.len(), 1);
}

#[rstest]
pub fn sqlite_writer_write_multi_thread(writer_fixture: (Writer, TempDir)) {
    // Get the dataset_saver from the fixture and tmp_dir (will be deleted after scope)
    let (writer, _tmp_dir) = writer_fixture;

    let writer = Arc::new(writer);
    let record_count = 20;

    let splits = ["train", "test"];

    (0..record_count).into_par_iter().for_each(|index: i64| {
        let thread_id: std::thread::ThreadId = std::thread::current().id();
        let sample = Complex {
            column_str: format!("test_{thread_id:?}_{index}"),
            column_bytes: vec![index as u8, 2, 3],
            column_int: index,
            column_bool: true,
            column_float: 1.0,
            column_complex: vec![vec![vec![[1, index as u8, 3]]]],
        };

        // half for train and half for test
        let split = splits[index as usize % 2];

        let _index = writer.write(split, &sample).unwrap();
    });

    let mut writer = Arc::try_unwrap(writer).unwrap();

    writer
        .set_completed()
        .expect("Should set completed successfully");

    let train = SqliteDataset::<Complex>::from_db_file(writer.db_file.clone(), "train").unwrap();
    let test = SqliteDataset::<Complex>::from_db_file(writer.db_file, "test").unwrap();

    assert_eq!(train.len(), record_count as usize / 2);
    assert_eq!(test.len(), record_count as usize / 2);
}
