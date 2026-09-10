use std::path::{Path, PathBuf};

use sanitize_filename::sanitize;
use serde::{Serialize, de::DeserializeOwned};

use super::{Result, SqliteDataset, SqliteDatasetStorage, SqliteDatasetWriter};

impl SqliteDatasetStorage {
    /// Creates a new instance of `SqliteDatasetStorage` using a dataset name.
    ///
    /// # Arguments
    ///
    /// * `name` - A string slice that holds the name of the dataset.
    pub fn from_name(name: &str) -> Self {
        SqliteDatasetStorage {
            name: Some(name.to_string()),
            db_file: None,
            base_dir: None,
        }
    }

    /// Creates a new instance of `SqliteDatasetStorage` using a database file path.
    ///
    /// # Arguments
    ///
    /// * `db_file` - A reference to the Path that represents the database file path.
    pub fn from_file<P: AsRef<Path>>(db_file: P) -> Self {
        SqliteDatasetStorage {
            name: None,
            db_file: Some(db_file.as_ref().to_path_buf()),
            base_dir: None,
        }
    }

    /// Sets the base directory for storing the dataset.
    ///
    /// # Arguments
    ///
    /// * `base_dir` - A string slice that represents the base directory.
    pub fn with_base_dir<P: AsRef<Path>>(mut self, base_dir: P) -> Self {
        self.base_dir = Some(base_dir.as_ref().to_path_buf());
        self
    }

    /// Checks if the database file exists in the given path.
    ///
    /// # Returns
    ///
    /// * A boolean value indicating whether the file exists or not.
    pub fn exists(&self) -> bool {
        self.db_file().exists()
    }

    /// Fetches the database file path.
    ///
    /// # Returns
    ///
    /// * A `PathBuf` instance representing the file path.
    pub fn db_file(&self) -> PathBuf {
        match &self.db_file {
            Some(db_file) => db_file.clone(),
            None => {
                let name = sanitize(self.name.as_ref().expect("Name is not set"));
                Self::base_dir(self.base_dir.to_owned()).join(format!("{name}.db"))
            }
        }
    }

    /// Determines the base directory for storing the dataset.
    ///
    /// # Arguments
    ///
    /// * `base_dir` - An `Option` that may contain a `PathBuf` instance representing the base directory.
    ///
    /// # Returns
    ///
    /// * A `PathBuf` instance representing the base directory.
    pub fn base_dir(base_dir: Option<PathBuf>) -> PathBuf {
        match base_dir {
            Some(base_dir) => base_dir,
            None => dirs::cache_dir()
                .expect("Could not get cache directory")
                .join("ruda-dataset"),
        }
    }

    /// Provides a writer instance for the SQLite dataset.
    ///
    /// # Arguments
    ///
    /// * `overwrite` - A boolean indicating if the existing database file should be overwritten.
    ///
    /// # Returns
    ///
    /// * A `Result` which is `Ok` if the writer could be created, `Err` otherwise.
    pub fn writer<I>(&self, overwrite: bool) -> Result<SqliteDatasetWriter<I>>
    where
        I: Clone + Send + Sync + Serialize + DeserializeOwned,
    {
        SqliteDatasetWriter::new(self.db_file(), overwrite)
    }

    /// Provides a reader instance for the SQLite dataset.
    ///
    /// # Arguments
    ///
    /// * `split` - A string slice that defines the data split for reading (e.g., "train", "test").
    ///
    /// # Returns
    ///
    /// * A `Result` which is `Ok` if the reader could be created, `Err` otherwise.
    pub fn reader<I>(&self, split: &str) -> Result<SqliteDataset<I>>
    where
        I: Clone + Send + Sync + Serialize + DeserializeOwned,
    {
        if !self.exists() {
            panic!("The database file does not exist");
        }

        SqliteDataset::from_db_file(self.db_file(), split)
    }
}
