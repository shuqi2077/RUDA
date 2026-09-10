mod connection;
mod reader;
mod storage;
mod writer;

#[cfg(test)]
mod tests;

use std::{
    collections::HashSet,
    io,
    marker::PhantomData,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use gix_tempfile::{
    Handle,
    handle::{Writable, persist},
};
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;

/// Result type for the sqlite dataset.
pub type Result<T> = core::result::Result<T, SqliteDatasetError>;

/// Sqlite dataset error.
#[derive(thiserror::Error, Debug)]
pub enum SqliteDatasetError {
    /// IO related error.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// Sql related error.
    #[error("Sql error: {0}")]
    Sql(#[from] serde_rusqlite::rusqlite::Error),

    /// Serde related error.
    #[error("Serde error: {0}")]
    Serde(#[from] rmp_serde::encode::Error),

    /// The database file already exists error.
    #[error("Overwrite flag is set to false and the database file already exists: {0}")]
    FileExists(PathBuf),

    /// Error when creating the connection pool.
    #[error("Failed to create connection pool: {0}")]
    ConnectionPool(#[from] r2d2::Error),

    /// Error when persisting the temporary database file.
    #[error("Could not persist the temporary database file: {0}")]
    PersistDbFile(#[from] persist::Error<Writable>),

    /// Any other error.
    #[error("{0}")]
    Other(&'static str),
}

impl From<&'static str> for SqliteDatasetError {
    fn from(s: &'static str) -> Self {
        SqliteDatasetError::Other(s)
    }
}

/// This struct represents a dataset where all items are stored in an SQLite database.
/// Each instance of this struct corresponds to a specific table within the SQLite database,
/// and allows for interaction with the data stored in the table in a structured and typed manner.
///
/// The SQLite database must contain a table with the same name as the `split` field. This table should
/// have a primary key column named `row_id`, which is used to index the rows in the table. The `row_id`
/// should start at 1, while the corresponding dataset `index` should start at 0, i.e., `row_id` = `index` + 1.
///
/// Table columns can be represented in two ways:
///
/// 1. The table can have a column for each field in the `I` struct. In this case, the column names in the table
///    should match the field names of the `I` struct. The field names can be a subset of column names and
///    can be in any order.
///
/// For the supported field types, refer to:
/// - [Serialization field types](https://docs.rs/serde_rusqlite/latest/serde_rusqlite)
/// - [SQLite data types](https://www.sqlite.org/datatype3.html)
///
/// 2. The fields in the `I` struct can be serialized into a single column `item` in the table. In this case, the table
///    should have a single column named `item` of type `BLOB`. This is useful when the `I` struct contains complex fields
///    that cannot be mapped to a SQLite type, such as nested structs, vectors, etc. The serialization is done using
///    [MessagePack](https://msgpack.org/).
///
/// Note: The code automatically figures out which of the above two cases is applicable, and uses the appropriate
/// method to read the data from the table.
#[derive(Debug)]
pub struct SqliteDataset<I> {
    db_file: PathBuf,
    split: String,
    conn_pool: Pool<SqliteConnectionManager>,
    columns: Vec<String>,
    len: usize,
    select_statement: String,
    row_serialized: bool,
    phantom: PhantomData<I>,
}

/// The `SqliteDatasetStorage` struct represents a SQLite database for storing datasets.
/// It consists of an optional name, a database file path, and a base directory for storage.
#[derive(Clone, Debug)]
pub struct SqliteDatasetStorage {
    name: Option<String>,
    db_file: Option<PathBuf>,
    base_dir: Option<PathBuf>,
}

/// This `SqliteDatasetWriter` struct is a SQLite database writer dedicated to storing datasets.
/// It retains the current writer's state and its database connection.
///
/// Being thread-safe, this writer can be concurrently used across multiple threads.
///
/// Typical applications include:
///
/// - Generation of a new dataset
/// - Storage of preprocessed data or metadata
/// - Enlargement of a dataset's item count post preprocessing
#[derive(Debug)]
pub struct SqliteDatasetWriter<I> {
    db_file: PathBuf,
    db_file_tmp: Option<Handle<Writable>>,
    splits: Arc<RwLock<HashSet<String>>>,
    overwrite: bool,
    conn_pool: Option<Pool<SqliteConnectionManager>>,
    is_completed: Arc<RwLock<bool>>,
    phantom: PhantomData<I>,
}
