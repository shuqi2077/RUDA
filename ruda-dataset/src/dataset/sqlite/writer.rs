use std::{
    collections::HashSet,
    fs,
    marker::PhantomData,
    path::Path,
    sync::{Arc, RwLock},
};

use gix_tempfile::{AutoRemove, ContainingDirectory};
use r2d2::PooledConnection;
use r2d2_sqlite::SqliteConnectionManager;
use serde::{Serialize, de::DeserializeOwned};

use super::{Result, SqliteDatasetError, SqliteDatasetWriter, connection::create_conn_pool};

impl<I> SqliteDatasetWriter<I>
where
    I: Clone + Send + Sync + Serialize + DeserializeOwned,
{
    /// Creates a new instance of `SqliteDatasetWriter`.
    ///
    /// # Arguments
    ///
    /// * `db_file` - A reference to the Path that represents the database file path.
    /// * `overwrite` - A boolean indicating if the existing database file should be overwritten.
    ///
    /// # Returns
    ///
    /// * A `Result` which is `Ok` if the writer could be created, `Err` otherwise.
    pub fn new<P: AsRef<Path>>(db_file: P, overwrite: bool) -> Result<Self> {
        let writer = Self {
            db_file: db_file.as_ref().to_path_buf(),
            db_file_tmp: None,
            splits: Arc::new(RwLock::new(HashSet::new())),
            overwrite,
            conn_pool: None,
            is_completed: Arc::new(RwLock::new(false)),
            phantom: PhantomData,
        };

        writer.init()
    }

    /// Initializes the dataset writer by creating the database file, tables, and connection pool.
    ///
    /// # Returns
    ///
    /// * A `Result` which is `Ok` if the writer could be initialized, `Err` otherwise.
    fn init(mut self) -> Result<Self> {
        // Remove the db file if it already exists
        if self.db_file.exists() {
            if self.overwrite {
                fs::remove_file(&self.db_file)?;
            } else {
                return Err(SqliteDatasetError::FileExists(self.db_file));
            }
        }

        // Create the database file directory if it does not exist
        let db_file_dir = self
            .db_file
            .parent()
            .ok_or("Unable to get parent directory")?;

        if !db_file_dir.exists() {
            fs::create_dir_all(db_file_dir)?;
        }

        // Create a temp database file name as {base_dir}/{name}.db.tmp
        let mut db_file_tmp = self.db_file.clone();
        db_file_tmp.set_extension("db.tmp");
        if db_file_tmp.exists() {
            fs::remove_file(&db_file_tmp)?;
        }

        // Create the temp database file and wrap it with a gix_tempfile::Handle
        // This will ensure that the temp file is deleted when the writer is dropped
        // or when process exits with SIGINT or SIGTERM (tempfile crate does not do this)
        gix_tempfile::signal::setup(Default::default());
        self.db_file_tmp = Some(gix_tempfile::writable_at(
            &db_file_tmp,
            ContainingDirectory::Exists,
            AutoRemove::Tempfile,
        )?);

        let conn_pool = create_conn_pool(db_file_tmp, true)?;
        self.conn_pool = Some(conn_pool);

        Ok(self)
    }

    /// Serializes and writes an item to the database. The item is written to the table for the
    /// specified split. If the table does not exist, it is created. If the table exists, the item
    /// is appended to the table. The serialization is done using the [MessagePack](https://msgpack.org/)
    ///
    /// # Arguments
    ///
    /// * `split` - A string slice that defines the data split for writing (e.g., "train", "test").
    /// * `item` - A reference to the item to be written to the database.
    ///
    /// # Returns
    ///
    /// * A `Result` containing the index of the inserted row if successful, an error otherwise.
    pub fn write(&self, split: &str, item: &I) -> Result<usize> {
        // Acquire the read lock (wont't block other reads)
        let is_completed = self.is_completed.read().unwrap();

        // If the writer is completed, return an error
        if *is_completed {
            return Err(SqliteDatasetError::Other(
                "Cannot save to a completed dataset writer",
            ));
        }

        // create the table for the split if it does not exist
        if !self.splits.read().unwrap().contains(split) {
            self.create_table(split)?;
        }

        // Get a connection from the pool
        let conn_pool = self.conn_pool.as_ref().unwrap();
        let conn = conn_pool.get()?;

        // Serialize the item using MessagePack
        let serialized_item = rmp_serde::to_vec(item)?;

        // Turn off the synchronous and journal mode for speed up
        // We are sacrificing durability for speed but it's okay because
        // we always recreate the dataset if it is not completed.
        pragma_update_with_error_handling(&conn, "synchronous", "OFF")?;
        pragma_update_with_error_handling(&conn, "journal_mode", "OFF")?;

        // Insert the serialized item into the database
        let insert_statement = format!("insert into {split} (item) values (?)");
        conn.execute(insert_statement.as_str(), [serialized_item])?;

        // Get the primary key of the last inserted row and convert to index (row_id-1)
        let index = (conn.last_insert_rowid() - 1) as usize;

        Ok(index)
    }

    /// Marks the dataset as completed and persists the temporary database file.
    pub fn set_completed(&mut self) -> Result<()> {
        let mut is_completed = self.is_completed.write().unwrap();

        // Force close the connection pool
        // This is required on Windows platform where the connection pool prevents
        // from persisting the db by renaming the temp file.
        if let Some(pool) = self.conn_pool.take() {
            std::mem::drop(pool);
        }

        // Rename the database file from tmp to db
        let _file_result = self
            .db_file_tmp
            .take() // take ownership of the temporary file and set to None
            .unwrap() // unwrap the temporary file
            .persist(&self.db_file)?
            .ok_or("Unable to persist the database file")?;

        *is_completed = true;
        Ok(())
    }

    /// Creates table for the data split.
    ///
    /// Note: call is idempotent and thread-safe.
    ///
    /// # Arguments
    ///
    /// * `split` - A string slice that defines the data split for the table (e.g., "train", "test").
    ///
    /// # Returns
    ///
    /// * A `Result` which is `Ok` if the table could be created, `Err` otherwise.
    ///
    /// TODO (@antimora): add support creating a table with columns corresponding to the item fields
    fn create_table(&self, split: &str) -> Result<()> {
        // Check if the split already exists
        if self.splits.read().unwrap().contains(split) {
            return Ok(());
        }

        let conn_pool = self.conn_pool.as_ref().unwrap();
        let connection = conn_pool.get()?;
        let create_table_statement = format!(
            "create table if not exists  {split} (row_id integer primary key autoincrement not \
             null, item blob not null)"
        );

        connection.execute(create_table_statement.as_str(), [])?;

        // Add the split to the splits
        self.splits.write().unwrap().insert(split.to_string());

        Ok(())
    }
}

/// Runs a pragma update and ignores the `ExecuteReturnedResults` error.
///
/// Sometimes ExecuteReturnedResults is returned when running a pragma update. This is not an error
/// and can be ignored. This function runs the pragma update and ignores the error if it is
/// `ExecuteReturnedResults`.
fn pragma_update_with_error_handling(
    conn: &PooledConnection<SqliteConnectionManager>,
    setting: &str,
    value: &str,
) -> Result<()> {
    let result = conn.pragma_update(None, setting, value);
    if let Err(error) = result
        && error != rusqlite::Error::ExecuteReturnedResults
    {
        return Err(SqliteDatasetError::Sql(error));
    }

    Ok(())
}
