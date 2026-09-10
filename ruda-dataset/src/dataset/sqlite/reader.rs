use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
};

use r2d2::Pool;
use r2d2_sqlite::{SqliteConnectionManager, rusqlite::OptionalExtension};
use serde::de::DeserializeOwned;
use serde_rusqlite::{columns_from_statement, from_row_with_columns};

use super::{Result, SqliteDataset, connection::create_conn_pool};
use crate::Dataset;

impl<I> SqliteDataset<I> {
    /// Initializes a `SqliteDataset` from a SQLite database file and a split name.
    pub fn from_db_file<P: AsRef<Path>>(db_file: P, split: &str) -> Result<Self> {
        // Create a connection pool
        let conn_pool = create_conn_pool(&db_file, false)?;

        // Determine how the table is stored
        let row_serialized = Self::check_if_row_serialized(&conn_pool, split)?;

        // Create a select statement and save it
        let select_statement = if row_serialized {
            format!("select item from {split} where row_id = ?")
        } else {
            format!("select * from {split} where row_id = ?")
        };

        // Save the column names and the number of rows
        let (columns, len) = fetch_columns_and_len(&conn_pool, &select_statement, split)?;

        Ok(SqliteDataset {
            db_file: db_file.as_ref().to_path_buf(),
            split: split.to_string(),
            conn_pool,
            columns,
            len,
            select_statement,
            row_serialized,
            phantom: PhantomData,
        })
    }

    /// Returns true if table has two columns: row_id (integer) and item (blob).
    ///
    /// This is used to determine if the table is row serialized or not.
    fn check_if_row_serialized(
        conn_pool: &Pool<SqliteConnectionManager>,
        split: &str,
    ) -> Result<bool> {
        // This struct is used to store the column name and type
        struct Column {
            name: String,
            ty: String,
        }

        const COLUMN_NAME: usize = 1;
        const COLUMN_TYPE: usize = 2;

        let sql_statement = format!("PRAGMA table_info({split})");

        let conn = conn_pool.get()?;

        let mut stmt = conn.prepare(sql_statement.as_str())?;
        let column_iter = stmt.query_map([], |row| {
            Ok(Column {
                name: row
                    .get::<usize, String>(COLUMN_NAME)
                    .unwrap()
                    .to_lowercase(),
                ty: row
                    .get::<usize, String>(COLUMN_TYPE)
                    .unwrap()
                    .to_lowercase(),
            })
        })?;

        let mut columns: Vec<Column> = vec![];

        for column in column_iter {
            columns.push(column?);
        }

        if columns.len() != 2 {
            Ok(false)
        } else {
            // Check if the column names and types match the expected values
            Ok(columns[0].name == "row_id"
                && columns[0].ty == "integer"
                && columns[1].name == "item"
                && columns[1].ty == "blob")
        }
    }

    /// Get the database file name.
    pub fn db_file(&self) -> PathBuf {
        self.db_file.clone()
    }

    /// Get the split name.
    pub fn split(&self) -> &str {
        self.split.as_str()
    }
}

impl<I> Dataset<I> for SqliteDataset<I>
where
    I: Clone + Send + Sync + DeserializeOwned,
{
    /// Get an item from the dataset.
    fn get(&self, index: usize) -> Option<I> {
        // Row ids start with 1 (one) and index starts with 0 (zero)
        let row_id = index + 1;

        // Get a connection from the pool
        let connection = self.conn_pool.get().unwrap();
        let mut statement = connection.prepare(self.select_statement.as_str()).unwrap();

        if self.row_serialized {
            // Fetch with a single column `item` and deserialize it with MessagePack
            statement
                .query_row([row_id], |row| {
                    // Deserialize item (blob) with MessagePack (rmp-serde)
                    Ok(
                        rmp_serde::from_slice::<I>(row.get_ref(0).unwrap().as_blob().unwrap())
                            .unwrap(),
                    )
                })
                .optional() //Converts Error (not found) to None
                .unwrap()
        } else {
            // Fetch a row with multiple columns and deserialize it serde_rusqlite
            statement
                .query_row([row_id], |row| {
                    // Deserialize the row with serde_rusqlite
                    Ok(from_row_with_columns::<I>(row, &self.columns).unwrap())
                })
                .optional() //Converts Error (not found) to None
                .unwrap()
        }
    }

    /// Return the number of rows in the dataset.
    fn len(&self) -> usize {
        self.len
    }
}

/// Fetch the column names and the number of rows from the database.
fn fetch_columns_and_len(
    conn_pool: &Pool<SqliteConnectionManager>,
    select_statement: &str,
    split: &str,
) -> Result<(Vec<String>, usize)> {
    // Save the column names
    let connection = conn_pool.get()?;
    let statement = connection.prepare(select_statement)?;
    let columns = columns_from_statement(&statement);

    // Count the number of rows and save it as len
    //
    // NOTE: Using coalesce(max(row_id), 0) instead of count(*) because count(*) is super slow for large tables.
    // The coalesce(max(row_id), 0) returns 0 if the table is empty, otherwise it returns the max row_id,
    // which corresponds to the number of rows in the table.
    // The main assumption, which always holds true, is that the row_id is always increasing and there are no gaps.
    // This is true for all the datasets that we are using, otherwise row_id will not correspond to the index.
    let mut statement =
        connection.prepare(format!("select coalesce(max(row_id), 0) from {split}").as_str())?;

    let len = statement.query_row([], |row| {
        let len: usize = row.get(0)?;
        Ok(len)
    })?;
    Ok((columns, len))
}
