use std::{
    fmt::Write,
    marker::PhantomData,
    path::{Path, PathBuf},
};

use r2d2::Pool;
use r2d2_sqlite::{
    SqliteConnectionManager,
    rusqlite::{OptionalExtension, Row, Statement, limits::Limit, params_from_iter},
};
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

    fn read_item(&self, statement: &mut Statement<'_>, index: usize) -> Option<I>
    where
        I: DeserializeOwned,
    {
        // Row ids start with 1 (one) and index starts with 0 (zero)
        let row_id = index + 1;

        statement
            .query_row([row_id], |row| Ok(self.item_from_row(row)))
            .optional()
            .unwrap()
    }

    fn item_from_row(&self, row: &Row<'_>) -> I
    where
        I: DeserializeOwned,
    {
        if self.row_serialized {
            // Fetch with a single column `item` and deserialize it with MessagePack
            rmp_serde::from_slice::<I>(row.get_ref(0).unwrap().as_blob().unwrap()).unwrap()
        } else {
            // Fetch a row with multiple columns and deserialize it serde_rusqlite
            from_row_with_columns::<I>(row, &self.columns).unwrap()
        }
    }
}

impl<I> Dataset<I> for SqliteDataset<I>
where
    I: Clone + Send + Sync + DeserializeOwned,
{
    /// Get an item from the dataset.
    fn get(&self, index: usize) -> Option<I> {
        let connection = self.conn_pool.get().unwrap();
        let mut statement = connection
            .prepare_cached(self.select_statement.as_str())
            .unwrap();
        self.read_item(&mut statement, index)
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        if indices.is_empty() {
            return Some(Vec::new());
        }
        let connection = self.conn_pool.get().unwrap();
        let variable_limit = connection.limit(Limit::SQLITE_LIMIT_VARIABLE_NUMBER).unwrap() as usize;
        let sql_limit = (connection.limit(Limit::SQLITE_LIMIT_SQL_LENGTH).unwrap() as usize)
            .min(i32::MAX as usize - 1);
        let prefix = "WITH req(row_id, ord) AS (VALUES ";
        let selection = if self.row_serialized { "t.item" } else { "t.*" };
        let suffix = format!(
            ") SELECT {selection}, t.row_id FROM req LEFT JOIN {} t ON t.row_id = req.row_id ORDER BY req.ord",
            self.split,
        );
        if variable_limit == 0 || prefix.len() + suffix.len() + 5 > sql_limit {
            let mut statement = connection
                .prepare_cached(self.select_statement.as_str())
                .unwrap();
            return indices
                .iter()
                .map(|&index| self.read_item(&mut statement, index))
                .collect();
        }

        let mut items = Vec::with_capacity(indices.len());
        let mut query = String::new();
        let mut start = 0;
        while start < indices.len() {
            query.clear();
            query.push_str(prefix);
            let mut end = start;
            while end < indices.len() && end - start < variable_limit {
                let ordinal = end - start;
                let digits = if ordinal == 0 {
                    1
                } else {
                    ordinal.ilog10() as usize + 1
                };
                let comma = usize::from(ordinal != 0);
                if query.len() + suffix.len() + 4 + digits + comma > sql_limit {
                    break;
                }
                if ordinal != 0 {
                    query.push(',');
                }
                let _ = write!(query, "(?,{ordinal})");
                end += 1;
            }
            query.push_str(&suffix);
            let mut statement = connection.prepare(&query).unwrap();
            let present_column = statement.column_count() - 1;
            let rows = statement
                .query_map(
                    params_from_iter(indices[start..end].iter().map(|&index| index + 1)),
                    |row| {
                        if row.get::<_, Option<i64>>(present_column)?.is_none() {
                            Ok(None)
                        } else {
                            Ok(Some(self.item_from_row(row)))
                        }
                    },
                )
                .unwrap();
            let before = items.len();
            for row in rows {
                items.push(row.unwrap()?);
            }
            if items.len() - before != end - start {
                return None;
            }
            start = end;
        }
        Some(items)
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
