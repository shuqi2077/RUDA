use std::path::Path;

use r2d2::Pool;
use r2d2_sqlite::{SqliteConnectionManager, rusqlite::OpenFlags};

use super::{Result, SqliteDatasetError};

/// Helper function to create a connection pool
pub(super) fn create_conn_pool<P: AsRef<Path>>(
    db_file: P,
    write: bool,
) -> Result<Pool<SqliteConnectionManager>> {
    let sqlite_flags = if write {
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE
    } else {
        OpenFlags::SQLITE_OPEN_READ_ONLY
    };

    let manager = SqliteConnectionManager::file(db_file).with_flags(sqlite_flags);
    Pool::new(manager).map_err(SqliteDatasetError::ConnectionPool)
}
