use super::RecorderError;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

pub(super) struct RecordWriter {
    writer: Option<BufWriter<File>>,
    temporary: Option<PathBuf>,
    destination: PathBuf,
}

fn io_error(error: io::Error) -> RecorderError {
    match error.kind() {
        io::ErrorKind::NotFound => RecorderError::FileNotFound(error.to_string()),
        _ => RecorderError::Unknown(error.to_string()),
    }
}

impl RecordWriter {
    pub(super) fn new(destination: &Path) -> Result<Self, RecorderError> {
        let destination = std::path::absolute(destination).map_err(io_error)?;
        let parent = destination.parent().ok_or_else(|| {
            RecorderError::Unknown("record destination has no parent directory".into())
        })?;
        let name = destination.file_name().ok_or_else(|| {
            RecorderError::Unknown("record destination has no file name".into())
        })?;
        fs::create_dir_all(parent).map_err(io_error)?;

        loop {
            let mut temporary_name = std::ffi::OsString::from(".");
            temporary_name.push(name);
            temporary_name.push(format!(
                ".ruda-{}-{}.tmp",
                std::process::id(), NEXT_FILE.fetch_add(1, Ordering::Relaxed),
            ));
            let temporary = parent.join(temporary_name);
            match OpenOptions::new().write(true).create_new(true).open(&temporary) {
                Ok(file) => return Ok(Self {
                    writer: Some(BufWriter::new(file)),
                    temporary: Some(temporary),
                    destination,
                }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(io_error(error)),
            }
        }
    }

    pub(super) fn commit(mut self) -> Result<(), RecorderError> {
        let writer = self.writer.take().expect("record writer is open");
        let file = writer.into_inner().map_err(|error| {
            RecorderError::Unknown(error.to_string())
        })?;
        file.sync_all().map_err(io_error)?;
        drop(file);
        fs::rename(
            self.temporary.as_ref().expect("record temporary file exists"),
            &self.destination,
        ).map_err(io_error)?;
        self.temporary = None;
        Ok(())
    }
}

impl Write for RecordWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.writer.as_mut().expect("record writer is open").write(bytes)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.as_mut().expect("record writer is open").flush()
    }
}

impl Drop for RecordWriter {
    fn drop(&mut self) {
        drop(self.writer.take());
        if let Some(path) = self.temporary.take() {
            if let Err(error) = fs::remove_file(&path) {
                if error.kind() != io::ErrorKind::NotFound {
                    log::warn!("Cannot remove incomplete record {:?}: {}", path, error);
                }
            }
        }
    }
}
