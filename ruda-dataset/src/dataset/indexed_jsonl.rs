//! Random-access JSONL without retaining deserialized examples in memory.
use super::Dataset;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    marker::PhantomData,
    path::Path,
    sync::Arc,
    time::UNIX_EPOCH,
};
#[cfg(not(unix))]
use std::sync::Mutex;

struct JsonlSource {
    file: File,
    #[cfg(not(unix))]
    cursor: Mutex<()>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Caller-selected handling of physically empty or JSON-whitespace-only rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JsonlBlankLines {
    /// Return an error without assigning that row a dataset index.
    Reject,
    /// Exclude blank rows, preserving the order of all other rows.
    Skip,
}

/// Immutable-source identity and explicit JSONL indexing policies.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonlIndexOptions {
    /// Caller-supplied identity of the immutable source/version, not its path.
    pub source_id: String,
    /// Whether physically blank rows belong to an invalid source or are skipped.
    pub blank_lines: JsonlBlankLines,
    /// Optional maximum raw row length, excluding LF but including CR.
    /// None imposes no row-size limit.
    pub max_record_bytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct FileIdentity {
    length: u64,
    modified: Option<(u64, u32)>,
}

impl FileIdentity {
    fn read(file: &File) -> io::Result<Self> {
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(invalid("indexed JSONL requires a regular immutable file"));
        }
        let modified = metadata.modified().ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| (duration.as_secs(), duration.subsec_nanos()));
        Ok(Self { length: metadata.len(), modified })
    }

    fn check(&self, file: &File) -> io::Result<()> {
        if self != &Self::read(file)? {
            return Err(invalid("JSONL source length or modification time changed"));
        }
        Ok(())
    }
}

/// Actual byte range of one retained physical row, without its LF delimiter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonlRecordLocation {
    /// Absolute byte position in the original source.
    pub offset: u64,
    /// Raw row length; CR in a CRLF row is retained as JSON whitespace.
    pub length: u64,
    /// One-based physical row number, including any skipped blank rows.
    pub line_number: u64,
}

/// Serializable indexing checkpoint; no source data or filesystem sidecar.
///
/// Byte ranges retain physical source order. The caller owns the source's
/// immutability and its version identity: length/mtime are change detection,
/// not a content hash. The index records rows, not their deserialization result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JsonlIndexState {
    version: u32,
    options: JsonlIndexOptions,
    identity: FileIdentity,
    processed_bytes: u64,
    pending_start: u64,
    pending_nonblank: bool,
    next_line: u64,
    skipped_lines: u64,
    records: Vec<JsonlRecordLocation>,
    finished: bool,
}

/// Work actually completed by a bounded indexing call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonlIndexProgress {
    /// Bytes whose indexing state has been committed, not buffered read-ahead.
    pub processed_bytes: u64,
    /// Immutable file length, including delimiters and skipped rows.
    pub total_bytes: u64,
    /// Completed retained records; a partial final record is not counted.
    pub records: usize,
    /// Physically blank rows explicitly excluded by the selected policy.
    pub skipped_lines: u64,
    /// True only after the final physical row has been finalized.
    pub finished: bool,
}

impl JsonlIndexState {
    /// Source identity and row policies recorded by this checkpoint.
    pub fn options(&self) -> &JsonlIndexOptions { &self.options }

    /// The completed rows in their original source order.
    pub fn records(&self) -> &[JsonlRecordLocation] { &self.records }

    /// Actual bytes/records processed so far, including an unfinished row.
    pub fn progress(&self) -> JsonlIndexProgress {
        JsonlIndexProgress {
            processed_bytes: self.processed_bytes,
            total_bytes: self.identity.length,
            records: self.records.len(),
            skipped_lines: self.skipped_lines,
            finished: self.finished,
        }
    }

    /// Validate checkpoint accounting before restoring it against a source.
    pub fn validate(&self) -> io::Result<()> {
        if self.version != 1 || self.options.source_id.is_empty() || self.next_line == 0 {
            return Err(invalid("invalid JSONL checkpoint version or source identity"));
        }
        if self.pending_start > self.processed_bytes || self.processed_bytes > self.identity.length
            || (self.pending_start == self.processed_bytes && self.pending_nonblank)
            || (self.finished && (self.processed_bytes != self.identity.length
                || self.pending_start != self.processed_bytes)) {
            return Err(invalid("JSONL checkpoint byte accounting is inconsistent"));
        }
        let completed_lines = u64::try_from(self.records.len())
            .ok().and_then(|count| count.checked_add(self.skipped_lines));
        if completed_lines.and_then(|count| count.checked_add(1)) != Some(self.next_line) {
            return Err(invalid("JSONL checkpoint physical row accounting is inconsistent"));
        }
        let pending_length = self.processed_bytes - self.pending_start;
        if self.options.max_record_bytes.is_some_and(|limit| pending_length > limit) {
            return Err(invalid("JSONL checkpoint contains an oversized pending row"));
        }
        let mut previous_end = 0;
        let mut previous_line = 0;
        for row in &self.records {
            let end = row.offset.checked_add(row.length)
                .ok_or_else(|| invalid("JSONL row byte range overflow"))?;
            if row.length == 0 || row.offset < previous_end || end > self.pending_start
                || row.line_number <= previous_line || row.line_number >= self.next_line
                || self.options.max_record_bytes.is_some_and(|limit| row.length > limit) {
                return Err(invalid("JSONL checkpoint row locations are inconsistent"));
            }
            previous_end = end.checked_add(1).unwrap_or(end);
            previous_line = row.line_number;
        }
        Ok(())
    }
}

/// Incremental byte-range indexer with serializable, caller-persisted state.
///
/// `scan_step` reads at most its byte budget. Index memory is proportional to
/// completed rows; scan memory is bounded independently of row/file length.
/// Invalid rows are never silently truncated, replaced, or indexed as blanks.
pub struct JsonlIndexBuilder {
    file: File,
    state: JsonlIndexState,
}

impl JsonlIndexBuilder {
    /// Open a source using explicit blank-line and row-size policies.
    pub fn open(path: impl AsRef<Path>, options: JsonlIndexOptions) -> io::Result<Self> {
        if options.source_id.is_empty() {
            return Err(invalid("supply an immutable JSONL source identity"));
        }
        let file = File::open(path)?;
        let identity = FileIdentity::read(&file)?;
        let state = JsonlIndexState {
            version: 1, options, identity, processed_bytes: 0, pending_start: 0,
            pending_nonblank: false, next_line: 1, skipped_lines: 0,
            records: Vec::new(), finished: false,
        };
        Ok(Self { file, state })
    }

    /// Resume exact byte/row accounting, including a partially scanned row.
    /// A matching caller identity is required even if the path has changed.
    pub fn resume(path: impl AsRef<Path>, source_id: &str, state: JsonlIndexState) -> io::Result<Self> {
        state.validate()?;
        if source_id != state.options.source_id {
            return Err(invalid("JSONL continuation belongs to a different source version"));
        }
        let file = File::open(path)?;
        state.identity.check(&file)?;
        Ok(Self { file, state })
    }

    /// Current state, suitable for caller-controlled checkpoint serialization.
    pub fn state(&self) -> &JsonlIndexState { &self.state }

    /// Current actual byte/record work counters.
    pub fn progress(&self) -> JsonlIndexProgress { self.state.progress() }

    fn finish_line(&mut self) -> io::Result<()> {
        let length = self.state.processed_bytes - self.state.pending_start;
        let next_line = self.state.next_line.checked_add(1)
            .ok_or_else(|| invalid("JSONL physical line count overflow"))?;
        if !self.state.pending_nonblank {
            match self.state.options.blank_lines {
                JsonlBlankLines::Reject => return Err(invalid(format!(
                    "blank JSONL row at physical line {}", self.state.next_line))),
                JsonlBlankLines::Skip => self.state.skipped_lines += 1,
            }
        } else {
            self.state.records.push(JsonlRecordLocation {
                offset: self.state.pending_start, length, line_number: self.state.next_line,
            });
        }
        self.state.next_line = next_line;
        self.state.pending_nonblank = false;
        Ok(())
    }

    /// Scan at most `byte_budget` source bytes, retaining exact restart state.
    /// Zero returns current counters without performing work. EOF finalizes a
    /// nonempty unterminated final row without adding a phantom trailing row.
    pub fn scan_step(&mut self, byte_budget: u64) -> io::Result<JsonlIndexProgress> {
        self.state.identity.check(&self.file)?;
        if self.state.finished || byte_budget == 0 { return Ok(self.progress()); }
        self.file.seek(SeekFrom::Start(self.state.processed_bytes))?;
        let mut remaining = byte_budget.min(self.state.identity.length - self.state.processed_bytes);
        let mut buffer = [0u8; 16 * 1024];
        while remaining != 0 {
            let length = remaining.min(buffer.len() as u64) as usize;
            self.file.read_exact(&mut buffer[..length])?;
            for &byte in &buffer[..length] {
                if byte == b'\n' {
                    self.finish_line()?;
                    self.state.processed_bytes += 1;
                    self.state.pending_start = self.state.processed_bytes;
                } else {
                    let row_length = self.state.processed_bytes - self.state.pending_start;
                    if self.state.options.max_record_bytes.is_some_and(|limit| row_length >= limit) {
                        return Err(invalid(format!("JSONL row {} exceeds the selected byte limit", self.state.next_line)));
                    }
                    self.state.pending_nonblank |= !matches!(byte, b' ' | b'\t' | b'\r');
                    self.state.processed_bytes += 1;
                }
            }
            remaining -= length as u64;
        }
        if self.state.processed_bytes == self.state.identity.length {
            if self.state.pending_start != self.state.processed_bytes {
                self.finish_line()?;
                self.state.pending_start = self.state.processed_bytes;
            }
            self.state.finished = true;
        }
        self.state.identity.check(&self.file)?;
        Ok(self.progress())
    }

    /// Consume a completed index and retain the already-open source handle.
    pub fn into_dataset<I: DeserializeOwned>(self) -> io::Result<IndexedJsonlDataset<I>> {
        self.state.validate()?;
        if !self.state.finished { return Err(invalid("JSONL indexing has not reached EOF")); }
        self.state.identity.check(&self.file)?;
        Ok(IndexedJsonlDataset {
            file: Arc::new(JsonlSource {
                file: self.file,
                #[cfg(not(unix))]
                cursor: Mutex::new(()),
            }), state: Arc::new(self.state), item: PhantomData,
        })
    }
}

/// Immutable indexed JSONL, with deserialization performed only on requested rows.
///
/// Clones share the source/index, not materialized examples. Unix reads use
/// independent byte offsets; other targets serialize cursor-based reads. Storage remains
/// proportional to the offset index plus concurrently requested row buffers.
pub struct IndexedJsonlDataset<I> {
    file: Arc<JsonlSource>,
    state: Arc<JsonlIndexState>,
    item: PhantomData<fn() -> I>,
}

impl<I> Clone for IndexedJsonlDataset<I> {
    fn clone(&self) -> Self {
        Self { file: self.file.clone(), state: self.state.clone(), item: PhantomData }
    }
}

impl<I: DeserializeOwned> IndexedJsonlDataset<I> {
    /// Reopen a fully indexed source without rebuilding or copying its contents.
    pub fn from_index(path: impl AsRef<Path>, source_id: &str, state: JsonlIndexState) -> io::Result<Self> {
        JsonlIndexBuilder::resume(path, source_id, state)?.into_dataset()
    }

    /// Completed immutable byte-range index, suitable for caller persistence.
    pub fn index(&self) -> &JsonlIndexState { &self.state }

    /// Return the actual row bytes, retaining CR and excluding LF.
    /// None means only that the dataset index is out of bounds.
    pub fn get_raw(&self, index: usize) -> io::Result<Option<Vec<u8>>> {
        let Some(row) = self.state.records.get(index) else { return Ok(None); };
        #[cfg(not(unix))]
        let _cursor = self.file.cursor.lock().map_err(|_| io::Error::other("JSONL source lock poisoned"))?;
        let file = &self.file.file;
        self.state.identity.check(file)?;
        let bytes = Self::read_row(file, row)?;
        self.state.identity.check(file)?;
        Ok(Some(bytes))
    }

    fn read_row(file: &File, row: &JsonlRecordLocation) -> io::Result<Vec<u8>> {
        let length = usize::try_from(row.length)
            .map_err(|_| invalid("JSONL row is larger than the process address space"))?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(length).map_err(|error| io::Error::other(error.to_string()))?;
        bytes.resize(length, 0);
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            file.read_exact_at(&mut bytes, row.offset)?;
        }
        #[cfg(not(unix))]
        {
            let mut file = file;
            file.seek(SeekFrom::Start(row.offset))?;
            file.read_exact(&mut bytes)?;
        }
        Ok(bytes)
    }

    /// Deserialize one actual row, distinguishing missing index from parse/I/O failure.
    pub fn get_result(&self, index: usize) -> io::Result<Option<I>> {
        let Some(bytes) = self.get_raw(index)? else { return Ok(None); };
        serde_json::from_slice(&bytes).map(Some).map_err(|error| invalid(format!(
            "JSONL dataset index {index}, physical line {}: {error}",
            self.state.records[index].line_number)))
    }

    /// Fetch requested actual rows under one metadata check pair.
    /// Requested order and duplicate indices are preserved; any missing index
    /// returns None before reading. An empty request returns an empty collection.
    pub fn get_many_raw(&self, indices: &[usize]) -> io::Result<Option<Vec<Vec<u8>>>> {
        if indices.iter().any(|&index| index >= self.state.records.len()) { return Ok(None); }
        if indices.is_empty() { return Ok(Some(Vec::new())); }
        #[cfg(not(unix))]
        let _cursor = self.file.cursor.lock().map_err(|_| io::Error::other("JSONL source lock poisoned"))?;
        let file = &self.file.file;
        self.state.identity.check(file)?;
        let rows = indices.iter().map(|&index| Self::read_row(file, &self.state.records[index]))
            .collect::<io::Result<Vec<_>>>()?;
        self.state.identity.check(file)?;
        Ok(Some(rows))
    }

    /// Deserialize a requested batch after completing its source reads.
    /// Parse/I/O failure is distinct from an out-of-bounds request, and no
    /// partially loaded collection is returned as a successful batch.
    pub fn get_many_result(&self, indices: &[usize]) -> io::Result<Option<Vec<I>>> {
        let Some(rows) = self.get_many_raw(indices)? else { return Ok(None); };
        indices.iter().zip(rows).map(|(&index, bytes)| {
            serde_json::from_slice(&bytes).map_err(|error| invalid(format!(
                "JSONL dataset index {index}, physical line {}: {error}",
                self.state.records[index].line_number)))
        }).collect::<io::Result<Vec<_>>>().map(Some)
    }
}

impl<I: DeserializeOwned> Dataset<I> for IndexedJsonlDataset<I> {
    /// The collection interface cannot carry errors: malformed/I/O-failed rows
    /// panic rather than masquerading as absent data. Use get_result to handle them.
    fn get(&self, index: usize) -> Option<I> {
        self.get_result(index).unwrap_or_else(|error| panic!("indexed JSONL read failed: {error}"))
    }

    fn get_many(&self, indices: &[usize]) -> Option<Vec<I>> {
        self.get_many_result(indices).unwrap_or_else(|error| panic!("indexed JSONL batch read failed: {error}"))
    }

    fn len(&self) -> usize { self.state.records.len() }
}
