use std::{
    cell::RefCell,
    fs::{self, File},
    io::{Cursor, Write},
    path::{Path, PathBuf},
    string::{String, ToString},
    sync::Arc,
    vec::Vec,
};

use ciborium::{de::from_reader, ser::into_writer};
use hashbrown::HashMap;
use serde::Serialize;

use crate::cache::{
    Cache, CacheError, CacheKey, CacheOption, CacheValue, Entry, sanitize_path_segment,
};

/// The in-memory cache used by the chunked kernel cache.
/// Callers own an `Arc`, so loading another chunk or dropping the cache cannot
/// invalidate a value that has already been returned.
type InMemoryCache<K, V> = RefCell<HashMap<K, Arc<V>>>;

/// A chunked cache for compilation artifacts. Uses a human readable table of contents, with binary
/// storage for the compiled kernel.
#[derive(Debug)]
pub struct CompilationCache<K: CacheKey, V: CacheValue> {
    toc: Cache<K, String>,
    in_memory_cache: InMemoryCache<K, V>,
    current_chunk: File,
    current_chunk_path_normalized: String,
    cache_root: PathBuf,
}

/// Error related to caching.
#[derive(Debug)]
pub enum CompilationCacheError<K: Serialize, V: Serialize> {
    /// Can't insert an entry with the same key, but different value.
    #[allow(missing_docs)]
    DuplicatedKey {
        key: K,
        value_previous: V,
        value_updated: V,
    },
    /// The table of contents cache had an error
    #[allow(missing_docs)]
    TocError(CacheError<K, String>),
}

impl<K: CacheKey, V: CacheValue> CompilationCache<K, V> {
    /// Create a new cache and load the data from the provided path if it exists.
    #[cfg_attr(feature="tracing", tracing::instrument(
        level = "trace",
        skip(path),
        fields(path = ?path.as_ref())))]
    pub fn new<P: AsRef<Path>>(path: P, option: CacheOption) -> Self {
        let (_, name, version, root, _) = option.clone().resolve();
        let path = path.as_ref();
        let toc_path = path.join("toc");
        // `.cbor` suffix (was `.bin` with bincode) invalidates old on-disk chunks.
        let chunk_path = Path::new("chunk0.cbor"); // Split later

        let cache_root = get_persistent_cache_root(path, root, name, version);
        let chunk_path = get_persistent_chunk_file_path(chunk_path, &cache_root);

        let in_memory_cache = InMemoryCache::default();
        let toc = Cache::new(toc_path, option);

        if fs::exists(&chunk_path).unwrap_or(false) {
            Self::read_chunk(&chunk_path, &in_memory_cache);
        }

        let current_chunk = open_chunk_writable(&chunk_path);

        Self {
            toc,
            in_memory_cache,
            current_chunk,
            current_chunk_path_normalized: normalized_path(
                chunk_path
                    .strip_prefix(&cache_root)
                    .expect("Should contain root"),
            ),
            cache_root,
        }
    }

    /// Fetch an independently owned item from the cache.
    ///
    /// The returned value remains valid across subsequent cache loads, inserts,
    /// and destruction of the cache. Borrow through `Arc::as_ref()` when needed.
    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        if let Some(value) = self.get_cached(key) {
            return Some(value);
        }
        let chunk = self.toc.get(key)?;
        Self::read_chunk(&self.cache_root.join(chunk), &self.in_memory_cache);
        self.get_cached(key)
    }

    fn get_cached(&self, key: &K) -> Option<Arc<V>> {
        self.in_memory_cache.borrow().get(key).cloned()
    }

    fn read_chunk(chunk: &PathBuf, cache: &InMemoryCache<K, V>) {
        let data = match fs::read(chunk) {
            Ok(data) => data,
            Err(err) => {
                // A compilation cache is disposable. A stale table of contents
                // must produce a cache miss, not terminate a training process.
                log::warn!("Unable to read compilation cache chunk {chunk:?}: {err}");
                return;
            }
        };
        let mut cursor = Cursor::new(data);
        // Collect new entries first so we only need to lock once everything is loaded
        let mut new_entries = Vec::new();
        let mut idx = 0;
        loop {
            let pos = cursor.position() as usize;
            let total_len = cursor.get_ref().len();
            if pos >= total_len {
                break;
            }
            match from_reader::<Entry<K, V>, _>(&mut cursor) {
                Ok(entry) => {
                    new_entries.push((entry.key, Arc::new(entry.value)));
                }
                Err(err) => {
                    // Entries have no independent framing. After a decode error
                    // we cannot safely locate another entry; keep only the valid
                    // prefix rather than interpreting arbitrary trailing bytes.
                    log::warn!(
                        "Corrupted cache file {chunk:?}, stopping at entry {idx}: {err}",
                    );
                    break;
                }
            }
            idx += 1;
        }

        let mut cache = cache.borrow_mut();
        for (key, value) in new_entries {
            match cache.entry(key) {
                hashbrown::hash_map::Entry::Vacant(entry) => {
                    entry.insert(value);
                }
                hashbrown::hash_map::Entry::Occupied(entry) => {
                    // Preserve the append-only contract within this cache
                    // instance even if another writer produced a duplicate.
                    if entry.get().as_ref() != value.as_ref() {
                        log::warn!("Conflicting duplicate in compilation cache chunk {chunk:?}");
                    }
                }
            }
        }
    }

    /// Insert a new item to the cache.
    ///
    /// Returns an error if a different value already exists for the key.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), CompilationCacheError<K, V>> {
        if let Some(existing) = self.get(&key) {
            if existing.as_ref() != &value {
                return Err(CompilationCacheError::DuplicatedKey {
                    key,
                    value_previous: existing.as_ref().clone(),
                    value_updated: value,
                });
            } else {
                return Ok(());
            }
        }

        // Insert
        {
            let entry = Entry {
                key: key.clone(),
                value,
            };
            let mut bytes = Vec::new();
            into_writer(&entry, &mut bytes).expect("Can serialize data");
            self.current_chunk
                .write_all(&bytes)
                .expect("Failed to write to chunk");

            let mut cache = self.in_memory_cache.borrow_mut();
            cache.insert(entry.key, Arc::new(entry.value));
        }
        self.toc
            .insert(key, self.current_chunk_path_normalized.clone())
            .map_err(CompilationCacheError::TocError)?;

        Ok(())
    }
}

fn get_persistent_cache_root(
    path_partial: impl AsRef<Path>,
    root: PathBuf,
    name: String,
    version: String,
) -> PathBuf {
    let path_partial = path_partial.as_ref();
    let mut path = root
        .join(sanitize_path_segment(&name))
        .join(sanitize_path_segment(&version));

    for segment in path_partial.iter() {
        // Skip the name directory since it resets the previous path segments.
        if segment == "/" {
            continue;
        }
        path = path.join(sanitize_path_segment(segment.to_str().unwrap()));
    }

    std::path::absolute(path).expect("Not empty, so can't fail")
}

fn get_persistent_chunk_file_path<P: AsRef<Path>>(path_partial: P, chunks_root: &Path) -> PathBuf {
    let path_partial: &Path = path_partial.as_ref();

    let mut path = chunks_root.to_path_buf();

    for segment in path_partial.iter() {
        // Skip the name directory since it resets the previous path segments.
        if segment == "/" {
            continue;
        }
        path = path.join(sanitize_path_segment(segment.to_str().unwrap()));
    }

    std::path::absolute(path).expect("Not empty, so can't fail")
}

fn normalized_path(path: &Path) -> String {
    let path = path.to_string_lossy().to_string();
    path.replace("\\", "/")
}

fn open_chunk_writable(path: &Path) -> File {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("Failed to create parent");
    }
    let file = File::options().append(true).create(true).open(path);
    file.expect("Failed to open write chunk")
}


#[cfg(test)]
mod safety_tests {
    use super::*;

    fn options(root: &Path) -> CacheOption {
        CacheOption::default().root(root).name("ruda-cache-regression").version("1")
    }

    #[test]
    fn value_remains_alive_after_cache_drop() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = CompilationCache::<u64, String>::new("ptx", options(dir.path()));
        cache.insert(1, "kept".into()).unwrap();
        let held = cache.get(&1).unwrap();
        drop(cache);
        assert_eq!(held.as_str(), "kept");
    }

    #[test]
    fn two_instances_reloading_a_chunk_keep_existing_owners_valid() {
        let dir = tempfile::tempdir().unwrap();
        let mut a = CompilationCache::<u64, String>::new("ptx", options(dir.path()));
        let mut b = CompilationCache::<u64, String>::new("ptx", options(dir.path()));
        b.insert(1, "first".into()).unwrap();
        // Inserting synchronizes A's table of contents with B's entry, while
        // the value for key 1 is not yet in A's in-memory chunk cache.
        a.insert(2, "second".into()).unwrap();
        let held = a.get(&2).unwrap();
        assert_eq!(a.get(&1).unwrap().as_str(), "first");
        assert!(Arc::ptr_eq(&held, &a.get(&2).unwrap()));
        drop(a);
        drop(b);
        assert_eq!(held.as_str(), "second");
    }

    #[test]
    fn duplicate_insert_cannot_replace_a_live_value() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = CompilationCache::<u64, String>::new("ptx", options(dir.path()));
        cache.insert(1, "original".into()).unwrap();
        let held = cache.get(&1).unwrap();
        assert!(matches!(cache.insert(1, "changed".into()),
            Err(CompilationCacheError::DuplicatedKey { .. })));
        assert_eq!(held.as_str(), "original");
        cache.insert(1, "original".into()).unwrap();
        assert!(Arc::ptr_eq(&held, &cache.get(&1).unwrap()));
    }

    #[test]
    fn missing_chunk_is_a_cache_miss() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = CompilationCache::<u64, String>::new("ptx", options(dir.path()));
        cache.toc.insert(7, "missing.cbor".into()).unwrap();
        assert!(cache.get(&7).is_none());
    }

    #[test]
    fn truncated_chunk_retains_only_the_valid_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("truncated.cbor");
        let mut bytes = Vec::new();
        into_writer(&Entry { key: 1_u64, value: String::from("valid") }, &mut bytes).unwrap();
        bytes.push(0x9f); // Unterminated CBOR indefinite-length array.
        fs::write(&file, bytes).unwrap();
        let cache = InMemoryCache::<u64, String>::default();
        CompilationCache::<u64, String>::read_chunk(&file, &cache);
        let map = cache.borrow();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1).unwrap().as_str(), "valid");
    }

    #[test]
    fn conflicting_disk_duplicates_preserve_the_first_value() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("duplicates.cbor");
        let mut bytes = Vec::new();
        for value in ["first", "conflict"] {
            into_writer(&Entry { key: 1_u64, value: value.to_string() }, &mut bytes).unwrap();
        }
        fs::write(&file, bytes).unwrap();
        let cache = InMemoryCache::<u64, String>::default();
        CompilationCache::<u64, String>::read_chunk(&file, &cache);
        assert_eq!(cache.borrow().get(&1).unwrap().as_str(), "first");
    }
}
