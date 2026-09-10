//! Bounded persistent records. Files contain the FULL key; the filename hash is only an index.
use std::{fs::{self, OpenOptions}, io::{self, Read, Write}, path::{Path, PathBuf},
    string::{String, ToString}, sync::atomic::{AtomicU64, Ordering}, time::{SystemTime, UNIX_EPOCH}, vec::Vec};

const MAGIC: &[u8; 8] = b"RUDATN01";
const MAX_FILE: usize = 512 * 1024;
const MAX_KEY: usize = 384 * 1024;
const MAX_NAME: usize = 4096;
static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct Record {
    pub scope: u8,
    pub key: String,
    pub winner: String,
    pub created: u64,
    pub reference_ns: u64,
    pub winner_ns: u64,
    pub ratio: f64,
    pub verified: bool,
}
/// This is NOT a cryptographic authenticity check. Caches must be in a trusted user directory.
pub fn digest(bytes: &[u8]) -> String {
    let mut a = 0xcbf29ce484222325u64;
    let mut b = 0x84222325cbf29ce4u64;
    for &v in bytes {
        a = (a ^ v as u64).wrapping_mul(0x100000001b3);
        b = (b ^ (v as u64).wrapping_add(1)).wrapping_mul(0x100000001b3);
    }
    std::format!("{a:016x}{b:016x}")
}
pub fn now_seconds() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }
impl Record {
    pub fn is_fresh(&self, now: u64, ttl: u64) -> bool {
        self.created <= now && now - self.created < ttl
    }
    fn encode(&self) -> io::Result<Vec<u8>> {
        if self.scope > 2 || self.key.len() > MAX_KEY || self.winner.len() > MAX_NAME || self.winner.is_empty()
            || self.reference_ns == 0 || self.winner_ns == 0 || !self.ratio.is_finite() || self.ratio <= 0.0 {
            return Err(invalid("invalid tuning record"));
        }
        let mut bytes = MAGIC.to_vec();
        for text in [&self.key, &self.winner] {
            bytes.extend_from_slice(&(text.len() as u32).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        }
        for value in [self.created, self.reference_ns, self.winner_ns, self.ratio.to_bits()] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.push(u8::from(self.verified));
        bytes.push(self.scope);
        let checksum = digest(&bytes);
        bytes.extend_from_slice(checksum.as_bytes());
        Ok(bytes)
    }
    fn decode(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() > MAX_FILE || bytes.len() < 83 || !bytes.starts_with(MAGIC) { return Err(invalid("bad tuning cache header/length")); }
        let end = bytes.len() - 32;
        if digest(&bytes[..end]).as_bytes() != &bytes[end..] { return Err(invalid("tuning cache checksum mismatch")); }
        let mut p = 8;
        fn take<'a>(bytes: &'a [u8], p: &mut usize, count: usize) -> io::Result<&'a [u8]> {
            let stop = p.checked_add(count).ok_or_else(|| invalid("cache length overflow"))?;
            let value = bytes.get(*p..stop).ok_or_else(|| invalid("truncated tuning cache"))?;
            *p = stop; Ok(value)
        }
        let mut text = |limit: usize| -> io::Result<String> {
            let length = u32::from_le_bytes(take(&bytes[..end], &mut p, 4)?.try_into().map_err(|_| invalid("bad length"))?) as usize;
            if length > limit { return Err(invalid("cache field too large")); }
            String::from_utf8(take(&bytes[..end], &mut p, length)?.to_vec()).map_err(|_| invalid("invalid cache UTF-8"))
        };
        let key = text(MAX_KEY)?;
        let winner = text(MAX_NAME)?;
        let mut number = || -> io::Result<u64> {
            Ok(u64::from_le_bytes(take(&bytes[..end], &mut p, 8)?.try_into().map_err(|_| invalid("bad u64"))?))
        };
        let created = number()?;
        let reference_ns = number()?;
        let winner_ns = number()?;
        let ratio = f64::from_bits(number()?);
        let verified = match take(&bytes[..end], &mut p, 1)?[0] { 0 => false, 1 => true, _ => return Err(invalid("invalid validation flag")) };
        let scope = take(&bytes[..end], &mut p, 1)?[0];
        if scope > 2 || p != end || winner.is_empty() || reference_ns == 0 || winner_ns == 0 || !ratio.is_finite() || ratio <= 0.0 { return Err(invalid("invalid cache record")); }
        Ok(Self { scope, key, winner, created, reference_ns, winner_ns, ratio, verified })
    }
}
fn invalid(message: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, message.to_string()) }

#[derive(Debug)]
pub struct DiskCache { root: PathBuf, capacity: usize }
impl DiskCache {
    /// A dedicated directory is used, never the legacy autotune cache directory itself.
    pub fn new(root: PathBuf, capacity: usize) -> Self { Self { root: root.join("stack-autotune-v1"), capacity } }
    fn path(&self, key: &str) -> PathBuf { self.root.join(std::format!("stack-v1-{}.rtune", digest(key.as_bytes()))) }
    pub fn load(&self, key: &str) -> io::Result<Option<Record>> {
        let path = self.path(key);
        let mut file = match fs::File::open(path) { Ok(f) => f, Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None), Err(e) => return Err(e) };
        if file.metadata()?.len() > MAX_FILE as u64 { return Err(invalid("tuning cache file too large")); }
        let mut bytes = Vec::new();
        (&mut file).take((MAX_FILE + 1) as u64).read_to_end(&mut bytes)?;
        let record = Record::decode(&bytes)?;
        // Hash collision or copied cache files are misses, not alternate executable choices.
        if record.key == key { Ok(Some(record)) } else { Ok(None) }
    }
    pub fn remove(&self, key: &str) -> io::Result<()> {
        match fs::remove_file(self.path(key)) { Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()), r => r }
    }
    pub fn save(&self, record: &Record) -> io::Result<()> {
        let bytes = record.encode()?;
        fs::create_dir_all(&self.root)?;
        let temp = self.root.join(std::format!(".tmp-{}-{}", std::process::id(), NEXT_FILE.fetch_add(1, Ordering::Relaxed)));
        // create_new avoids silently following an existing temporary path or symlink.
        let result = (|| {
            let mut file = OpenOptions::new().write(true).create_new(true).open(&temp)?;
            file.write_all(&bytes)?; file.sync_all()?; drop(file);
            // Unix atomically replaces. Platforms rejecting replacement report a cache warning;
            // they never delete a known good record first to force a rename.
            fs::rename(&temp, self.path(&record.key))?;
            self.prune()
        })();
        if result.is_err() { let _ = fs::remove_file(&temp); }
        result
    }
    fn prune(&self) -> io::Result<()> {
        let mut files = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?; let name = entry.file_name(); let name = name.to_string_lossy();
            if name.starts_with("stack-v1-") && name.ends_with(".rtune") && entry.file_type()?.is_file() {
                files.push((entry.metadata()?.modified()?, entry.path()));
            }
        }
        files.sort_by(|a,b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        let extra = files.len().saturating_sub(self.capacity);
        for (_, path) in files.into_iter().take(extra) {
            if let Err(e) = fs::remove_file(path) { if e.kind() != io::ErrorKind::NotFound { return Err(e); } }
        }
        Ok(())
    }
    pub fn directory(&self) -> &Path { &self.root }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn record() -> Record { Record { scope: 0, key: "test/测试".into(), winner: "safe".into(), created: 50, reference_ns: 100, winner_ns: 75, ratio: 0.75, verified: true } }
    #[test] fn python_generated_golden_fixture() {
        let raw = include_bytes!("testdata/cache_v1.fixture");
        let record = Record::decode(raw).unwrap();
        assert_eq!(record.key, "known/测试"); assert_eq!(record.winner, "reference-plan");
        assert_eq!(record.scope, 1); assert_eq!(record.ratio, 0.8); assert_eq!(record.encode().unwrap(), raw.to_vec());
    }
    #[test] fn binary_round_trip() { let r = record(); let d = Record::decode(&r.encode().unwrap()).unwrap(); assert_eq!(d.key, r.key); assert_eq!(d.winner, r.winner); assert!(d.verified); }
    #[test] fn all_truncations_rejected() { let b = record().encode().unwrap(); for i in 0..b.len() { assert!(Record::decode(&b[..i]).is_err()); } }
    #[test] fn corruption_rejected() { let mut b = record().encode().unwrap(); b[14] ^= 1; assert!(Record::decode(&b).is_err()); }
    #[test] fn oversized_field_rejected() { let mut r = record(); r.key = "x".repeat(MAX_KEY+1); assert!(r.encode().is_err()); }
    #[test] fn nan_ratio_rejected() { let mut r = record(); r.ratio = f64::NAN; assert!(r.encode().is_err()); }
    #[test] fn zero_duration_rejected() { let mut r = record(); r.winner_ns = 0; assert!(r.encode().is_err()); }
    #[test] fn future_and_expired_records_miss() { let r = record(); assert!(!r.is_fresh(49, 100)); assert!(!r.is_fresh(150, 100)); assert!(r.is_fresh(149, 100)); }
    #[test] fn file_round_trip_and_prune() {
        let dir = std::env::temp_dir().join(std::format!("ruda-stack-test-{}-{}", std::process::id(), NEXT_FILE.fetch_add(1, Ordering::Relaxed)));
        let c = DiskCache::new(dir.clone(), 1); let r = record();
        c.save(&r).unwrap(); assert!(c.load(&r.key).unwrap().is_some());
        let mut r2 = r; r2.key = "another".into(); c.save(&r2).unwrap();
        assert_eq!(fs::read_dir(c.directory()).unwrap().count(), 1);
        fs::remove_dir_all(dir).unwrap();
    }
}
