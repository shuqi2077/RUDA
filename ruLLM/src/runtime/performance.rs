use super::{DeviceFingerprint, RuntimeError, TensorDType, TensorLayout};
use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PerformanceKey {
    pub device: DeviceFingerprint,
    /// Include a versioned algorithm name, e.g. packed-decode-v2.
    pub operation: String,
    pub dtype: TensorDType,
    pub layouts: Vec<TensorLayout>,
    /// Include precision/determinism settings; never mix accuracy policies.
    pub numerical_policy: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelMode { Automatic, Portable }

#[derive(Debug, Clone, Copy)]
pub struct PerformanceGuardConfig {
    pub capacity: usize,
    pub warmup_pairs: usize,
    pub measurement_pairs: usize,
    pub slowdown_ratio: f64,
}
impl Default for PerformanceGuardConfig {
    fn default() -> Self { Self { capacity: 128, warmup_pairs: 2, measurement_pairs: 7, slowdown_ratio: 1.15 } }
}
struct Measurements { seen: usize, ratios: VecDeque<f64>, portable: bool }

/// Bounded, explicitly driven regression guard. Timing is not injected into
/// every token step. Supply paired, synchronized, correctness-checked timings
/// under comparable load. Keep separate guards for unrelated workload regimes.
/// Eviction is FIFO; a regressed key stays portable until eviction or reset.
pub struct PerformanceGuard {
    config: PerformanceGuardConfig,
    entries: BTreeMap<PerformanceKey, Measurements>,
    order: VecDeque<PerformanceKey>,
}
impl PerformanceGuard {
    pub fn new(config: PerformanceGuardConfig) -> Result<Self, RuntimeError> {
        if config.capacity == 0 || config.measurement_pairs < 3
            || !config.slowdown_ratio.is_finite() || config.slowdown_ratio <= 1.0 {
            return Err(RuntimeError::invalid("invalid performance guard capacity, sample count or slowdown threshold"));
        }
        Ok(Self { config, entries: BTreeMap::new(), order: VecDeque::new() })
    }
    pub fn mode(&self, key: &PerformanceKey) -> KernelMode {
        if self.entries.get(key).is_some_and(|e| e.portable) { KernelMode::Portable } else { KernelMode::Automatic }
    }
    pub fn record_pair(&mut self, key: PerformanceKey, portable: Duration, optimized: Duration) -> Result<KernelMode, RuntimeError> {
        if portable.is_zero() || optimized.is_zero() {
            return Err(RuntimeError::invalid("zero duration is not a usable GPU timing"));
        }
        if !self.entries.contains_key(&key) {
            if self.entries.len() == self.config.capacity {
                if let Some(oldest) = self.order.pop_front() { self.entries.remove(&oldest); }
            }
            self.order.push_back(key.clone());
            self.entries.insert(key.clone(), Measurements { seen: 0, ratios: VecDeque::new(), portable: false });
        }
        let entry = self.entries.get_mut(&key).expect("measurement entry was inserted");
        if entry.seen < self.config.warmup_pairs { entry.seen += 1; return Ok(self.mode(&key)); }
        if entry.ratios.len() == self.config.measurement_pairs { entry.ratios.pop_front(); }
        entry.ratios.push_back(optimized.as_secs_f64() / portable.as_secs_f64());
        if entry.ratios.len() == self.config.measurement_pairs {
            let mut ratios: Vec<_> = entry.ratios.iter().copied().collect();
            ratios.sort_by(f64::total_cmp);
            let middle = ratios.len() / 2;
            let median = if ratios.len() % 2 == 0 { (ratios[middle - 1] + ratios[middle]) / 2.0 } else { ratios[middle] };
            if median > self.config.slowdown_ratio { entry.portable = true; }
        }
        Ok(self.mode(&key))
    }
    pub fn reset(&mut self) { self.entries.clear(); self.order.clear(); }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}
