//! Offline calibration on a single Amd device; no automatic model download.
type B = rullm::backends::Amd<half::bf16, i32, u8>;
#[path = "support/qwen2_autotune.rs"] mod common;
fn main() -> Result<(), Box<dyn std::error::Error>> { common::run() }
