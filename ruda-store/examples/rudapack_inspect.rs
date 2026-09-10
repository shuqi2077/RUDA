//! Example: Generate a Rudapack file for inspection
//!
//! This example creates a simple Rudapack file that you can examine to understand the format.
//!
//! Usage:
//!   cargo run --example rudapack-inspect [output_path]
//!
//! Example:
//!   cargo run --example rudapack-inspect sample.bpk
//!   cargo run --example rudapack-inspect /tmp/test.bpk
//!
//! After generating the file, examine it with:
//!   hexdump -C sample.bpk | head -100
//!   xxd sample.bpk | head -100
//!   hexyl sample.bpk

use ruda_model::module::Module;
use ruda_tensor_host::Host;
use ruda_nn::{Linear, LinearConfig};
use ruda_store::{RudapackStore, ModuleSnapshot};
use ruda_tensor::api::backend::Backend;
use std::env;

// Simple model with a few layers
#[derive(Module, Debug)]
struct SampleModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
}

impl<B: Backend> SampleModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(128, 64).init(device),
            linear2: LinearConfig::new(64, 32).init(device),
            linear3: LinearConfig::new(32, 10).init(device),
        }
    }
}

fn main() {
    type Backend = Host;

    // Get output path from command line or use default
    let output_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "sample.bpk".to_string());

    println!("Creating sample Rudapack file: {}", output_path);
    println!();

    // Create a simple model
    let device = Default::default();
    let model = SampleModel::<Backend>::new(&device);

    // Save to Rudapack format with metadata
    let mut store = RudapackStore::from_file(&output_path)
        .overwrite(true)
        .metadata("format", "rudapack")
        .metadata("description", "Sample file for examining Rudapack format")
        .metadata("version", env!("CARGO_PKG_VERSION"))
        .metadata("author", "Ruda Example");

    model.save_into(&mut store).expect("Failed to save model");

    println!("✅ Successfully created: {}", output_path);
    println!();
    println!("📋 File Structure:");
    println!("  ┌─────────────────────────────────────┐");
    println!("  │ Header (10 bytes)                   │");
    println!("  ├─────────────────────────────────────┤");
    println!("  │ - Magic: 0x4E525542 (RUDA in LE)   │");
    println!("  │ - Version: 0x0001 (2 bytes)         │");
    println!("  │ - Metadata size: (4 bytes, u32 LE)  │");
    println!("  ├─────────────────────────────────────┤");
    println!("  │ Metadata (CBOR format)              │");
    println!("  ├─────────────────────────────────────┤");
    println!("  │ - Tensor descriptors                │");
    println!("  │   * name, dtype, shape, offsets     │");
    println!("  │ - User metadata                     │");
    println!("  ├─────────────────────────────────────┤");
    println!("  │ Tensor Data (raw bytes, LE)         │");
    println!("  ├─────────────────────────────────────┤");
    println!("  │ - linear1.weight [64, 128]          │");
    println!("  │ - linear1.bias [64]                 │");
    println!("  │ - linear2.weight [32, 64]           │");
    println!("  │ - linear2.bias [32]                 │");
    println!("  │ - linear3.weight [10, 32]           │");
    println!("  │ - linear3.bias [10]                 │");
    println!("  └─────────────────────────────────────┘");
    println!();
    println!("📊 Model Contents:");
    println!("  - linear1.weight: [64, 128] = 8,192 params → 32,768 bytes");
    println!("  - linear1.bias:   [64]      = 64 params    → 256 bytes");
    println!("  - linear2.weight: [32, 64]  = 2,048 params → 8,192 bytes");
    println!("  - linear2.bias:   [32]      = 32 params    → 128 bytes");
    println!("  - linear3.weight: [10, 32]  = 320 params   → 1,280 bytes");
    println!("  - linear3.bias:   [10]      = 10 params    → 40 bytes");
    println!("  ───────────────────────────────────────────────────────");

    let total_params = 8192 + 64 + 2048 + 32 + 320 + 10;
    let total_bytes = total_params * 4;
    println!(
        "  Total: {} parameters = {} KB",
        total_params,
        total_bytes / 1024
    );
    println!();

    // Get actual file size
    if let Ok(metadata) = std::fs::metadata(&output_path) {
        let file_size = metadata.len();
        println!(
            "📦 File size: {} bytes ({:.2} KB)",
            file_size,
            file_size as f64 / 1024.0
        );
    }

    println!();
    println!("🔍 Inspection Commands:");
    println!();
    println!("  # View first 100 bytes in hex:");
    println!("  hexdump -C {} | head -20", output_path);
    println!();
    println!("  # View header only (10 bytes):");
    println!("  head -c 10 {} | hexdump -C", output_path);
    println!();
    println!("  # View with prettier hex viewer (if installed):");
    println!("  hexyl {} | head -50", output_path);
    println!();
    println!("  # View in binary format:");
    println!("  xxd -b {} | head -20", output_path);
    println!();
    println!("  # Extract and examine header:");
    println!("  # Magic (bytes 0-3): Should be 42 55 52 4E (RUDA)");
    println!("  # Version (bytes 4-5): Should be 01 00");
    println!("  # Metadata size (bytes 6-9): u32 little-endian");
    println!();
    println!("  # Load back the model:");
    println!(
        "  # let mut store = RudapackStore::from_file(\"{}\");",
        output_path
    );
    println!("  # model.load_from(&mut store)?;");
}
