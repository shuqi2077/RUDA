use cfg_aliases::cfg_aliases;

fn main() {
    match stack_build_fingerprint() {
        Ok(id) => println!("cargo:rustc-env=RUDA_STACK_BUILD_ID={id}"),
        Err(error) => {
            println!("cargo:warning=Autotune build fingerprint unavailable: {error}");
            println!("cargo:rustc-env=RUDA_STACK_BUILD_ID=unavailable");
        }
    }

    // Setup cfg aliases
    cfg_aliases! {
        // Some features like autotune caching, compilation caching, and config loading
        // require std with OS-level filesystem and environment access.
        std_io: { all(feature = "runtime-std", any(target_os = "windows", target_os = "linux", target_os = "macos", target_os = "android")) },
        exclusive_memory_only: { any(feature = "runtime-exclusive-memory-only", target_family = "wasm") },
        multi_threading: { all(feature = "runtime-std", not(target_family = "wasm")) },
    }
}

/// Fingerprint checked-in Rust sources, manifests, lockfile and compilation environment.
/// This is a cache invalidation id, not a cryptographic signature or an assertion of correctness.
fn stack_build_fingerprint() -> std::io::Result<String> {
    use std::{env, fs, path::Path, process::Command};
    fn walk(dir: &Path, root: &Path, files: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?; let ty = entry.file_type()?; let path = entry.path();
            if ty.is_symlink() { continue; }
            if ty.is_dir() {
                let name = entry.file_name(); let name = name.to_string_lossy();
                if ["target", ".git", ".venv", "node_modules", "__pycache__"].contains(&name.as_ref()) { continue; }
                walk(&path, root, files)?;
            } else if path.extension().is_some_and(|s| ["rs", "h", "hpp", "c", "cc", "cpp", "cu", "cuh", "ptx", "wgsl", "mlir"].iter().any(|x| s == *x))
                || path.file_name().is_some_and(|s| s == "Cargo.toml" || s == "Cargo.lock") {
                let _ = root;
                println!("cargo:rerun-if-changed={}", path.display());
                files.push(path);
            }
        }
        Ok(())
    }
    let manifest = std::path::PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    // A packaged crate may live inside a large Cargo registry; never scan that registry.
    let parent = manifest.parent().unwrap_or(&manifest);
    let parent_manifest = fs::read_to_string(parent.join("Cargo.toml")).unwrap_or_default();
    let root = if parent_manifest.contains("[workspace]") { parent } else { &manifest };
    let mut files = Vec::new(); walk(root, root, &mut files)?; files.sort();
    let mut hash = 0xcbf29ce484222325u64;
    let mut hash2 = 0x84222325cbf29ce4u64;
    let mut feed = |bytes: &[u8]| {
        for byte in (bytes.len() as u64).to_le_bytes().iter().chain(bytes) {
            hash = (hash ^ *byte as u64).wrapping_mul(0x100000001b3);
            hash2 = (hash2 ^ (*byte as u64).wrapping_add(1)).wrapping_mul(0x100000001b3);
        }
    };
    for path in files {
        feed(path.strip_prefix(root).unwrap_or(&path).to_string_lossy().as_bytes());
        feed(&fs::read(path)?);
    }
    for name in ["TARGET", "HOST", "PROFILE", "OPT_LEVEL", "DEBUG", "CARGO_ENCODED_RUSTFLAGS", "RUSTFLAGS", "RUDA_AUTOTUNE_BUILD_TAG"] {
        println!("cargo:rerun-if-env-changed={name}"); feed(name.as_bytes());
        feed(env::var(name).unwrap_or_default().as_bytes());
    }
    let mut features: Vec<_> = env::vars().filter(|(k,_)| k.starts_with("CARGO_FEATURE_")).collect();
    features.sort();
    for (name,value) in features { feed(name.as_bytes()); feed(value.as_bytes()); }
    let compiler = Command::new(env::var_os("RUSTC").unwrap_or_else(|| "rustc".into())).arg("-vV").output()?;
    if !compiler.status.success() { return Err(std::io::Error::other("rustc version query failed")); }
    feed(&compiler.stdout);
    Ok(format!("source-v1-{hash:016x}{hash2:016x}"))
}
