fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "mlir")]
    // required on macos
    tracel_llvm_bundler::config::set_homebrew_library_path()?;
    Ok(())
}
