//! Executes the production shared-IR compiler; no replacement emitter.
use ruda_compiler::ascend::{AscendCompiler, AscendOptions, AscendTarget,
    programs::{MapProgram, definition}, row_programs::{self, RowProgram}};
use ruda_core::{compiler::Compiler, ir::UIntKind, launch::ExecutionMode};
use std::{path::PathBuf, error::Error};
fn main() -> Result<(), Box<dyn Error>> {
    let mut op = "silu_mul".to_owned(); let mut elements = 1025u64;
    let mut out = None; let mut tile = 256; let mut cores = 32; let mut reuse = true;
    let mut row_width = None; let mut epsilon = 1e-5f32;
    let mut a = std::env::args().skip(1);
    while let Some(flag) = a.next() { match flag.as_str() {
        "--op" => op = a.next().ok_or("missing op")?,
        "--elements" => elements = a.next().ok_or("missing elements")?.parse()?,
        "--tile" => tile = a.next().ok_or("missing tile")?.parse()?,
        "--cores" => cores = a.next().ok_or("missing cores")?.parse()?,
        "--row-width" => row_width = Some(a.next().ok_or("missing row width")?.parse::<u32>()?),
        "--epsilon" => epsilon = a.next().ok_or("missing epsilon")?.parse()?,
        "--out" => out = Some(PathBuf::from(a.next().ok_or("missing output")?)),
        "--no-reuse" => reuse = false,
        _ => return Err(format!("unknown option {flag}").into()),
    }}
    let out = out.ok_or("--out is required")?;
    if out.exists() { return Err("refusing to overwrite output directory".into()); }
    let ir = if let Some(width) = row_width {
        row_programs::definition(RowProgram::parse(&op).ok_or("unknown row op")?, width, epsilon)?
    } else { definition(MapProgram::parse(&op).ok_or("unknown map op (row op requires --row-width)")?) };
    let ir_dump = format!("{ir:#?}");
    let kernel = AscendCompiler.compile(ir, &AscendOptions { target: Some(AscendTarget::Ascend950DT),
        elements, row_width, tile_elements: tile, vector_cores: cores, reuse_temporaries: reuse,
        ..Default::default() }, ExecutionMode::Checked, UIntKind::U64.into())?;
    std::fs::create_dir_all(&out)?;
    std::fs::write(out.join("kernel.asc"), kernel.source())?;
    std::fs::write(out.join("kernel.contract"), kernel.build_contract())?;
    std::fs::write(out.join("kernel.common-ir.txt"), ir_dump)?;
    println!("RUDA_ASCEND_IR_EMITTED entry={} elements={} row_width={:?} ub_bytes={} temporary_slots={} device_executed=false",
        kernel.entrypoint(), kernel.elements(), kernel.row_width(), kernel.ub_bytes(), kernel.temporary_slots());
    Ok(())
}
