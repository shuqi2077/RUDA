//! Real Ascend row-program validation. Host calculations are references only.
//! No library stubs, CPU execution fallback or fake device success path.
use ruda_compiler::ascend::{AscendCompiler, AscendOptions, AscendTarget, row_programs::{self, RowProgram}};
use ruda_core::{compiler::Compiler, ir::UIntKind, launch::ExecutionMode};
use ruda_driver_cann::{CannDevice, tensor::{CannSession, common_ir::CannProgram}};

fn inputs(op: RowProgram, rows: usize, width: usize) -> Vec<Vec<f32>> {
    let x: Vec<f32> = (0..rows*width).map(|i| (i%41) as f32/7.0-2.0).collect();
    let dy: Vec<f32> = (0..rows*width).map(|i| (i%13) as f32/9.0-0.5).collect();
    let weight: Vec<f32> = (0..width).map(|i| 0.7+(i%7) as f32/13.0).collect();
    match op {
        RowProgram::RmsNorm => vec![x, weight],
        RowProgram::RmsNormInputBackward => {
            let r: Vec<f32> = x.chunks(width).map(|row| (row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()/width as f64+1e-5).sqrt().recip() as f32).collect();
            vec![x, dy, weight, r]
        }
        RowProgram::SoftmaxBackward | RowProgram::LogSoftmaxBackward => {
            let forward = if op == RowProgram::SoftmaxBackward { RowProgram::Softmax } else { RowProgram::LogSoftmax };
            let y = reference(forward, rows, width, &[x]).remove(0); vec![y, dy]
        }
        _ => vec![x],
    }
}
fn reference(op: RowProgram, rows: usize, width: usize, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let scalar = matches!(op, RowProgram::Sum | RowProgram::Mean | RowProgram::Max);
    let mut out = vec![vec![0.0; if scalar { rows } else { rows*width }]];
    if op == RowProgram::RmsNorm { out.push(vec![0.0; rows]); }
    for row in 0..rows {
        let offset = row*width; let a: Vec<f64> = x[0][offset..offset+width].iter().map(|&v| v as f64).collect();
        match op {
            RowProgram::Sum | RowProgram::Mean | RowProgram::Max => {
                let mut v = if op == RowProgram::Max { a.iter().copied().fold(f64::NEG_INFINITY, f64::max) } else { a.iter().sum() };
                if op == RowProgram::Mean { v /= width as f64; } out[0][row] = v as f32;
            }
            RowProgram::Softmax | RowProgram::LogSoftmax => {
                let m = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp: Vec<f64> = a.iter().map(|v| (v-m).exp()).collect(); let sum: f64 = exp.iter().sum();
                for j in 0..width { out[0][offset+j] = (if op == RowProgram::Softmax { exp[j]/sum } else { a[j]-m-sum.ln() }) as f32; }
            }
            RowProgram::RmsNorm => {
                let r = (a.iter().map(|v| v*v).sum::<f64>()/width as f64+1e-5).sqrt().recip();
                out[1][row] = r as f32;
                for j in 0..width { out[0][offset+j] = (a[j]*r*x[1][j] as f64) as f32; }
            }
            RowProgram::SoftmaxBackward | RowProgram::LogSoftmaxBackward => {
                let g: Vec<f64> = x[1][offset..offset+width].iter().map(|&v| v as f64).collect();
                let dot: f64 = if op == RowProgram::SoftmaxBackward { a.iter().zip(&g).map(|(y,g)| y*g).sum() } else { g.iter().sum() };
                for j in 0..width { out[0][offset+j] = (if op == RowProgram::SoftmaxBackward { a[j]*(g[j]-dot) } else { g[j]-a[j].exp()*dot }) as f32; }
            }
            RowProgram::RmsNormInputBackward => {
                let g: Vec<f64> = x[1][offset..offset+width].iter().zip(&x[2]).map(|(&v,&w)| v as f64*w as f64).collect();
                let r = x[3][row] as f64; let mean: f64 = g.iter().zip(&a).map(|(g,x)| g*x).sum::<f64>()/width as f64;
                for j in 0..width { out[0][offset+j] = (r*(g[j]-a[j]*r*r*mean)) as f32; }
            }
        }
    } out
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::PathBuf::from(std::env::args().nth(1).ok_or("usage: common_rows_validate ARTIFACT_ROOT")?);
    let acl = std::env::var("RUDA_CANN_LIBRARY").unwrap_or_else(|_| "libascendcl.so".into());
    let opapi = std::env::var("RUDA_CANN_OPAPI").unwrap_or_else(|_| "libopapi.so".into());
    // SAFETY: isolated process, exclusive context, trusted SDK/artifact directory.
    let session = unsafe { CannSession::open_exclusive(CannDevice::new(0)?, acl, opapi)? };
    let mut cases = 0; let mut launches = 0;
    for op in RowProgram::ALL { for (rows, width) in [(0usize, 32u32), (1, 32), (3, 96), (7, 256), (33, 4096)] {
        let compiled = AscendCompiler.compile(row_programs::definition(op, width, 1e-5)?, &AscendOptions {
            target: Some(AscendTarget::Ascend950DT), elements: rows as u64*width as u64, row_width: Some(width), ..Default::default()
        }, ExecutionMode::Checked, UIntKind::U64.into())?;
        let program = unsafe { CannProgram::load(&session, compiled, root.join(format!("{}-{rows}-{width}", op.name())))? };
        let host = inputs(op, rows, width as usize); let want = reference(op, rows, width as usize, &host);
        let device: Vec<_> = host.iter().map(|v| session.from_f32(&[v.len() as i64], v)).collect::<Result<_,_>>()?;
        let refs: Vec<_> = device.iter().collect(); let mut outputs = program.run(&refs)?;
        if outputs.len() != want.len() { return Err("wrong output count".into()); }
        for repeat in 0..2 {
            if repeat != 0 { let mut refs_out: Vec<_> = outputs.iter_mut().collect(); program.run_into(&refs, &mut refs_out)?; }
            for (out, expected) in outputs.iter().zip(&want) {
                let actual = out.to_f32()?;
                if actual.len() != expected.len() { return Err(format!("{} scalar/matrix output size mismatch", op.name()).into()); }
                for (j, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                    if !a.is_finite() || (a-b).abs() > 5e-4+5e-4*b.abs() {
                        return Err(format!("{} rows={rows} width={width} repeat={repeat} index={j}: {a} vs {b}", op.name()).into());
                    }
                }
            }
        }
        let expected_launches = if rows == 0 { 0 } else { 2 };
        if program.stats().launches != expected_launches { return Err("native launch count mismatch".into()); }
        if rows == 0 && program.stats().empty_calls != 2 { return Err("empty row calls were not explicit no-ops".into()); }
        launches += program.stats().launches; cases += 1;
        println!("RUDA_ASCEND_ROW_CASE op={} rows={rows} width={width} passed=true", op.name());
    }}
    if cases != 45 || launches != 72 { return Err("incomplete device coverage".into()); }
    println!("RUDA_ASCEND_ROWS_DEVICE_OK cases={cases} launches={launches}"); Ok(())
}
