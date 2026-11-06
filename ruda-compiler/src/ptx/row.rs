//! Launch alternatives for kernels where a group of threads owns one row.
//! A layout is a specialization input, not a rewrite of arbitrary kernel IR.
use super::{PtxCompilationOptions, PtxCompiler, PtxKernel, Result, invalid};
use ruda_core::{
    compiler::Compiler,
    ir::StorageType,
    kernel::KernelDefinition,
    launch::{ExecutionMode, RudaDim},
};

/// Each row uses whole warps. Independent rows may share the same block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowLayout {
    threads_per_row: u32,
    rows_per_block: u32,
}

impl RowLayout {
    pub fn threads_per_row(&self) -> u32 {
        self.threads_per_row
    }
    pub fn rows_per_block(&self) -> u32 {
        self.rows_per_block
    }

    /// Required one-dimensional CTA for a kernel specialized for this layout.
    pub fn block(&self) -> RudaDim {
        RudaDim::new_1d(self.threads_per_row * self.rows_per_block)
    }

    /// Zero means no work. A partial final block must mask its invalid rows.
    pub fn blocks(&self, rows: u32) -> u32 {
        rows.div_ceil(self.rows_per_block)
    }
}

/// Generate a bounded set of configurations, not a table of winning shapes.
/// Pass the number of vector items (not bytes) processed across one row.
/// `block_limit` must already reflect device and caller resource limits.
/// Kernels must support each returned layout and guard tails appropriately.
pub fn row_layouts(items_per_row: usize, block_limit: u32) -> Vec<RowLayout> {
    let mut out = Vec::new();
    if items_per_row == 0 {
        return out;
    }
    for threads in [32u32, 64, 128, 256, 512, 1024] {
        // One warp is the minimum participation unit. Beyond that, do not
        // double the group if the additional warp would be completely idle.
        if threads > block_limit || (threads > 32 && items_per_row <= threads as usize / 2) {
            break;
        }
        let mut previous = 0;
        for target in [threads, 128, 256] {
            let rows = (target / threads).max(1);
            if rows <= previous || threads * rows > block_limit {
                continue;
            }
            out.push(RowLayout {
                threads_per_row: threads,
                rows_per_block: rows,
            });
            previous = rows;
        }
    }
    out
}

/// Generate direct PTX for a *user-provided* cooperative row operator.
/// `build` specializes the IR (including any scratch storage) for each
/// layout; return `None` to reject a layout-specific resource or numerical
/// constraint. A compilation error is not a silent fallback. Rank successful
/// variants separately on the actual device after checking their results.
pub fn compile_row_variants(
    items_per_row: usize,
    block_limit: u32,
    options: &PtxCompilationOptions,
    mode: ExecutionMode,
    address_type: StorageType,
    mut build: impl FnMut(RowLayout) -> Option<KernelDefinition>,
) -> Result<Vec<(RowLayout, PtxKernel)>> {
    let mut variants = Vec::new();
    for layout in row_layouts(items_per_row, block_limit) {
        let Some(kernel) = build(layout) else {
            continue;
        };
        if kernel.ruda_dim != layout.block() {
            return Err(invalid(
                "row kernel block does not match its specialization",
            ));
        }
        let ptx = PtxCompiler.compile(kernel, options, mode, address_type)?;
        variants.push((layout, ptx));
    }
    Ok(variants)
}

/// Select the fastest *verified* variant for a device and workload. The caller
/// owns the device, stream, reference output and tolerance. Return a measured
/// duration (e.g. a median), or `None` for a mismatch / unavailable timing.
/// No measurement means no winner; a tie keeps the earlier layout.
/// Cache the selected layout, not timings or inputs, when reusing the result.
pub fn select_row_variant(
    variants: &[(RowLayout, PtxKernel)],
    mut verify_and_time: impl FnMut(RowLayout, &PtxKernel) -> Option<std::time::Duration>,
) -> Option<&(RowLayout, PtxKernel)> {
    variants
        .iter()
        .filter_map(|variant| {
            let time = verify_and_time(variant.0, &variant.1)?;
            (!time.is_zero()).then_some((time, variant))
        })
        .min_by_key(|(time, _)| *time)
        .map(|(_, variant)| variant)
}
