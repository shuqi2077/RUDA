// SPDX-License-Identifier: Apache-2.0
//! One full 32-lane plane per block/system. All collectives are outside divergent
//! lane branches; all shared-memory hand-offs have explicit block barriers.
//! No warp-lockstep memory assumption and no per-lane early termination.
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

#[ruda(launch)]
pub(super) fn cholesky_solve_warp(
    a: &Array<f32>, b: &Array<f32>, lower: &mut Array<f32>, x: &mut Array<f32>,
    info: &mut Array<i32>, shift: f32, symmetry_atol: f32, symmetry_rtol: f32,
    #[comptime] n: usize, #[comptime] nr: usize, #[comptime] pitch: usize,
    #[comptime] _source: String,
) {
    let system = RUDA_POS_X as usize;
    // Uniform for the entire block, before any collective or barrier.
    if system >= info.len() { terminate!(); }
    let lane = UNIT_POS as usize;
    let base = system * n * n;
    let rb = system * n * nr;
    let mut matrix = SharedMemory::<f32>::new(n * pitch);
    let mut rhs = SharedMemory::<f32>::new(n * nr);
    let mut bad = false;
    let mut i = lane;
    while i < n * n {
        let value = a[base + i];
        matrix[(i / n) * pitch + i % n] = value;
        if value.is_nan() || value.is_inf() { bad = true; }
        i += 32;
    }
    i = lane;
    while i < n * nr {
        let value = b[rb + i];
        rhs[i] = value;
        if value.is_nan() || value.is_inf() { bad = true; }
        i += 32;
    }
    sync_ruda();
    let mut code = 0i32;
    if plane_any(bad) { code = -1; }
    if code == 0 {
        bad = false;
        if lane < n {
            for j in 0..lane {
                let u = matrix[lane * pitch + j];
                let v = matrix[j * pitch + lane];
                let scale = f32::max(u.abs(), v.abs());
                // Preserve the serial solver's max(atol/scale, rtol) contract.
                if scale > 0.0 {
                    if (u / scale - v / scale).abs() > f32::max(symmetry_atol / scale, symmetry_rtol) {
                        bad = true;
                    }
                }
            }
        }
        if plane_any(bad) { code = -2; }
    }
    // Every lane must have finished symmetry reads before factor writes.
    sync_ruda();
    for j in 0..n {
        if code == 0 {
            let mut pivot_code = 0i32;
            if lane == 0 {
                let mut diagonal = matrix[j * pitch + j] + shift;
                for k in 0..j {
                    let value = matrix[j * pitch + k];
                    diagonal = diagonal - value * value;
                }
                if diagonal.is_nan() || diagonal.is_inf() { pivot_code = -3; }
                else if diagonal <= 0.0 { pivot_code = (j + 1) as i32; }
                else { matrix[j * pitch + j] = diagonal.sqrt(); }
            }
            code = plane_broadcast(pivot_code, 0u32);
            sync_ruda();
            let mut row_code = 0i32;
            if code == 0 && lane > j && lane < n {
                let mut value = matrix[lane * pitch + j];
                // Ordered k accumulation matches the original algorithm; only
                // independent rows are parallelized (no tree-sum reassociation).
                for k in 0..j {
                    value = value - matrix[lane * pitch + k] * matrix[j * pitch + k];
                }
                value = value / matrix[j * pitch + j];
                if value.is_nan() || value.is_inf() { row_code = -3; }
                else { matrix[lane * pitch + j] = value; }
            }
            let all_rows_code = plane_min(row_code);
            if code == 0 { code = all_rows_code; }
            sync_ruda();
        }
    }
    if code == 0 {
        let mut solve_code = 0i32;
        // Each RHS column has one owner; dependencies within that column stay
        // sequential, but columns no longer serialize with each other.
        if lane < nr {
            for row in 0..n {
                let mut value = rhs[row * nr + lane];
                for k in 0..row {
                    value = value - matrix[row * pitch + k] * rhs[k * nr + lane];
                }
                value = value / matrix[row * pitch + row];
                if value.is_nan() || value.is_inf() { solve_code = -3; }
                else { rhs[row * nr + lane] = value; }
            }
            let mut end = n as usize;
            while end > 0 {
                let row = end - 1;
                let mut value = rhs[row * nr + lane];
                for k in row + 1..n {
                    value = value - matrix[k * pitch + row] * rhs[k * nr + lane];
                }
                value = value / matrix[row * pitch + row];
                if value.is_nan() || value.is_inf() { solve_code = -3; }
                else { rhs[row * nr + lane] = value; }
                end -= 1;
            }
        }
        code = plane_min(solve_code);
    }
    // RHS ownership changes to contiguous output lanes here.
    sync_ruda();
    i = lane;
    while i < n * n {
        let mut value = 0.0f32;
        if code == 0 && i / n >= i % n { value = matrix[(i / n) * pitch + i % n]; }
        lower[base + i] = value;
        i += 32;
    }
    i = lane;
    while i < n * nr {
        let mut value = 0.0f32;
        if code == 0 { value = rhs[i]; }
        x[rb + i] = value;
        i += 32;
    }
    if lane == 0 { info[system] = code; }
}

#[ruda(launch)]
pub(super) fn lu_solve_warp(
    a: &Array<f32>, b: &Array<f32>, lu: &mut Array<f32>, x: &mut Array<f32>,
    piv: &mut Array<i32>, info: &mut Array<i32>, atol: f32, rtol: f32,
    #[comptime] n: usize, #[comptime] nr: usize, #[comptime] pitch: usize,
    #[comptime] _source: String,
) {
    let system = RUDA_POS_X as usize;
    if system >= info.len() { terminate!(); }
    let lane = UNIT_POS as usize;
    let base = system * n * n;
    let rb = system * n * nr;
    let mut matrix = SharedMemory::<f32>::new(n * pitch);
    let mut rhs = SharedMemory::<f32>::new(n * nr);
    let mut pivots = SharedMemory::<i32>::new(n);
    let mut bad = false;
    let mut local_scale = 0.0f32;
    let mut i = lane;
    while i < n * n {
        let value = a[base + i];
        matrix[(i / n) * pitch + i % n] = value;
        if value.is_nan() || value.is_inf() { bad = true; }
        else { local_scale = f32::max(local_scale, value.abs()); }
        i += 32;
    }
    i = lane;
    while i < n * nr {
        let value = b[rb + i];
        rhs[i] = value;
        if value.is_nan() || value.is_inf() { bad = true; }
        i += 32;
    }
    if lane < n { pivots[lane] = lane as i32; }
    sync_ruda();
    let scale = plane_max(local_scale);
    let mut code = 0i32;
    if plane_any(bad) { code = -1; }
    if code == 0 {
        if scale == 0.0 { code = 1; }
        else {
            i = lane;
            while i < n * n {
                let pos = (i / n) * pitch + i % n;
                matrix[pos] = matrix[pos] / scale;
                i += 32;
            }
            bad = false;
            i = lane;
            while i < n * nr {
                let value = rhs[i] / scale;
                rhs[i] = value;
                if value.is_nan() || value.is_inf() { bad = true; }
                i += 32;
            }
            if plane_any(bad) { code = -3; }
        }
    }
    sync_ruda();
    for k in 0..n {
        if code == 0 {
            let mut candidate = 0.0f32;
            if lane >= k && lane < n { candidate = matrix[lane * pitch + k].abs(); }
            let best = plane_max(candidate);
            let mut candidate_row = n as u32;
            if lane >= k && lane < n && candidate == best { candidate_row = lane as u32; }
            // Lowest row on ties, matching the serial first-maximum pivot rule.
            let pivot = plane_min(candidate_row) as usize;
            // Finish all column reads before a different lane overwrites that row.
            sync_ruda();
            if best <= f32::max(atol / scale, rtol) { code = (k + 1) as i32; }
            else {
                if lane == 0 { pivots[k] = pivot as i32; }
                if pivot != k {
                    if lane < n {
                        let value = matrix[k * pitch + lane];
                        matrix[k * pitch + lane] = matrix[pivot * pitch + lane];
                        matrix[pivot * pitch + lane] = value;
                    }
                    if lane < nr {
                        let value = rhs[k * nr + lane];
                        rhs[k * nr + lane] = rhs[pivot * nr + lane];
                        rhs[pivot * nr + lane] = value;
                    }
                }
                sync_ruda();
                let mut row_code = 0i32;
                if lane > k && lane < n {
                    let ratio = matrix[lane * pitch + k] / matrix[k * pitch + k];
                    matrix[lane * pitch + k] = ratio;
                    if ratio.is_nan() || ratio.is_inf() { row_code = -3; }
                    for col in k + 1..n {
                        let value = matrix[lane * pitch + col] - ratio * matrix[k * pitch + col];
                        matrix[lane * pitch + col] = value;
                        if value.is_nan() || value.is_inf() { row_code = -3; }
                    }
                }
                code = plane_min(row_code);
                sync_ruda();
            }
        }
    }
    if code == 0 {
        let mut solve_code = 0i32;
        if lane < nr {
            for row in 0..n {
                let mut value = rhs[row * nr + lane];
                for k in 0..row { value = value - matrix[row * pitch + k] * rhs[k * nr + lane]; }
                rhs[row * nr + lane] = value;
            }
            let mut end = n as usize;
            while end > 0 {
                let row = end - 1;
                let mut value = rhs[row * nr + lane];
                for k in row + 1..n { value = value - matrix[row * pitch + k] * rhs[k * nr + lane]; }
                value = value / matrix[row * pitch + row];
                rhs[row * nr + lane] = value;
                if value.is_nan() || value.is_inf() { solve_code = -3; }
                end -= 1;
            }
        }
        code = plane_min(solve_code);
    }
    // No lane may rescale U while another lane still uses it in its solve.
    sync_ruda();
    if code == 0 {
        let mut scale_code = 0i32;
        i = lane;
        while i < n * n {
            if i / n <= i % n {
                let pos = (i / n) * pitch + i % n;
                let value = matrix[pos] * scale;
                matrix[pos] = value;
                if value.is_nan() || value.is_inf() { scale_code = -3; }
            }
            i += 32;
        }
        code = plane_min(scale_code);
    }
    sync_ruda();
    i = lane;
    while i < n * n {
        let mut value = 0.0f32;
        if code == 0 { value = matrix[(i / n) * pitch + i % n]; }
        lu[base + i] = value;
        i += 32;
    }
    i = lane;
    while i < n * nr {
        let mut value = 0.0f32;
        if code == 0 { value = rhs[i]; }
        x[rb + i] = value;
        i += 32;
    }
    if lane < n {
        let mut value = -1i32;
        if code == 0 { value = pivots[lane]; }
        piv[system * n + lane] = value;
    }
    if lane == 0 { info[system] = code; }
}
