// SPDX-License-Identifier: Apache-2.0
//! Layout only: no GPU, allocation, host gradient data, or hidden fallback.
//! Used by the production training adapter before any kernel is submitted.
/// Maximum input rows merged by one warp in each hierarchy stage.
pub const STATS_FAN_IN: usize = 1024;
/// Native limit: 4096 active parameters with up to 1024 rows each.
pub const MAX_STATS_ROWS: usize = 4096 * 1024;

/// One disjoint output region in the reusable reduction workspace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatsStage {
    /// Number of rows read from the previous stage.
    pub input_rows: usize,
    /// Number of three-element rows written by this stage.
    pub output_rows: usize,
    /// Offset in the FP32 scratch buffer, not bytes or rows.
    pub output_offset: usize,
}
/// Bounded host layout used to launch actual device reduction kernels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatsPlan {
    /// Parallel reduction stages before the final report kernel.
    pub stages: Vec<StatsStage>,
    /// Total number of FP32 elements, excluding the initial statistics.
    pub scratch_elements: usize,
    /// Row count consumed by the final report kernel (at most 1024).
    pub final_rows: usize,
}
impl StatsPlan {
    /// Construct a plan, rejecting zero or more than MAX_STATS_ROWS rows.
    pub fn new(mut rows: usize) -> Result<Self, &'static str> {
        if rows == 0 || rows > MAX_STATS_ROWS {
            return Err("gradient statistics rows must be in 1..=4194304");
        }
        let mut stages = Vec::new();
        let mut elements = 0usize;
        while rows > STATS_FAN_IN {
            let output_rows = rows.div_ceil(STATS_FAN_IN);
            stages.push(StatsStage { input_rows: rows, output_rows, output_offset: elements });
            elements += output_rows * 3; // bounded by MAX_STATS_ROWS
            rows = output_rows;
        }
        Ok(Self { stages, scratch_elements: elements, final_rows: rows })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn small_rows_need_no_scratch() {
        for rows in [1, 31, 32, 1023, 1024] {
            let p = StatsPlan::new(rows).unwrap();
            assert!(p.stages.is_empty()); assert_eq!(p.scratch_elements, 0);
            assert_eq!(p.final_rows, rows);
        }
    }
    #[test] fn first_boundary() {
        let p = StatsPlan::new(1025).unwrap();
        assert_eq!(p.stages, vec![StatsStage { input_rows: 1025, output_rows: 2, output_offset: 0 }]);
        assert_eq!((p.scratch_elements, p.final_rows), (6, 2));
    }
    #[test] fn second_boundary() {
        let p = StatsPlan::new(1024 * 1024 + 1).unwrap();
        assert_eq!(p.stages.len(), 2);
        assert_eq!(p.stages[0].output_rows, 1025);
        assert_eq!(p.stages[1].output_offset, 3075);
        assert_eq!((p.scratch_elements, p.final_rows), (3081, 2));
    }
    #[test] fn maximum_layout() {
        let p = StatsPlan::new(MAX_STATS_ROWS).unwrap();
        assert_eq!(p.stages.len(), 2);
        assert_eq!(p.stages[0].output_rows, 4096);
        assert_eq!(p.stages[1].output_rows, 4);
        assert_eq!(p.scratch_elements * 4, 49200);
        assert_eq!(p.final_rows, 4);
    }
    #[test] fn bad_inputs_rejected() {
        for n in [0, MAX_STATS_ROWS + 1, usize::MAX] { assert!(StatsPlan::new(n).is_err()); }
    }
    #[test] fn stages_cover_every_row_and_do_not_overlap() {
        for n in (1..=MAX_STATS_ROWS).step_by(997) {
            let p = StatsPlan::new(n).unwrap();
            let mut previous_rows = n; let mut end = 0;
            for stage in &p.stages {
                assert_eq!(stage.input_rows, previous_rows);
                assert_eq!(stage.output_offset, end);
                assert!(stage.output_rows * STATS_FAN_IN >= previous_rows);
                assert!((stage.output_rows - 1) * STATS_FAN_IN < previous_rows);
                previous_rows = stage.output_rows; end += stage.output_rows * 3;
            }
            assert!(previous_rows <= STATS_FAN_IN);
            assert_eq!(p.scratch_elements, end);
        }
    }
}
