use crate::dsl::prelude::*;

use crate::tiling::ruda_count::{RudaCountPlan, RudaCountPlanKind, GlobalOrder, swizzle};

#[derive(RudaType, RudaLaunch)]
/// Runtime-side counterpart of [RudaCountPlan]: given the ruda position,
/// resolves the conceptual `(x, y, z)` coordinates in problem space.
///
/// Each operation is responsible for mapping the returned generic `(x, y, z)`
/// tuple to its own domain axes (e.g. matmul interprets them as `(m, n, batch)`,
/// gemv as `(matrix_axis, _, batch)`, attention as `(seq_q, batch_heads, _)`).
pub struct RudaMapping {
    strategy: RudaMappingStrategy,
    #[ruda(comptime)]
    pub can_yield_extra_rudas: bool,
    #[ruda(comptime)]
    global_order: GlobalOrder,
}

#[derive(RudaType, RudaLaunch)]
/// [RudaCountPlanKind] stripped of non-essential runtime information.
///
/// Given as runtime input to kernels.
#[allow(unused)] // Constructed via RudaMappingStrategyArgs only
pub(crate) enum RudaMappingStrategy {
    FromProblem,
    SmFirst {
        x_rudas: u32,
        y_rudas: u32,
        z_rudas: u32,
    },
    RudaFirst {
        x_rudas: u32,
        y_rudas: u32,
        z_rudas: u32,
    },
    Flattened {
        x_rudas: u32,
        y_rudas: u32,
    },
    Spread {
        x_rudas: u32,
        y_rudas: u32,
        z_rudas: u32,
    },
}

#[ruda]
impl RudaMapping {
    /// Returns the number of valid rudas (problem-space volume).
    pub fn num_valid_rudas(&self) -> usize {
        match &self.strategy {
            RudaMappingStrategy::FromProblem | RudaMappingStrategy::Flattened { .. } => {
                panic!("Shouldn't need to be called because the ruda count should always be exact")
            }
            RudaMappingStrategy::SmFirst {
                x_rudas,
                y_rudas,
                z_rudas,
            }
            | RudaMappingStrategy::RudaFirst {
                x_rudas,
                y_rudas,
                z_rudas,
            }
            | RudaMappingStrategy::Spread {
                x_rudas,
                y_rudas,
                z_rudas,
            } => *x_rudas as usize * *y_rudas as usize * *z_rudas as usize,
        }
    }

    /// Given a ruda position, returns the generic problem-space coordinates `(x, y, z)`.
    ///
    /// Consumers assign meaning to `x/y/z` (matmul: `m/n/batch`, gemv: `matrix/_/batch`, etc.).
    pub fn ruda_pos_to_xyz(&self) -> (u32, u32, u32) {
        match &self.strategy {
            RudaMappingStrategy::FromProblem => (RUDA_POS_X, RUDA_POS_Y, RUDA_POS_Z),

            RudaMappingStrategy::SmFirst {
                x_rudas, y_rudas, ..
            } => {
                self.strategy
                    .absolute_index_to_xyz(RUDA_POS, *x_rudas, *y_rudas, self.global_order)
            }

            RudaMappingStrategy::RudaFirst {
                x_rudas, y_rudas, ..
            } => self.strategy.absolute_index_to_xyz(
                RUDA_POS_Y as usize * RUDA_COUNT_X as usize + RUDA_POS_X as usize,
                *x_rudas,
                *y_rudas,
                self.global_order,
            ),

            RudaMappingStrategy::Flattened { x_rudas, y_rudas } => self
                .strategy
                .absolute_index_to_xyz(RUDA_POS_X as usize, *x_rudas, *y_rudas, self.global_order),

            RudaMappingStrategy::Spread {
                x_rudas, y_rudas, ..
            } => {
                self.strategy
                    .absolute_index_to_xyz(RUDA_POS, *x_rudas, *y_rudas, self.global_order)
            }
        }
    }
}

#[ruda]
impl RudaMappingStrategy {
    fn absolute_index_to_xyz(
        &self,
        absolute_index: usize,
        x_rudas: u32,
        y_rudas: u32,
        #[comptime] global_order: GlobalOrder,
    ) -> (u32, u32, u32) {
        let z_stride = (x_rudas * y_rudas) as usize;
        let z_pos = absolute_index / z_stride;
        let xy_pos = absolute_index % z_stride;

        let (x_pos, y_pos) = match comptime!(global_order) {
            GlobalOrder::RowMajor => ((xy_pos / y_rudas as usize) as u32, xy_pos as u32 % y_rudas),
            GlobalOrder::ColMajor => (xy_pos as u32 % x_rudas, (xy_pos / x_rudas as usize) as u32),
            GlobalOrder::SwizzleRow(w) => {
                let (x, y) = swizzle(xy_pos, y_rudas as usize, w);
                (y, x)
            }
            GlobalOrder::SwizzleCol(w) => swizzle(xy_pos, x_rudas as usize, w),
        };

        (x_pos, y_pos, z_pos as u32)
    }
}

/// Build a [RudaMappingLaunch] from a resolved [RudaCountPlan].
pub fn ruda_mapping_launch<R: Runtime>(ruda_count_plan: &RudaCountPlan) -> RudaMappingLaunch<R> {
    RudaMappingLaunch::new(
        mapping_strategy(&ruda_count_plan.kind),
        ruda_count_plan.kind.can_yield_extra_rudas(),
        ruda_count_plan.global_order,
    )
}

fn mapping_strategy<R: Runtime>(
    ruda_count_plan_kind: &RudaCountPlanKind,
) -> RudaMappingStrategyArgs<R> {
    match ruda_count_plan_kind {
        RudaCountPlanKind::FromProblem { .. } => RudaMappingStrategyArgs::FromProblem,

        RudaCountPlanKind::Sm {
            rudas_first,
            problem_count,
            ..
        } => {
            if *rudas_first {
                RudaMappingStrategyArgs::RudaFirst {
                    x_rudas: problem_count.x,
                    y_rudas: problem_count.y,
                    z_rudas: problem_count.z,
                }
            } else {
                RudaMappingStrategyArgs::SmFirst {
                    x_rudas: problem_count.x,
                    y_rudas: problem_count.y,
                    z_rudas: problem_count.z,
                }
            }
        }

        RudaCountPlanKind::Flattened { problem_count, .. } => RudaMappingStrategyArgs::Flattened {
            x_rudas: problem_count.x,
            y_rudas: problem_count.y,
        },

        RudaCountPlanKind::Spread { problem_count, .. } => RudaMappingStrategyArgs::Spread {
            x_rudas: problem_count.x,
            y_rudas: problem_count.y,
            z_rudas: problem_count.z,
        },
    }
}
