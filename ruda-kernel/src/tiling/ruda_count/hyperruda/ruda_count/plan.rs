use crate::dsl::RudaCount;

use crate::tiling::ruda_count::{RudaCountStrategy, GlobalOrder, HyperrudaBlueprint, SmAllocation};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct RudaCountPlan {
    pub global_order: GlobalOrder,
    pub kind: RudaCountPlanKind,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Count3d {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl From<(u32, u32, u32)> for Count3d {
    fn from(value: (u32, u32, u32)) -> Self {
        Count3d {
            x: value.0,
            y: value.1,
            z: value.2,
        }
    }
}

impl Count3d {
    // Use u64 to avoid overflow when multiplying large ruda counts.
    pub(crate) fn total(&self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum RudaCountPlanKind {
    FromProblem {
        problem_count: Count3d,
    },
    Sm {
        rudas_first: bool,
        num_sms_used: u32,
        rudas_per_sm: u32,
        problem_count: Count3d,
        num_sms: u32,
        sm_usage: SmAllocation,
    },
    Flattened {
        problem_count: Count3d,
    },
    Spread {
        problem_count: Count3d,
        spread_count: Count3d,
    },
}

impl RudaCountPlan {
    // Will check if the wanted ruda count plan is possible, otherwise will fallback to spread
    pub fn from_blueprint(
        blueprint: &HyperrudaBlueprint,
        problem_count: Count3d,
        max_ruda_count: &(u32, u32, u32),
    ) -> RudaCountPlan {
        let (max_x, max_y, max_z) = *max_ruda_count;

        let plan_kind = match blueprint.ruda_count_strategy {
            RudaCountStrategy::FromProblem => {
                if problem_count.x > max_x || problem_count.y > max_y || problem_count.z > max_z {
                    None
                } else {
                    Some(RudaCountPlanKind::FromProblem { problem_count })
                }
            }
            RudaCountStrategy::Sm {
                rudas_first,
                num_sms,
                sm_usage,
            } => {
                let (num_sms_used, rudas_per_sm) =
                    sm_usage.allocate(num_sms, problem_count.total() as usize);

                if (rudas_per_sm >= if rudas_first { max_x } else { max_y })
                    || (num_sms_used >= if rudas_first { max_y } else { max_x })
                {
                    None
                } else {
                    Some(RudaCountPlanKind::Sm {
                        rudas_first,
                        num_sms_used,
                        rudas_per_sm,
                        problem_count,
                        num_sms,
                        sm_usage,
                    })
                }
            }
            RudaCountStrategy::Flattened => {
                if problem_count.total() >= max_x as u64 {
                    None
                } else {
                    Some(RudaCountPlanKind::Flattened { problem_count })
                }
            }
            RudaCountStrategy::Spread => None,
        };

        // Validate swizzle: fall back to non-swizzled order when the swizzle width
        // does not evenly divide the problem dimension (m_rudas for Row, n_rudas for Col).
        // Without this check, the swizzle produces incorrect ruda-to-tile mappings.
        let global_order = match blueprint.global_order {
            GlobalOrder::SwizzleRow(w) if !problem_count.x.is_multiple_of(w) => {
                GlobalOrder::RowMajor
            }
            GlobalOrder::SwizzleCol(w) if !problem_count.y.is_multiple_of(w) => {
                GlobalOrder::ColMajor
            }
            other => other,
        }
        .canonicalize();

        RudaCountPlan {
            global_order,
            kind: plan_kind
                .unwrap_or_else(|| spread_ruda_count_plan(problem_count, max_x, max_y, max_z)),
        }
    }

    pub fn new_from_problem(target_count: Count3d) -> Self {
        Self {
            global_order: Default::default(),
            kind: RudaCountPlanKind::FromProblem {
                problem_count: target_count,
            },
        }
    }

    pub fn can_yield_extra_rudas(&self) -> bool {
        self.kind.can_yield_extra_rudas()
    }

    pub fn resolve(&self) -> RudaCount {
        self.kind.resolve()
    }
}

impl RudaCountPlanKind {
    pub fn can_yield_extra_rudas(&self) -> bool {
        match self {
            RudaCountPlanKind::FromProblem { .. } | RudaCountPlanKind::Flattened { .. } => false,

            RudaCountPlanKind::Sm {
                num_sms_used,
                rudas_per_sm,
                problem_count,
                ..
            } => (num_sms_used * rudas_per_sm) as u64 != problem_count.total(),

            RudaCountPlanKind::Spread {
                problem_count,
                spread_count,
            } => problem_count.total() != spread_count.total(),
        }
    }

    fn resolve(&self) -> RudaCount {
        match self {
            RudaCountPlanKind::FromProblem { problem_count } => {
                RudaCount::Static(problem_count.x, problem_count.y, problem_count.z)
            }

            RudaCountPlanKind::Sm {
                rudas_first,
                num_sms_used,
                rudas_per_sm,
                ..
            } => {
                if *rudas_first {
                    RudaCount::Static(*rudas_per_sm, *num_sms_used, 1)
                } else {
                    RudaCount::Static(*num_sms_used, *rudas_per_sm, 1)
                }
            }

            RudaCountPlanKind::Flattened { problem_count } => {
                RudaCount::Static(problem_count.total() as u32, 1, 1)
            }

            RudaCountPlanKind::Spread { spread_count, .. } => {
                RudaCount::Static(spread_count.x, spread_count.y, spread_count.z)
            }
        }
    }
}

/// Heuristic algorithm to factor the total number of rudas into (x, y, z) dimensions
/// such that no dimension surpasses its maximum.
fn spread_ruda_count_plan(
    problem_count: Count3d,
    max_x: u32,
    max_y: u32,
    max_z: u32,
) -> RudaCountPlanKind {
    let mut best = None;

    let mut z = max_z;
    while z >= 1 {
        let xy_rudas = problem_count.total().div_ceil(z as u64);

        let mut y = max_y;
        while y >= 1 {
            let x64 = xy_rudas.div_ceil(y as u64);
            if x64 <= max_x as u64 {
                let x = x64 as u32;
                let volume = x as u64 * y as u64 * z as u64;
                let score = (volume, std::cmp::Reverse(z), std::cmp::Reverse(y));

                if best.is_none_or(|(_, _, _, _, best_score)| score < best_score) {
                    best = Some((x, y, z, volume, score));
                }
            }

            if y == 1 {
                break;
            }
            y /= 2;
        }

        if z == 1 {
            break;
        }
        z /= 2;
    }

    if let Some((x, y, z, _, _)) = best {
        RudaCountPlanKind::Spread {
            problem_count,
            spread_count: Count3d { x, y, z },
        }
    } else {
        panic!("No valid ruda spread plan")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX_RUDA_COUNT: (u32, u32, u32) = (65535, 65535, 65535);

    #[test]
    fn swizzle_row_falls_back_when_m_rudas_not_divisible_by_w() {
        // m_rudas=3 is not divisible by w=4, must fall back to RowMajor
        let blueprint = HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(4))
            .build();
        let plan = RudaCountPlan::from_blueprint(
            &blueprint,
            Count3d { x: 3, y: 5, z: 1 },
            &MAX_RUDA_COUNT,
        );
        assert_eq!(plan.global_order, GlobalOrder::RowMajor);
    }

    #[test]
    fn swizzle_row_kept_when_m_rudas_divisible_by_w() {
        let blueprint = HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(4))
            .build();
        let plan = RudaCountPlan::from_blueprint(
            &blueprint,
            Count3d { x: 8, y: 5, z: 1 },
            &MAX_RUDA_COUNT,
        );
        assert_eq!(plan.global_order, GlobalOrder::SwizzleRow(4));
    }

    #[test]
    fn swizzle_col_falls_back_when_n_rudas_not_divisible_by_w() {
        let blueprint = HyperrudaBlueprint::builder()
            .global_order(GlobalOrder::SwizzleCol(4))
            .build();
        let plan = RudaCountPlan::from_blueprint(
            &blueprint,
            Count3d { x: 8, y: 3, z: 1 },
            &MAX_RUDA_COUNT,
        );
        assert_eq!(plan.global_order, GlobalOrder::ColMajor);
    }
}
