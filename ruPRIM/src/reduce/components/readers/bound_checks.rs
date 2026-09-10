use ruda_kernel::dsl as kernel_dsl;
use crate::reduce::{BoundChecks, ReduceInstruction, ReducePrecision};
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::tensor::View;
use ruda_kernel::library::tensor::layout::Coords1d;

#[derive(RudaType)]
#[allow(unused)]
pub enum ReaderBoundChecks<P: ReducePrecision> {
    NotRequired,
    Required(RequiredReaderBoundChecks<P>),
}

#[derive(RudaType)]
pub struct RequiredReaderBoundChecks<P: ReducePrecision> {
    #[ruda(comptime)]
    bound_checks: BoundChecks,
    pos_max: usize,
    null_input: Vector<P::EI, P::SI>,
}

#[ruda]
impl<P: ReducePrecision> ReaderBoundChecks<P> {
    pub fn new<I: ReduceInstruction<P>>(
        inst: &I,
        pos_max: usize,
        idle: ComptimeOption<bool>,
        #[comptime] bound_checks: BoundChecks,
    ) -> ReaderBoundChecks<P> {
        #[comptime]
        let pos_max = match idle {
            // When idle we set the pos_max to zero so that we always mask values.
            ComptimeOption::Some(idle) => pos_max * usize::cast_from(!idle),
            ComptimeOption::None => pos_max,
        };

        let bound_checks = comptime!(match (idle.is_some(), bound_checks) {
            (true, BoundChecks::None) => BoundChecks::Mask,
            _ => bound_checks,
        });

        match bound_checks {
            BoundChecks::None => ReaderBoundChecks::new_NotRequired(),
            BoundChecks::Mask | BoundChecks::Branch => {
                ReaderBoundChecks::new_Required(RequiredReaderBoundChecks::<P> {
                    bound_checks,
                    pos_max,
                    null_input: I::null_input(inst),
                })
            }
        }
    }
    pub fn read(
        &self,
        pos: usize,
        offset: usize,
        view: &View<Vector<P::EI, P::SI>, Coords1d>,
    ) -> Vector<P::EI, P::SI> {
        #[comptime]
        match self {
            ReaderBoundChecks::NotRequired => view[offset],
            ReaderBoundChecks::Required(checks) => match checks.bound_checks.comptime() {
                BoundChecks::None => view[offset],
                BoundChecks::Mask => {
                    let mask = pos < checks.pos_max;
                    let index = offset * usize::cast_from(mask);
                    select(mask, view[index], checks.null_input)
                }
                BoundChecks::Branch => {
                    if pos < checks.pos_max {
                        view[offset]
                    } else {
                        checks.null_input
                    }
                }
            },
        }
    }
}
