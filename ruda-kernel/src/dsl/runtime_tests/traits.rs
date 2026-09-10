use crate::dsl::prelude::*;

#[ruda]
pub(crate) trait UnaryOp: 'static + Send + Sync {
    type Options: LaunchArg;

    fn do_stuff<C: RudaPrimitive>(input: C, option: Self::Options) -> C;
}

#[ruda(launch)]
pub(crate) fn associated_type_input<O: UnaryOp>(_options: &O::Options) {}

pub struct Identity;

#[ruda]
impl UnaryOp for Identity {
    type Options = ();

    fn do_stuff<C: RudaPrimitive>(input: C, _option: Self::Options) -> C {
        input
    }
}

#[ruda]
pub(crate) fn trait_as<C: RudaPrimitive>(val: C) -> C {
    <Identity as UnaryOp>::do_stuff::<C>(val, ())
}
