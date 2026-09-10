use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;

use ruda_kernel::tiling::tile::{Plane, Tile};

#[derive(RudaType)]
/// Query input to the Tile Attention
pub struct Query<L: Numeric> {
    pub tile: Tile<L, Plane, ReadWrite>,
}

#[ruda]
impl<L: Numeric> Query<L> {
    pub fn new(tile: Tile<L, Plane, ReadWrite>) -> Query<L> {
        Query::<L> { tile }
    }
}

#[derive(RudaType)]
pub struct Key<R: Numeric> {
    pub tile: Tile<R, Plane, ReadWrite>,
}

#[ruda]
impl<R: Numeric> Key<R> {
    pub fn new(tile: Tile<R, Plane, ReadWrite>) -> Key<R> {
        Key::<R> { tile }
    }
}

#[derive(RudaType)]
pub struct Value<R: Numeric> {
    pub tile: Tile<R, Plane, ReadWrite>,
}

#[ruda]
impl<R: Numeric> Value<R> {
    pub fn new(tile: Tile<R, Plane, ReadWrite>) -> Value<R> {
        Value::<R> { tile }
    }
}
