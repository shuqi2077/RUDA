use ruda_kernel::dsl::prelude::*;
use ruda_kernel::dsl as kernel_dsl;

#[ruda]
fn array_variable(x: u32, y: u32) {
    let _array = [x, y];
}

fn main() {}
