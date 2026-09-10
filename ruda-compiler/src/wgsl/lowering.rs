use ruda_core::{ir::{Processor, Scope, Variable}, launch::ExecutionMode};

pub trait WgslLowering: Clone + Default + Send + Sync + 'static {
    fn processors(mode: ExecutionMode, kernel_name: String, max_vector_size: usize) -> [Box<dyn Processor>; 3];
    fn expand_erf(scope: &mut Scope, input: Variable, out: Variable);
    fn expand_integer_power(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable);
    fn expand_hypot(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable);
    fn expand_rhypot(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable);
    fn expand_himul_64(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable);
    fn expand_himul_sim(scope: &mut Scope, lhs: Variable, rhs: Variable, out: Variable);
}
