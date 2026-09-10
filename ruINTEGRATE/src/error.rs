// SPDX-License-Identifier: Apache-2.0
use std::{
    error::Error, fmt
};
#[derive(Debug, Clone, PartialEq)]
pub enum IntegrationError {
    InvalidOption(&'static str),
    NonFiniteInput(&'static str),
    NonFiniteFunction {
        at: f64, index: usize
    },
    Arithmetic(&'static str),
    Callback(&'static str),
    Allocation,
}
impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>)->fmt::Result{
        match self{
            Self::InvalidOption(s)=>write!(f, "invalid integration option: {s}"),
            Self::NonFiniteInput(s)=>write!(f, "nonfinite integration input: {s}"),
            Self::NonFiniteFunction{
                at, index
            }=>write!(f, "callback returned nonfinite/missing element {index} at {at}"),
            Self::Arithmetic(s)=>write!(f, "nonfinite integration intermediate: {s}"),
            Self::Callback(s)=>write!(f, "callback error: {s}"),
            Self::Allocation=>write!(f, "integration workspace allocation failed"),
        }
    }
}
impl Error for IntegrationError{
}
pub(crate)fn checked(x: f64, where_: &'static str)->Result<f64, IntegrationError>{
    if x.is_finite(){
        Ok(x)
    } else{
        Err(IntegrationError::Arithmetic(where_))
    }
}
pub(crate)fn vector(n: usize)->Result<Vec<f64>, IntegrationError>{
    let mut v=Vec::new();
    v.try_reserve_exact(n).map_err(|_|IntegrationError::Allocation)?;
    v.resize(n, 0.0);
    Ok(v)
}
pub(crate)fn validate_tolerance(abs: f64, rel: f64)->Result<(), IntegrationError>{
    if !abs.is_finite()||!rel.is_finite()||abs<0.0||rel<0.0||(abs==0.0&&rel==0.0){
        Err(IntegrationError::InvalidOption("atol/rtol must be finite, nonnegative, and not both zero"))
    } else{
        Ok(())
    }
}
