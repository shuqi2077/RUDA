use super::{Backend, Tensor, ToElement};
use core::convert::Infallible;

pub(super) trait VectorReductions<B: Backend> {
    type Error;

    fn dot(
        &mut self,
        lhs: &Tensor<B, 1>,
        rhs: &Tensor<B, 1>,
    ) -> Result<Tensor<B, 1>, Self::Error>;

    fn sum_abs(&mut self, value: &Tensor<B, 1>) -> Result<f64, Self::Error>;

    fn max_abs(&mut self, value: &Tensor<B, 1>) -> Result<f64, Self::Error>;
}

pub(super) struct LocalReductions;

impl<B: Backend> VectorReductions<B> for LocalReductions {
    type Error = Infallible;

    fn dot(
        &mut self,
        lhs: &Tensor<B, 1>,
        rhs: &Tensor<B, 1>,
    ) -> Result<Tensor<B, 1>, Self::Error> {
        Ok(lhs.clone().dot(rhs.clone()))
    }

    fn sum_abs(&mut self, value: &Tensor<B, 1>) -> Result<f64, Self::Error> {
        Ok(value.clone().abs().sum().into_scalar().to_f64())
    }

    fn max_abs(&mut self, value: &Tensor<B, 1>) -> Result<f64, Self::Error> {
        Ok(value.clone().abs().max().into_scalar().to_f64())
    }
}
