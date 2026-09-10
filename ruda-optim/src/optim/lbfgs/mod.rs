#![allow(clippy::excessive_precision)]


use super::GradientsParams;
use crate::LearningRate;
use ruda_model::config::Config;
use ruda_model::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use ruda_model::prelude::ToElement;
use ruda_model::record::Record;
use ruda_model::tensor::backend::Backend;
use ruda_model::tensor::{Tensor, backend::AutodiffBackend, container::TensorContainer};
use hashbrown::HashSet;
use serde::{Deserialize, Serialize};

use alloc::vec;
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

mod line_search;
use line_search::strong_wolfe;
#[cfg(test)]
use line_search::cubic_interpolate;

/// Strategy for the line search optimization phase
#[derive(Clone, Default, Debug, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineSearchFn {
    /// No line search performed
    #[default]
    None,
    /// strong wolfe conditions
    ///
    /// See: <https://en.wikipedia.org/wiki/Wolfe_conditions>
    StrongWolfe,
}

/// LBFGS Configuration.
#[derive(Config, Debug)]
pub struct LBFGSConfig {
    /// Maximal number of iterations per optimization step (default: 20)
    #[config(default = 20)]
    pub max_iter: usize,
    /// Update history size (default: 100).
    #[config(default = 100)]
    pub history_size: usize,
    /// Termination tolerance on first order optimality (default: 1e-7).
    #[config(default = 1e-7)]
    pub tolerance_grad: f64,
    /// Termination tolerance on function value/parameter changes (default: 1e-9).
    #[config(default = 1e-9)]
    pub tolerance_change: f64,
    /// Maximal number of function evaluations per optimization step (default: max_iter * 1.25).
    #[config(default = "None")]
    pub max_eval: Option<usize>,
    /// Either ‘strong_wolfe’ or None (default: None).
    #[config(default = "LineSearchFn::None")]
    pub line_search_fn: LineSearchFn,
}

impl LBFGSConfig {
    /// Initialize AdamW optimizer
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module
    pub fn init<B: AutodiffBackend>(&self) -> LBFGS<B> {
        // by default max_eval = max_iter * 5/4
        let max_eval = self.max_eval.unwrap_or(self.max_iter * 5 / 4);
        LBFGS {
            config: LBFGSConfig {
                max_iter: self.max_iter,
                history_size: self.history_size,
                tolerance_grad: self.tolerance_grad,
                tolerance_change: self.tolerance_change,
                max_eval: Some(max_eval),
                line_search_fn: self.line_search_fn,
            },
            state: Default::default(),
        }
    }
}

/// Collects gradients in module visit order.
struct FlattenGradsVisitorInner<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    tensors: &'a mut Vec<Tensor<B::InnerBackend, 1>>,
    seen: HashSet<ParamId>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for FlattenGradsVisitorInner<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let tensor = param.val();
        if !tensor.is_require_grad() || !self.seen.insert(param.id) {
            return;
        }
        let grad = self.grads.get::<B::InnerBackend, D>(param.id)
            .unwrap_or_else(|| tensor.inner().zeros_like());
        let numel = grad.shape().num_elements();
        self.tensors.push(grad.reshape([numel]));
    }
}

/// Flatten params to inner backend 1D tensor.
fn flatten_params_inner<B: AutodiffBackend, M: Module<B>>(
    module: &M,
) -> Option<Tensor<B::InnerBackend, 1>> {
    let mut tensors = Vec::new();
    let mut visitor = FlattenParamsVisitorInner::<B> {
        tensors: &mut tensors,
        seen: HashSet::new(),
    };
    module.visit(&mut visitor);
    if tensors.is_empty() {
        return None;
    }
    Some(Tensor::cat(tensors, 0))
}

struct FlattenParamsVisitorInner<'a, B: AutodiffBackend> {
    tensors: &'a mut Vec<Tensor<B::InnerBackend, 1>>,
    seen: HashSet<ParamId>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for FlattenParamsVisitorInner<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let tensor = param.val();
        if !tensor.is_require_grad() || !self.seen.insert(param.id) {
            return;
        }
        let t = tensor.inner();
        let numel = t.shape().num_elements();
        self.tensors.push(t.reshape([numel]));
    }
}

/// Flatten gradients for a module.
fn flatten_grads_inner<B: AutodiffBackend, M: Module<B>>(
    module: &M,
    grads: &GradientsParams,
) -> Tensor<B::InnerBackend, 1> {
    let mut tensors = Vec::new();
    let mut visitor = FlattenGradsVisitorInner {
        grads,
        tensors: &mut tensors,
        seen: HashSet::new(),
    };
    module.visit(&mut visitor);
    if tensors.is_empty() {
        return Tensor::empty([0], &module.devices()[0]);
    }
    Tensor::cat(tensors, 0)
}

/// Mapper that assigns each float param from a flat inner-backend 1D tensor.
struct ParamsFromFlatMapperInner<'a, B: AutodiffBackend> {
    flat: &'a Tensor<B::InnerBackend, 1>,
    offset: &'a mut usize,
    updated: TensorContainer<ParamId>,
}

impl<B: AutodiffBackend> ParamsFromFlatMapperInner<'_, B> {
    fn take_slice(&mut self, numel: usize) -> Tensor<B::InnerBackend, 1> {
        let start = *self.offset;
        *self.offset += numel;
        self.flat.clone().slice(start..*self.offset)
    }
}

impl<B: AutodiffBackend> ModuleMapper<B> for ParamsFromFlatMapperInner<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        if !tensor.is_require_grad() {
            return Param::from_mapped_value(id, tensor, mapper);
        }
        if let Some(updated) = self.updated.get::<B>(&id) {
            return Param::from_mapped_value(id, Tensor::from_primitive(updated), mapper);
        }
        let numel = tensor.shape().num_elements();
        let slice_1d = self.take_slice(numel);
        let new_inner = slice_1d.reshape(tensor.shape());
        let new_tensor = Tensor::from_inner(new_inner).require_grad();
        self.updated.register::<B>(id, new_tensor.clone().into_primitive());
        Param::from_mapped_value(id, new_tensor, mapper)
    }
}

/// Overwrite module parameters from a flat inner-backend 1D tensor
fn set_params_from_flat_inner<B: AutodiffBackend, M: Module<B>>(
    module: M,
    flat: Tensor<B::InnerBackend, 1>,
) -> M {
    let mut offset = 0;
    let mut mapper = ParamsFromFlatMapperInner {
        flat: &flat,
        offset: &mut offset,
        updated: TensorContainer::new(),
    };
    module.map(&mut mapper)
}

/// L-BFGS optimizer state
#[derive(Clone, Record)]
pub struct LBFGSState<B: Backend> {
    /// Historical displacement vectors
    pub history_s: Vec<Tensor<B, 1>>,
    /// Historical gradient difference vectors
    pub history_y: Vec<Tensor<B, 1>>,
    /// Search direction
    pub d: Option<Tensor<B, 1>>,
    /// Step size from the previous iteration
    pub t: Option<f64>,
    /// Flattened gradient from the previous iteration
    pub prev_flat_grad: Option<Tensor<B, 1>>,
    /// Loss value from the previous iteration
    pub prev_loss: Option<f64>,
    /// Global iteration count
    pub g_iter: usize,
}

impl<B: Backend> LBFGSState<B> {
    /// Moves all historical tensors to the target device.
    pub fn to_device(self, device: &B::Device) -> Self {
        Self {
            history_s: self
                .history_s
                .into_iter()
                .map(|t| t.to_device(device))
                .collect(),
            history_y: self
                .history_y
                .into_iter()
                .map(|t| t.to_device(device))
                .collect(),
            d: self.d.map(|t| t.to_device(device)),
            t: self.t,
            prev_flat_grad: self.prev_flat_grad.map(|t| t.to_device(device)),
            prev_loss: self.prev_loss,
            g_iter: self.g_iter,
        }
    }
}
impl<B: Backend> Default for LBFGSState<B> {
    fn default() -> Self {
        Self {
            history_s: Vec::new(),
            history_y: Vec::new(),
            d: None,
            t: Some(1.0),
            prev_flat_grad: None,
            prev_loss: None,
            g_iter: 0,
        }
    }
}

/// L-BFGS optimizer.
///
/// Ported from [pytorch](https://github.com/pytorch/pytorch/torch/optim/lbfgs.py). Heavily inspired by [miniFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
///
/// See also:
/// - [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
///
/// # Note
/// This optimizer is memory intensive
#[derive(Clone)]
pub struct LBFGS<B: Backend + AutodiffBackend> {
    config: LBFGSConfig,
    state: LBFGSState<B::InnerBackend>,
}

impl<B: Backend + AutodiffBackend> LBFGS<B> {
    /// Export the optimizer state for checkpointing.
    pub fn to_record(&self) -> LBFGSState<B::InnerBackend> {
        self.state.clone()
    }

    /// Restore optimizer state for the same configuration and ordered parameter layout.
    pub fn load_record(mut self, record: LBFGSState<B::InnerBackend>) -> Self {
        self.state = record;
        self
    }

    /// A single optimization step for any tensor that represents the parameters of a model.
    pub fn step<M, F>(&mut self, lr: LearningRate, mut module: M, mut closure: F) -> (M, f64)
    where
        M: AutodiffModule<B> + Clone,
        F: FnMut(M) -> (f64, GradientsParams),
    {
        // evaluate initial f(x) and df/dx
        let (mut loss, grads) = closure(module.clone());
        let mut current_evals = 1;
        if self.config.max_iter == 0 || current_evals >= self.config.max_eval.unwrap() {
            return (module, loss);
        }

        let Some(mut x_flat) = flatten_params_inner::<B, M>(&module) else {
            return (module, loss);
        };
        if x_flat.shape().num_elements() == 0 {
            return (module, loss);
        }
        let mut flat_grad = flatten_grads_inner::<B, M>(&module, &grads);

        let opt_cond =
            flat_grad.clone().abs().max().into_scalar().to_f64() <= self.config.tolerance_grad;
        // optimal condition
        if opt_cond {
            return (module, loss);
        }

        // tensors cached in state
        let mut d = self
            .state
            .d
            .take()
            .unwrap_or_else(|| flat_grad.clone().neg());
        let mut t = self.state.t.unwrap_or(lr);
        let mut prev_flat_grad = self.state.prev_flat_grad.take();

        let mut n_iter = 0;

        // optimize for a max of max_iter iterations
        while n_iter < self.config.max_iter {
            // keep track of nb of iterations
            n_iter += 1;
            self.state.g_iter += 1;

            // compute gradient descent direction
            if self.state.g_iter == 1 {
                d = flat_grad.clone().neg();
                self.state.history_s.clear();
                self.state.history_y.clear();
            } else {
                // do lbfgs update (update memory)
                if let Some(pg) = prev_flat_grad.as_ref() {
                    let y = flat_grad.clone().sub(pg.clone());
                    let s = d.clone().mul_scalar(t);

                    let ys = y.clone().dot(s.clone()).into_scalar().to_f64();

                    if ys > 1e-10 && self.config.history_size > 0 {
                        // updating memory
                        if self.state.history_s.len() >= self.config.history_size {
                            // shift history by one (limited-memory)
                            self.state.history_s.remove(0);
                            self.state.history_y.remove(0);
                        }
                        self.state.history_s.push(s);
                        self.state.history_y.push(y);
                    }
                }

                // compute the approximate (L-BFGS) inverse Hessian
                // multiplied by the gradient
                let num_old = self.state.history_s.len();
                let mut q = flat_grad.clone().neg();
                let mut alphas: Vec<Tensor<B::InnerBackend, 1>> =
                    vec![Tensor::zeros([1], &flat_grad.device()); num_old];

                if num_old > 0 {
                    // multiply by initial Hessian
                    // r/d is the final direction
                    for i in (0..num_old).rev() {
                        let s = &self.state.history_s[i];
                        let y = &self.state.history_y[i];
                        let rho = y.clone().dot(s.clone()).powf_scalar(-1.0);
                        let alpha = rho.clone().mul(s.clone().dot(q.clone()));
                        alphas[i] = alpha.clone();
                        q = q.sub(y.clone().mul(alpha));
                    }

                    let last_s = &self.state.history_s[num_old - 1];
                    let last_y = &self.state.history_y[num_old - 1];
                    let ys = last_y.clone().dot(last_s.clone());
                    let yy = last_y.clone().dot(last_y.clone());
                    let h_diag = ys.div(yy);

                    let mut r = q.mul(h_diag);

                    for ((s, y), alpha) in self
                        .state
                        .history_s
                        .iter()
                        .zip(self.state.history_y.iter())
                        .zip(alphas)
                        .take(num_old)
                    {
                        let rho = y.clone().dot(s.clone()).powf_scalar(-1.0);

                        let beta = rho.mul(y.clone().dot(r.clone()));

                        r = r.add(s.clone().mul(alpha.sub(beta)));
                    }
                    d = r;
                } else {
                    d = q;
                }
            }

            prev_flat_grad = Some(flat_grad.clone());
            let prev_loss_iter = loss;

            // compute step len
            if self.state.g_iter == 1 {
                let grad_l1 = flat_grad.clone().abs().sum().into_scalar().to_f64();
                t = (1.0f64 / grad_l1).min(1.0) * lr;
            } else {
                t = lr;
            }

            // directional derivative
            let gtd = flat_grad.clone().dot(d.clone()).into_scalar().to_f64();

            if gtd > -self.config.tolerance_change {
                break;
            }

            let ls_func_evals;

            if let LineSearchFn::StrongWolfe = self.config.line_search_fn {
                // perform line search, using user function
                let mut obj_func =
                    |current_x: &Tensor<B::InnerBackend, 1>,
                     step: f64,
                     dir: &Tensor<B::InnerBackend, 1>| {
                        let update = dir.clone().mul_scalar(step);
                        let new_x = current_x.clone().add(update);
                        let tmp_module = set_params_from_flat_inner::<B, M>(module.clone(), new_x);
                        let (l, g) = closure(tmp_module);
                        (l, flatten_grads_inner::<B, M>(&module, &g))
                    };

                let (ls_f, ls_g, ls_t, evals) = strong_wolfe(
                    &mut obj_func,
                    &x_flat,
                    t,
                    &d,
                    loss,
                    flat_grad.clone(),
                    gtd,
                    1e-4,
                    0.9,
                    self.config.tolerance_change,
                    self.config.max_eval.unwrap() - current_evals,
                );

                loss = ls_f;
                flat_grad = ls_g;
                t = ls_t;
                ls_func_evals = evals;

                x_flat = x_flat.add(d.clone().mul_scalar(t));
                module = set_params_from_flat_inner::<B, M>(module, x_flat.clone());
            } else {
                // no line search, simply move with fixed-step
                let step_vec = d.clone().mul_scalar(t);
                x_flat = x_flat.add(step_vec);
                module = set_params_from_flat_inner::<B, M>(module, x_flat.clone());
                // re-evaluate function only if not in last iteration
                // the reason we do this: in a stochastic setting,
                // no use to re-evaluate that function here
                let (new_loss, new_grads) = closure(module.clone());
                loss = new_loss;
                flat_grad = flatten_grads_inner::<B, M>(&module, &new_grads);
                ls_func_evals = 1;
            }

            // update func eval
            current_evals += ls_func_evals;

            // check conditions

            if current_evals >= self.config.max_eval.unwrap() {
                break;
            }

            if flat_grad.clone().abs().max().into_scalar().to_f64() <= self.config.tolerance_grad {
                break;
            }

            if d.clone().mul_scalar(t).abs().max().into_scalar().to_f64()
                <= self.config.tolerance_change
            {
                break;
            }

            if (loss - prev_loss_iter).abs() < self.config.tolerance_change {
                break;
            }
        }
        self.state.d = Some(d);
        self.state.t = Some(t);
        self.state.prev_flat_grad = prev_flat_grad;
        self.state.prev_loss = Some(loss);
        (module, loss)
    }
    /// Moves the optimizer state to the specified device.
    pub fn to_device(self, device: &B::Device) -> Self {
        Self {
            config: self.config,
            // History tensors reside in InnerBackend, so we convert the device accordingly
            state: self.state.to_device(device),
        }
    }
}

#[cfg(test)]
mod tests;
