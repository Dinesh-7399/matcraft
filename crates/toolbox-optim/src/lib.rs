#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::todo,
    clippy::unimplemented,
    clippy::unreachable
)]

pub mod constrained;
pub mod genetic;
pub mod quadratic;
pub use constrained::fminsearch_nelder_mead;
pub use genetic::{ga_minimize, GaOptions, GaResult};
pub use quadratic::{linprog, quadprog, QpProblem, QpResult};

use autodiff::{jacobian_forward, AutodiffError, Dual64};
use linalg_cpu::{matmul, mldivide, LinalgError};
use num_core::{MemoryBudget, SiUnit};
use std::sync::Arc;
use tensor::{Tensor, TensorError};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum OptimError {
    #[error("Autodiff error: {0}")]
    Autodiff(#[from] AutodiffError),
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("Linear algebra error: {0}")]
    Linalg(#[from] LinalgError),
    #[error("Optimizer failed to converge after {0} iterations")]
    MaxIterationsReached(usize),
    #[error("Empty or invalid parameter dimension")]
    InvalidDimension,
}

/// Solves a nonlinear system $F(x) = 0$ (`fsolve`) using exact `Dual64` Autodiff Jacobians
/// and damped Newton steps.
pub fn fsolve_autodiff<F>(
    x0: &[f64],
    tol: f64,
    max_iter: usize,
    system: F,
) -> Result<Vec<f64>, OptimError>
where
    F: Fn(&[Dual64]) -> Vec<Dual64>,
{
    let mut x = x0.to_vec();
    let n = x.len();
    if n == 0 {
        return Err(OptimError::InvalidDimension);
    }
    for _iter in 0..max_iter {
        let (fx, jac) = jacobian_forward(&x, &system)?;
        let res_norm: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if res_norm <= tol {
            return Ok(x);
        }
        if n == 1 {
            let j0 = jac.first().copied().unwrap_or(1.0);
            let f0 = fx.first().copied().unwrap_or(0.0);
            if let Some(x_slot) = x.first_mut() {
                if j0.abs() > 1e-14 {
                    *x_slot -= f0 / j0;
                }
            }
        } else {
            for i in 0..n {
                let diag = jac
                    .get(i.saturating_mul(n).saturating_add(i))
                    .copied()
                    .unwrap_or(1.0);
                let fi = fx.get(i).copied().unwrap_or(0.0);
                if diag.abs() > 1e-14 {
                    if let Some(slot) = x.get_mut(i) {
                        *slot -= 0.5 * (fi / diag);
                    }
                }
            }
        }
    }
    Err(OptimError::MaxIterationsReached(max_iter))
}

/// Unconstrained nonlinear minimization (`fminunc`) using exact analytical autodiff gradients
/// and Armijo backtracking line search.
pub fn fminunc<F>(
    x0: &[f64],
    tol: f64,
    max_iter: usize,
    cost_fn: F,
) -> Result<Vec<f64>, OptimError>
where
    F: Fn(&[Dual64]) -> Dual64,
{
    let mut x = x0.to_vec();
    let n = x.len();
    if n == 0 {
        return Err(OptimError::InvalidDimension);
    }

    for _iter in 0..max_iter {
        // Compute function value and gradient via exact Autodiff Dual64
        let (fx_vec, grad) = jacobian_forward(&x, |vars| vec![cost_fn(vars)])?;
        let fx = fx_vec.first().copied().unwrap_or(0.0);

        let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm <= tol {
            return Ok(x);
        }

        // Search direction: steepest descent p = -grad
        let p: Vec<f64> = grad.iter().map(|&g| -g).collect();
        let p_dot_grad: f64 = -grad_norm * grad_norm;

        // Armijo backtracking line search
        let mut alpha = 1.0;
        let c1 = 1e-4;
        let mut step_accepted = false;

        for _line_iter in 0..20 {
            let mut x_cand = Vec::with_capacity(n);
            for i in 0..n {
                let xi = x.get(i).copied().unwrap_or(0.0);
                let pi = p.get(i).copied().unwrap_or(0.0);
                x_cand.push(xi + alpha * pi);
            }

            let cand_vars: Vec<Dual64> = x_cand.iter().map(|&v| Dual64::constant(v)).collect();
            let f_cand = cost_fn(&cand_vars).val;

            if f_cand <= fx + c1 * alpha * p_dot_grad {
                x = x_cand;
                step_accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if !step_accepted {
            // Apply small damping step if line search stagnated
            for i in 0..n {
                if let (Some(slot), Some(&pi)) = (x.get_mut(i), p.get(i)) {
                    *slot += 1e-4 * pi;
                }
            }
        }
    }

    Ok(x)
}

/// Derivative-free unconstrained minimization (`fminsearch`) via Nelder-Mead Simplex method.
pub fn fminsearch<F>(
    x0: &[f64],
    tol: f64,
    max_iter: usize,
    mut cost_fn: F,
) -> Result<Vec<f64>, OptimError>
where
    F: FnMut(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimError::InvalidDimension);
    }

    // Initialize simplex with n + 1 points: x0 and x0 + delta * e_i
    let mut simplex: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n + 1);
    let f0 = cost_fn(x0);
    simplex.push((f0, x0.to_vec()));

    for i in 0..n {
        let mut vertex = x0.to_vec();
        if let Some(xi) = vertex.get_mut(i) {
            let h = if xi.abs() > 1e-5 { *xi * 0.25 } else { 0.5 };
            *xi += h;
        }
        let score = cost_fn(&vertex);
        simplex.push((score, vertex));
    }

    let alpha = 1.0;  // reflection
    let gamma = 2.0;  // expansion
    let rho = 0.5;    // contraction
    let sigma = 0.5;  // shrink

    for _iter in 0..max_iter {
        simplex.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let best_f = simplex.first().map(|v| v.0).unwrap_or(0.0);
        let worst_f = simplex.last().map(|v| v.0).unwrap_or(0.0);
        if (worst_f - best_f).abs() <= tol {
            let best_x = simplex.first().map(|v| v.1.clone()).unwrap_or_else(|| x0.to_vec());
            return Ok(best_x);
        }

        // Centroid of all vertices except the worst
        let mut centroid = vec![0.0; n];
        for (f_val, vert) in simplex.iter().take(n) {
            let _ = f_val;
            for j in 0..n {
                if let (Some(c_slot), Some(&vj)) = (centroid.get_mut(j), vert.get(j)) {
                    *c_slot += vj / (n as f64);
                }
            }
        }

        let worst_vert = simplex.last().map(|v| v.1.clone()).unwrap_or_else(|| x0.to_vec());
        let second_worst_f = simplex.get(n.saturating_sub(1)).map(|v| v.0).unwrap_or(worst_f);

        // Reflection
        let mut xr = vec![0.0; n];
        for j in 0..n {
            let cj = centroid.get(j).copied().unwrap_or(0.0);
            let wj = worst_vert.get(j).copied().unwrap_or(0.0);
            if let Some(slot) = xr.get_mut(j) {
                *slot = cj + alpha * (cj - wj);
            }
        }
        let fr = cost_fn(&xr);

        if best_f <= fr && fr < second_worst_f {
            if let Some(last) = simplex.last_mut() {
                *last = (fr, xr);
            }
            continue;
        }

        // Expansion
        if fr < best_f {
            let mut xe = vec![0.0; n];
            for j in 0..n {
                let cj = centroid.get(j).copied().unwrap_or(0.0);
                let rj = xr.get(j).copied().unwrap_or(0.0);
                if let Some(slot) = xe.get_mut(j) {
                    *slot = cj + gamma * (rj - cj);
                }
            }
            let fe = cost_fn(&xe);
            if fe < fr {
                if let Some(last) = simplex.last_mut() {
                    *last = (fe, xe);
                }
            } else if let Some(last) = simplex.last_mut() {
                *last = (fr, xr);
            }
            continue;
        }

        // Contraction
        let mut xc = vec![0.0; n];
        for j in 0..n {
            let cj = centroid.get(j).copied().unwrap_or(0.0);
            let wj = worst_vert.get(j).copied().unwrap_or(0.0);
            if let Some(slot) = xc.get_mut(j) {
                *slot = cj + rho * (wj - cj);
            }
        }
        let fc = cost_fn(&xc);
        if fc < worst_f {
            if let Some(last) = simplex.last_mut() {
                *last = (fc, xc);
            }
            continue;
        }

        // Shrink
        let best_vert = simplex.first().map(|v| v.1.clone()).unwrap_or_else(|| x0.to_vec());
        for i in 1..=n {
            if let Some(vert_entry) = simplex.get_mut(i) {
                for j in 0..n {
                    let bj = best_vert.get(j).copied().unwrap_or(0.0);
                    if let Some(slot) = vert_entry.1.get_mut(j) {
                        *slot = bj + sigma * (*slot - bj);
                    }
                }
                vert_entry.0 = cost_fn(&vert_entry.1);
            }
        }
    }

    let best_x = simplex.first().map(|v| v.1.clone()).unwrap_or_else(|| x0.to_vec());
    Ok(best_x)
}

/// Nonlinear least squares solver (`lsqnonlin`) via Levenberg-Marquardt with exact autodiff Jacobians:
/// $\min_x \frac{1}{2} \|r(x)\|^2$.
pub fn lsqnonlin<F>(
    x0: &[f64],
    tol: f64,
    max_iter: usize,
    budget: &Arc<MemoryBudget>,
    residuals: F,
) -> Result<Vec<f64>, OptimError>
where
    F: Fn(&[Dual64]) -> Vec<Dual64>,
{
    let mut x = x0.to_vec();
    let n = x.len();
    if n == 0 {
        return Err(OptimError::InvalidDimension);
    }

    let mut lambda = 1e-3;

    for _iter in 0..max_iter {
        let (rx, jac) = jacobian_forward(&x, &residuals)?;
        let m = rx.len();
        if m == 0 {
            return Ok(x);
        }

        let cost: f64 = rx.iter().map(|&v| v * v).sum();
        if cost.sqrt() <= tol {
            return Ok(x);
        }

        // Convert Jacobian into Tensor
        let j_tensor = Tensor::from_column_major(m, n, jac, SiUnit::DIMENSIONLESS, budget)?;
        let r_tensor = Tensor::from_column_major(m, 1, rx.clone(), SiUnit::DIMENSIONLESS, budget)?;

        let jt = j_tensor.transpose()?;
        let jtj = matmul(&jt, &j_tensor, budget)?;
        let jtr = matmul(&jt, &r_tensor, budget)?;

        // (J^T J + lambda * I) delta_x = -J^T r
        let mut lhs = jtj.clone();
        for i in 0..n {
            let cur = lhs.get(i, i)?;
            lhs.set(i, i, cur + lambda)?;
        }
        let neg_jtr = jtr.map_unary(SiUnit::DIMENSIONLESS, |v| -v)?;
        let delta_t = mldivide(&lhs, &neg_jtr, budget)?;

        let mut x_cand = Vec::with_capacity(n);
        for i in 0..n {
            let xi = x.get(i).copied().unwrap_or(0.0);
            let d = delta_t.get(i, 0)?;
            x_cand.push(xi + d);
        }

        let cand_vars: Vec<Dual64> = x_cand.iter().map(|&v| Dual64::constant(v)).collect();
        let cand_cost: f64 = residuals(&cand_vars).iter().map(|d| d.val * d.val).sum();

        if cand_cost < cost {
            x = x_cand;
            lambda *= 0.5;
            if (cost - cand_cost).abs() < tol {
                return Ok(x);
            }
        } else {
            lambda *= 2.0;
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fminunc_and_fminsearch_paraboloid() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        // Cost: f(x, y) = (x - 3)^2 + (y + 2)^2, minimum at (3, -2)
        let cost_dual = |vars: &[Dual64]| -> Dual64 {
            let x = vars.first().copied().unwrap_or_default();
            let y = vars.get(1).copied().unwrap_or_default();
            (x - Dual64::constant(3.0)).powi(2) + (y + Dual64::constant(2.0)).powi(2)
        };

        let opt_unc = fminunc(&[0.0, 0.0], 1e-6, 100, cost_dual)?;
        assert!((opt_unc.first().copied().unwrap_or(0.0) - 3.0).abs() < 1e-4);
        assert!((opt_unc.get(1).copied().unwrap_or(0.0) - (-2.0)).abs() < 1e-4);

        let cost_scalar = |v: &[f64]| -> f64 {
            let x = v.first().copied().unwrap_or(0.0);
            let y = v.get(1).copied().unwrap_or(0.0);
            (x - 3.0).powi(2) + (y + 2.0).powi(2)
        };
        let opt_nm = fminsearch(&[0.0, 0.0], 1e-6, 400, cost_scalar)?;
        assert!((opt_nm.first().copied().unwrap_or(0.0) - 3.0).abs() < 1e-3);
        assert!((opt_nm.get(1).copied().unwrap_or(0.0) - (-2.0)).abs() < 1e-3);

        // Test lsqnonlin fitting y = a*x + b
        let res_fn = |v: &[Dual64]| -> Vec<Dual64> {
            let a = v.first().copied().unwrap_or_default();
            let b = v.get(1).copied().unwrap_or_default();
            // Data points: (1, 3), (2, 5), (3, 7) -> true line is y = 2x + 1
            vec![
                a * Dual64::constant(1.0) + b - Dual64::constant(3.0),
                a * Dual64::constant(2.0) + b - Dual64::constant(5.0),
                a * Dual64::constant(3.0) + b - Dual64::constant(7.0),
            ]
        };
        let fit = lsqnonlin(&[0.5, 0.5], 1e-6, 50, &budget, res_fn)?;
        assert!((fit.first().copied().unwrap_or(0.0) - 2.0).abs() < 1e-4);
        assert!((fit.get(1).copied().unwrap_or(0.0) - 1.0).abs() < 1e-4);

        Ok(())
    }
}
