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

//! Convex Quadratic Programming (`quadprog`) and Linear Programming (`linprog`)
//! via Primal-Dual Interior-Point method.

use crate::OptimError;

/// Specification of a convex Quadratic Programming (QP) problem:
///
/// Minimize:
///   `1/2 * x^T * H * x + f^T * x`
/// Subject to:
///   `A_ineq * x <= b_ineq`
///   `A_eq * x == b_eq`
///   `lb <= x <= ub`
#[derive(Clone, Debug, Default)]
pub struct QpProblem {
    /// Hessian matrix (n × n, row-major, symmetric positive semi-definite).
    /// If None, the problem reduces to Linear Programming (`linprog`).
    pub h: Option<Vec<f64>>,
    /// Linear objective vector (n × 1).
    pub f: Vec<f64>,
    /// Inequality constraint matrix (m × n, row-major).
    pub a_ineq: Option<Vec<f64>>,
    /// Inequality constraint RHS (m × 1).
    pub b_ineq: Option<Vec<f64>>,
    /// Number of inequality constraints.
    pub m_ineq: usize,
    /// Equality constraint matrix (meq × n, row-major).
    pub a_eq: Option<Vec<f64>>,
    /// Equality constraint RHS (meq × 1).
    pub b_eq: Option<Vec<f64>>,
    /// Number of equality constraints.
    pub m_eq: usize,
    /// Lower bounds (n × 1).
    pub lb: Option<Vec<f64>>,
    /// Upper bounds (n × 1).
    pub ub: Option<Vec<f64>>,
}

/// Result of QP / LP optimization.
#[derive(Clone, Debug, PartialEq)]
pub struct QpResult {
    /// Primal solution vector x*.
    pub x: Vec<f64>,
    /// Objective value at optimum: 1/2 x^T H x + f^T x.
    pub fval: f64,
    /// Number of interior-point iterations executed.
    pub iterations: usize,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
}

/// Convex Quadratic Programming solver (`quadprog`) using Primal-Dual Interior-Point Method.
pub fn quadprog(prob: &QpProblem, max_iter: usize, tol: f64) -> Result<QpResult, OptimError> {
    let n = prob.f.len();
    if n == 0 {
        return Err(OptimError::InvalidDimension);
    }

    // Convert bounds into extra inequality constraints:
    // x >= lb  =>  -x <= -lb
    // x <= ub  =>   x <=  ub
    let mut total_a_ineq = Vec::new();
    let mut total_b_ineq = Vec::new();
    let mut m_total_ineq = prob.m_ineq;

    if let (Some(a), Some(b)) = (&prob.a_ineq, &prob.b_ineq) {
        total_a_ineq.extend_from_slice(a);
        total_b_ineq.extend_from_slice(b);
    }

    if let Some(lb) = &prob.lb {
        for (i, &lo) in lb.iter().enumerate() {
            if lo.is_finite() {
                let mut row = vec![0.0; n];
                if let Some(slot) = row.get_mut(i) {
                    *slot = -1.0;
                }
                total_a_ineq.extend(row);
                total_b_ineq.push(-lo);
                m_total_ineq = m_total_ineq.saturating_add(1);
            }
        }
    }

    if let Some(ub) = &prob.ub {
        for (i, &hi) in ub.iter().enumerate() {
            if hi.is_finite() {
                let mut row = vec![0.0; n];
                if let Some(slot) = row.get_mut(i) {
                    *slot = 1.0;
                }
                total_a_ineq.extend(row);
                total_b_ineq.push(hi);
                m_total_ineq = m_total_ineq.saturating_add(1);
            }
        }
    }

    // Primal variable x
    let mut x = vec![0.0; n];
    // Slack variables s > 0
    let mut s = vec![1.0; m_total_ineq];
    // Dual variables lambda > 0
    let mut lambda = vec![1.0; m_total_ineq];
    // Dual variables nu for equalities
    let mut nu = vec![0.0; prob.m_eq];

    // Initialize s such that s_i > 0
    for i in 0..m_total_ineq {
        let mut ax = 0.0;
        for j in 0..n {
            let aij = total_a_ineq.get(i.saturating_mul(n).saturating_add(j)).copied().unwrap_or(0.0);
            ax += aij * x.get(j).copied().unwrap_or(0.0);
        }
        let bi = total_b_ineq.get(i).copied().unwrap_or(0.0);
        let si = (bi - ax).max(1.0);
        if let Some(slot) = s.get_mut(i) {
            *slot = si;
        }
    }

    let mut converged = false;
    let mut iter_count = 0;

    for iter in 0..max_iter {
        iter_count = iter + 1;

        // Duality gap mu = (s^T * lambda) / m
        let mu = if m_total_ineq > 0 {
            let mut dot = 0.0;
            for i in 0..m_total_ineq {
                dot += s.get(i).copied().unwrap_or(0.0) * lambda.get(i).copied().unwrap_or(0.0);
            }
            dot / (m_total_ineq as f64)
        } else {
            0.0
        };

        // Primal residual r_p = A_ineq * x + s - b_ineq
        let mut r_p = vec![0.0; m_total_ineq];
        let mut r_p_norm: f64 = 0.0;
        for i in 0..m_total_ineq {
            let mut ax = 0.0;
            for j in 0..n {
                let aij = total_a_ineq.get(i.saturating_mul(n).saturating_add(j)).copied().unwrap_or(0.0);
                ax += aij * x.get(j).copied().unwrap_or(0.0);
            }
            let res = ax + s.get(i).copied().unwrap_or(0.0) - total_b_ineq.get(i).copied().unwrap_or(0.0);
            r_p_norm = r_p_norm.max(res.abs());
            if let Some(slot) = r_p.get_mut(i) {
                *slot = res;
            }
        }

        // Dual residual r_d = H * x + f + A_ineq^T * lambda + A_eq^T * nu
        let mut r_d = vec![0.0; n];
        let mut r_d_norm: f64 = 0.0;
        for j in 0..n {
            let mut rd_j = prob.f.get(j).copied().unwrap_or(0.0);
            if let Some(h) = &prob.h {
                for k in 0..n {
                    let hjk = h.get(j.saturating_mul(n).saturating_add(k)).copied().unwrap_or(0.0);
                    rd_j += hjk * x.get(k).copied().unwrap_or(0.0);
                }
            }
            for i in 0..m_total_ineq {
                let aij = total_a_ineq.get(i.saturating_mul(n).saturating_add(j)).copied().unwrap_or(0.0);
                rd_j += aij * lambda.get(i).copied().unwrap_or(0.0);
            }
            if let Some(a_eq) = &prob.a_eq {
                for i in 0..prob.m_eq {
                    let aeq_ij = a_eq.get(i.saturating_mul(n).saturating_add(j)).copied().unwrap_or(0.0);
                    rd_j += aeq_ij * nu.get(i).copied().unwrap_or(0.0);
                }
            }
            r_d_norm = r_d_norm.max(rd_j.abs());
            if let Some(slot) = r_d.get_mut(j) {
                *slot = rd_j;
            }
        }

        if mu < tol && r_p_norm < tol && r_d_norm < tol {
            converged = true;
            break;
        }

        // Centering parameter
        let gamma = 0.1 * mu;

        // Form reduced KKT matrix and solve for [dx; dnu]
        let kkt_dim = n + prob.m_eq;
        let mut kkt_mat = vec![0.0; kkt_dim.saturating_mul(kkt_dim)];
        let mut kkt_rhs = vec![0.0; kkt_dim];

        // Fill H
        if let Some(h) = &prob.h {
            for j in 0..n {
                for k in 0..n {
                    let val = h.get(j.saturating_mul(n).saturating_add(k)).copied().unwrap_or(0.0);
                    if let Some(slot) = kkt_mat.get_mut(j.saturating_mul(kkt_dim).saturating_add(k)) {
                        *slot = val;
                    }
                }
            }
        }

        // Add A_ineq^T * diag(lambda / s) * A_ineq and small diagonal regularizer
        for i in 0..m_total_ineq {
            let si = s.get(i).copied().unwrap_or(1.0).max(1e-12);
            let li = lambda.get(i).copied().unwrap_or(1.0).max(1e-12);
            let weight = li / si;
            for j in 0..n {
                let aij = total_a_ineq.get(i.saturating_mul(n).saturating_add(j)).copied().unwrap_or(0.0);
                if aij == 0.0 {
                    continue;
                }
                for k in 0..n {
                    let aik = total_a_ineq.get(i.saturating_mul(n).saturating_add(k)).copied().unwrap_or(0.0);
                    if aik == 0.0 {
                        continue;
                    }
                    if let Some(slot) = kkt_mat.get_mut(j.saturating_mul(kkt_dim).saturating_add(k)) {
                        *slot += aij * aik * weight;
                    }
                }
            }
        }

        // Diagonal regularization for stability
        for j in 0..n {
            if let Some(slot) = kkt_mat.get_mut(j.saturating_mul(kkt_dim).saturating_add(j)) {
                *slot += 1e-8;
            }
        }

        // Fill equality constraints
        if let Some(a_eq) = &prob.a_eq {
            for i in 0..prob.m_eq {
                let row_idx = n + i;
                for j in 0..n {
                    let aeq_ij = a_eq.get(i.saturating_mul(n).saturating_add(j)).copied().unwrap_or(0.0);
                    if let Some(slot) = kkt_mat.get_mut(row_idx.saturating_mul(kkt_dim).saturating_add(j)) {
                        *slot = aeq_ij;
                    }
                    if let Some(slot) = kkt_mat.get_mut(j.saturating_mul(kkt_dim).saturating_add(row_idx)) {
                        *slot = aeq_ij;
                    }
                }
            }
        }

        // Form RHS: - (H*x + f) - A_ineq^T * (gamma / s + D * r_p)
        for j in 0..n {
            let mut hx_f = prob.f.get(j).copied().unwrap_or(0.0);
            if let Some(h) = &prob.h {
                for k in 0..n {
                    let hjk = h.get(j.saturating_mul(n).saturating_add(k)).copied().unwrap_or(0.0);
                    hx_f += hjk * x.get(k).copied().unwrap_or(0.0);
                }
            }
            let mut at_term = 0.0;
            for i in 0..m_total_ineq {
                let aij = total_a_ineq.get(i.saturating_mul(n).saturating_add(j)).copied().unwrap_or(0.0);
                let si = s.get(i).copied().unwrap_or(1.0).max(1e-12);
                let li = lambda.get(i).copied().unwrap_or(1.0);
                let rpi = r_p.get(i).copied().unwrap_or(0.0);
                let di = li / si;
                at_term += aij * (gamma / si + di * rpi);
            }
            if let Some(slot) = kkt_rhs.get_mut(j) {
                *slot = -hx_f - at_term;
            }
        }

        // Equality constraint residual on RHS: - (A_eq * x - b_eq)
        if let (Some(a_eq), Some(b_eq)) = (&prob.a_eq, &prob.b_eq) {
            for i in 0..prob.m_eq {
                let mut ax_eq = 0.0;
                for j in 0..n {
                    ax_eq += a_eq.get(i.saturating_mul(n).saturating_add(j)).copied().unwrap_or(0.0) * x.get(j).copied().unwrap_or(0.0);
                }
                let beq_i = b_eq.get(i).copied().unwrap_or(0.0);
                if let Some(slot) = kkt_rhs.get_mut(n + i) {
                    *slot = -(ax_eq - beq_i);
                }
            }
        }

        // Solve KKT system via partial-pivoting Gaussian elimination
        let sol = match gauss_solve(&kkt_mat, &kkt_rhs, kkt_dim) {
            Some(v) => v,
            None => {
                break;
            }
        };

        // Extract dx and dnu
        let mut dx = vec![0.0; n];
        for j in 0..n {
            if let Some(slot) = dx.get_mut(j) {
                *slot = sol.get(j).copied().unwrap_or(0.0);
            }
        }
        for i in 0..prob.m_eq {
            if let Some(slot) = nu.get_mut(i) {
                *slot = sol.get(n + i).copied().unwrap_or(0.0);
            }
        }

        // Compute ds and dlambda
        // ds = - r_p - A_ineq * dx
        // dlambda = - lambda + (gamma - lambda * ds) / s
        let mut ds = vec![0.0; m_total_ineq];
        let mut dlambda = vec![0.0; m_total_ineq];
        let mut step_primal = 1.0;
        let mut step_dual = 1.0;

        for i in 0..m_total_ineq {
            let mut adx = 0.0;
            for j in 0..n {
                let aij = total_a_ineq.get(i.saturating_mul(n).saturating_add(j)).copied().unwrap_or(0.0);
                adx += aij * dx.get(j).copied().unwrap_or(0.0);
            }
            let si = s.get(i).copied().unwrap_or(1.0);
            let li = lambda.get(i).copied().unwrap_or(1.0);
            let rpi = r_p.get(i).copied().unwrap_or(0.0);
            let dsi = -rpi - adx;
            let dli = -li + (gamma - li * dsi) / si.max(1e-12);

            if let Some(slot) = ds.get_mut(i) {
                *slot = dsi;
            }
            if let Some(slot) = dlambda.get_mut(i) {
                *slot = dli;
            }

            // Fraction to the boundary rule (0.995)
            if dsi < 0.0 {
                let ratio = -0.995 * si / dsi;
                if ratio < step_primal {
                    step_primal = ratio;
                }
            }
            if dli < 0.0 {
                let ratio = -0.995 * li / dli;
                if ratio < step_dual {
                    step_dual = ratio;
                }
            }
        }

        // Step updates
        for j in 0..n {
            if let Some(slot) = x.get_mut(j) {
                *slot += step_primal * dx.get(j).copied().unwrap_or(0.0);
            }
        }
        for i in 0..m_total_ineq {
            if let Some(slot) = s.get_mut(i) {
                *slot = (*slot + step_primal * ds.get(i).copied().unwrap_or(0.0)).max(1e-12);
            }
            if let Some(slot) = lambda.get_mut(i) {
                *slot = (*slot + step_dual * dlambda.get(i).copied().unwrap_or(0.0)).max(1e-12);
            }
        }
    }

    // Compute final objective value
    let mut fval = 0.0;
    if let Some(h) = &prob.h {
        for j in 0..n {
            let mut hx_j = 0.0;
            for k in 0..n {
                let hjk = h.get(j.saturating_mul(n).saturating_add(k)).copied().unwrap_or(0.0);
                hx_j += hjk * x.get(k).copied().unwrap_or(0.0);
            }
            fval += 0.5 * x.get(j).copied().unwrap_or(0.0) * hx_j;
        }
    }
    for j in 0..n {
        fval += prob.f.get(j).copied().unwrap_or(0.0) * x.get(j).copied().unwrap_or(0.0);
    }

    Ok(QpResult {
        x,
        fval,
        iterations: iter_count,
        converged,
    })
}

/// Linear Programming solver (`linprog`) for minimizing `f^T * x` subject to constraints.
pub fn linprog(
    f: &[f64],
    a_ineq: &[f64],
    b_ineq: &[f64],
    m_ineq: usize,
    lb: Option<&[f64]>,
    ub: Option<&[f64]>,
) -> Result<QpResult, OptimError> {
    let prob = QpProblem {
        h: None,
        f: f.to_vec(),
        a_ineq: Some(a_ineq.to_vec()),
        b_ineq: Some(b_ineq.to_vec()),
        m_ineq,
        a_eq: None,
        b_eq: None,
        m_eq: 0,
        lb: lb.map(|s| s.to_vec()),
        ub: ub.map(|s| s.to_vec()),
    };
    quadprog(&prob, 100, 1e-5)
}

/// Gaussian elimination solver with partial pivoting.
fn gauss_solve(mat: &[f64], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = mat.to_vec();
    let mut b = rhs.to_vec();

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a.get(col.saturating_mul(n).saturating_add(col)).copied().unwrap_or(0.0).abs();
        for row in (col + 1)..n {
            let v = a.get(row.saturating_mul(n).saturating_add(col)).copied().unwrap_or(0.0).abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return None;
        }
        if max_row != col {
            for j in 0..n {
                let idx1 = col.saturating_mul(n).saturating_add(j);
                let idx2 = max_row.saturating_mul(n).saturating_add(j);
                if idx1 < a.len() && idx2 < a.len() {
                    a.swap(idx1, idx2);
                }
            }
            if col < b.len() && max_row < b.len() {
                b.swap(col, max_row);
            }
        }

        let pivot = a.get(col.saturating_mul(n).saturating_add(col)).copied().unwrap_or(1.0);
        for row in (col + 1)..n {
            let factor = a.get(row.saturating_mul(n).saturating_add(col)).copied().unwrap_or(0.0) / pivot;
            if let Some(slot) = a.get_mut(row.saturating_mul(n).saturating_add(col)) {
                *slot = 0.0;
            }
            for j in (col + 1)..n {
                let aij = a.get(col.saturating_mul(n).saturating_add(j)).copied().unwrap_or(0.0);
                if let Some(slot) = a.get_mut(row.saturating_mul(n).saturating_add(j)) {
                    *slot -= factor * aij;
                }
            }
            let bk = b.get(col).copied().unwrap_or(0.0);
            if let Some(slot) = b.get_mut(row) {
                *slot -= factor * bk;
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b.get(i).copied().unwrap_or(0.0);
        for j in (i + 1)..n {
            sum -= a.get(i.saturating_mul(n).saturating_add(j)).copied().unwrap_or(0.0) * x.get(j).copied().unwrap_or(0.0);
        }
        let diag = a.get(i.saturating_mul(n).saturating_add(i)).copied().unwrap_or(1.0);
        if diag.abs() < 1e-15 {
            return None;
        }
        if let Some(slot) = x.get_mut(i) {
            *slot = sum / diag;
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadprog_unconstrained_paraboloid() -> Result<(), Box<dyn std::error::Error>> {
        let prob = QpProblem {
            h: Some(vec![1.0, 0.0, 0.0, 1.0]),
            f: vec![-2.0, -4.0],
            a_ineq: None,
            b_ineq: None,
            m_ineq: 0,
            a_eq: None,
            b_eq: None,
            m_eq: 0,
            lb: None,
            ub: None,
        };
        let res = quadprog(&prob, 50, 1e-5)?;
        assert!(res.converged, "Should converge");
        let x1 = res.x.first().copied().unwrap_or(0.0);
        let x2 = res.x.get(1).copied().unwrap_or(0.0);
        assert!((x1 - 2.0).abs() < 1e-2, "x1={x1}");
        assert!((x2 - 4.0).abs() < 1e-2, "x2={x2}");
        assert!((res.fval - (-10.0)).abs() < 1e-2, "fval={}", res.fval);
        Ok(())
    }

    #[test]
    fn test_quadprog_box_constrained() -> Result<(), Box<dyn std::error::Error>> {
        let prob = QpProblem {
            h: Some(vec![1.0, 0.0, 0.0, 1.0]),
            f: vec![-5.0, -5.0],
            a_ineq: None,
            b_ineq: None,
            m_ineq: 0,
            a_eq: None,
            b_eq: None,
            m_eq: 0,
            lb: Some(vec![0.0, 0.0]),
            ub: Some(vec![2.0, 3.0]),
        };
        let res = quadprog(&prob, 60, 1e-4)?;
        assert!(res.converged, "Should converge");
        let x1 = res.x.first().copied().unwrap_or(0.0);
        let x2 = res.x.get(1).copied().unwrap_or(0.0);
        assert!((x1 - 2.0).abs() < 0.05, "x1={x1}");
        assert!((x2 - 3.0).abs() < 0.05, "x2={x2}");
        Ok(())
    }

    #[test]
    fn test_linprog_diet_problem() -> Result<(), Box<dyn std::error::Error>> {
        let f = [2.0, 3.0];
        let a_ineq = [-1.0, -1.0, -2.0, -1.0];
        let b_ineq = [-4.0, -5.0];
        let lb = [0.0, 0.0];
        let res = linprog(&f, &a_ineq, &b_ineq, 2, Some(&lb), None)?;
        assert!(res.converged, "linprog should converge");
        let x1 = res.x.first().copied().unwrap_or(0.0);
        let x2 = res.x.get(1).copied().unwrap_or(0.0);
        assert!((x1 - 4.0).abs() < 0.05, "x1={x1}");
        assert!((x2 - 0.0).abs() < 0.05, "x2={x2}");
        assert!((res.fval - 8.0).abs() < 0.05, "fval={}", res.fval);
        Ok(())
    }
}
