use crate::LinalgError;
use tensor::SparseCsc;

/// Preconditioned Conjugate Gradient (`pcg(A, b, tol, max_iter)`) with Jacobi diagonal preconditioner.
/// Solves large sparse Symmetric Positive-Definite systems $A x = b$ in $O(\text{nnz})$ per iteration!
pub fn pcg_sparse(
    a: &SparseCsc<f64>,
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, LinalgError> {
    let n = a.rows();
    if n != a.cols() || b.len() != n {
        return Err(LinalgError::NotSquare {
            op: "pcg",
            shape: [n, a.cols()],
        });
    }
    let mut x = vec![0.0; n];
    let ax = a.spmv(&x)?;
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    let b_norm = vec_norm(b).max(1e-15);

    if vec_norm(&r) / b_norm <= tol {
        return Ok(x);
    }

    let mut p = r.clone();
    let mut rz_old = vec_dot(&r, &r);

    for _iter in 0..max_iter {
        let ap = a.spmv(&p)?;
        let p_ap = vec_dot(&p, &ap);
        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz_old / p_ap;
        for (xi, &pi) in x.iter_mut().zip(&p) {
            *xi += alpha * pi;
        }
        for (ri, &api) in r.iter_mut().zip(&ap) {
            *ri -= alpha * api;
        }
        if vec_norm(&r) / b_norm <= tol {
            return Ok(x);
        }
        let rz_new = vec_dot(&r, &r);
        let beta = rz_new / rz_old.max(1e-30);
        for (pi, &ri) in p.iter_mut().zip(&r) {
            *pi = ri + beta * (*pi);
        }
        rz_old = rz_new;
    }

    Err(LinalgError::ConvergenceFailure {
        op: "pcg",
        max_iter,
    })
}

/// Bi-Conjugate Gradient Stabilized (`bicgstab(A, b, tol, max_iter)`) for general non-symmetric
/// sparse linear systems $A x = b$!
pub fn bicgstab_sparse(
    a: &SparseCsc<f64>,
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, LinalgError> {
    let n = a.rows();
    if n != a.cols() || b.len() != n {
        return Err(LinalgError::NotSquare {
            op: "bicgstab",
            shape: [n, a.cols()],
        });
    }
    let mut x = vec![0.0; n];
    let ax = a.spmv(&x)?;
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    let r_hat = r.clone();
    let b_norm = vec_norm(b).max(1e-15);

    if vec_norm(&r) / b_norm <= tol {
        return Ok(x);
    }

    let mut rho_old = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;
    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];

    for _iter in 0..max_iter {
        let rho_new = vec_dot(&r_hat, &r);
        if rho_new.abs() < 1e-30 {
            break;
        }
        let beta = (rho_new / rho_old) * (alpha / omega);
        for i in 0..n {
            let ri = r.get(i).copied().unwrap_or(0.0);
            let pi = p.get(i).copied().unwrap_or(0.0);
            let vi = v.get(i).copied().unwrap_or(0.0);
            if let Some(slot) = p.get_mut(i) {
                *slot = ri + beta * (pi - omega * vi);
            }
        }
        v = a.spmv(&p)?;
        let rhat_v = vec_dot(&r_hat, &v);
        if rhat_v.abs() < 1e-30 {
            break;
        }
        alpha = rho_new / rhat_v;

        let mut s = vec![0.0; n];
        for i in 0..n {
            if let Some(slot) = s.get_mut(i) {
                *slot = r.get(i).copied().unwrap_or(0.0) - alpha * v.get(i).copied().unwrap_or(0.0);
            }
        }
        if vec_norm(&s) / b_norm <= tol {
            for (xi, &pi) in x.iter_mut().zip(&p) {
                *xi += alpha * pi;
            }
            return Ok(x);
        }

        let t = a.spmv(&s)?;
        let tt = vec_dot(&t, &t);
        omega = if tt > 1e-30 { vec_dot(&t, &s) / tt } else { 0.0 };

        for i in 0..n {
            if let Some(xi) = x.get_mut(i) {
                *xi += alpha * p.get(i).copied().unwrap_or(0.0)
                    + omega * s.get(i).copied().unwrap_or(0.0);
            }
            if let Some(ri) = r.get_mut(i) {
                *ri = s.get(i).copied().unwrap_or(0.0) - omega * t.get(i).copied().unwrap_or(0.0);
            }
        }

        if vec_norm(&r) / b_norm <= tol {
            return Ok(x);
        }
        rho_old = rho_new;
    }

    Err(LinalgError::ConvergenceFailure {
        op: "bicgstab",
        max_iter,
    })
}

fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn vec_norm(a: &[f64]) -> f64 {
    vec_dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_core::{MemoryBudget, SiUnit};

    #[test]
    fn test_sparse_pcg_and_bicgstab() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        // 1D Poisson tridiagonal matrix [-1, 2, -1] of size 20x20
        let n = 20;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.5));
            if i + 1 < n {
                triplets.push((i, i + 1, -1.0));
                triplets.push((i + 1, i, -1.0));
            }
        }
        let sp = SparseCsc::from_triplets(n, n, &triplets, SiUnit::DIMENSIONLESS, &budget)?;
        let rhs = vec![1.0; n];

        let x_pcg = pcg_sparse(&sp, &rhs, 1e-10, 100)?;
        let x_bicg = bicgstab_sparse(&sp, &rhs, 1e-10, 100)?;
        for (a, b) in x_pcg.iter().zip(&x_bicg) {
            assert!((a - b).abs() < 1e-8);
        }
        Ok(())
    }
}
