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

pub mod decomp_extra;
pub mod eigs_sparse;
pub mod riccati_schur;
pub mod sparse_solvers;

pub use decomp_extra::{bidiag, gram_schmidt, hessenberg};
pub use eigs_sparse::{eigs_sparse, EigsResult, EigsTarget};
pub use riccati_schur::{dlyap_discrete, lyap_continuous};
pub use sparse_solvers::{bicgstab_sparse, pcg_sparse};

use num_core::{Complex64, MemoryBudget, SiUnit, UnitError};
use std::f64::consts::PI;
use std::sync::Arc;
use tensor::{Tensor, TensorError};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum LinalgError {
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("Unit error: {0}")]
    Unit(#[from] UnitError),
    #[error("Matrix must be square for `{op}`, but got shape {shape:?}")]
    NotSquare { op: &'static str, shape: [usize; 2] },
    #[error("Matrix is singular or numerically rank-deficient (pivot {pivot_val:e} at index {index})")]
    SingularMatrix { index: usize, pivot_val: f64 },
    #[error("Matrix is not symmetric positive-definite for Cholesky decomposition")]
    NotPositiveDefinite,
    #[error("Iterative decomposition `{op}` did not converge within {max_iter} iterations")]
    ConvergenceFailure { op: &'static str, max_iter: usize },
}

pub fn matmul(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<Tensor<f64>, LinalgError> {
    if a.cols() != b.rows() {
        return Err(LinalgError::Tensor(TensorError::ShapeMismatch {
            op: "matmul",
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        }));
    }
    let out_unit = a.unit().checked_mul(b.unit())?;
    let m = a.rows();
    let k = a.cols();
    let n = b.cols();

    let mut out = vec![0.0; m.saturating_mul(n)];
    for j in 0..n {
        for p in 0..k {
            let bpj = b.get(p, j)?;
            if bpj == 0.0 {
                continue;
            }
            for i in 0..m {
                let aip = a.get(i, p)?;
                if let Some(slot) = out.get_mut(j.saturating_mul(m).saturating_add(i)) {
                    *slot += aip * bpj;
                }
            }
        }
    }
    Ok(Tensor::from_column_major(m, n, out, out_unit, budget)?)
}

#[derive(Clone, Debug)]
pub struct LuDecomposition {
    pub l: Tensor<f64>,
    pub u: Tensor<f64>,
    pub p: Tensor<f64>,
    pub piv_sign: f64,
}

pub fn lu(a: &Tensor<f64>, budget: &Arc<MemoryBudget>) -> Result<LuDecomposition, LinalgError> {
    let n = a.rows();
    if n != a.cols() {
        return Err(LinalgError::NotSquare {
            op: "lu",
            shape: a.shape(),
        });
    }
    let mut u = a.clone();
    let mut l = Tensor::eye(n, budget)?;
    let mut perm: Vec<usize> = (0..n).collect();
    let mut piv_sign = 1.0;

    for k in 0..n {
        let mut max_row = k;
        let mut max_val = u.get(k, k)?.abs();
        for i in (k + 1)..n {
            let v = u.get(i, k)?.abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-15 {
            return Err(LinalgError::SingularMatrix {
                index: k,
                pivot_val: max_val,
            });
        }
        if max_row != k {
            for col in 0..n {
                let tmp = u.get(k, col)?;
                u.set(k, col, u.get(max_row, col)?)?;
                u.set(max_row, col, tmp)?;
            }
            for col in 0..k {
                let tmp = l.get(k, col)?;
                l.set(k, col, l.get(max_row, col)?)?;
                l.set(max_row, col, tmp)?;
            }
            perm.swap(k, max_row);
            piv_sign = -piv_sign;
        }

        let pivot = u.get(k, k)?;
        for i in (k + 1)..n {
            let factor = u.get(i, k)? / pivot;
            l.set(i, k, factor)?;
            u.set(i, k, 0.0)?;
            for j in (k + 1)..n {
                let val = u.get(i, j)? - factor * u.get(k, j)?;
                u.set(i, j, val)?;
            }
        }
    }

    let mut p = Tensor::zeros(n, n, budget)?;
    for (row_idx, &orig_row) in perm.iter().enumerate() {
        p.set(row_idx, orig_row, 1.0)?;
    }

    Ok(LuDecomposition { l, u, p, piv_sign })
}

pub fn mldivide(
    a: &Tensor<f64>,
    b: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<Tensor<f64>, LinalgError> {
    if a.rows() != b.rows() {
        return Err(LinalgError::Tensor(TensorError::ShapeMismatch {
            op: "mldivide (\\)",
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        }));
    }
    let out_unit = b.unit().checked_div(a.unit())?;
    if a.rows() == a.cols() {
        let decomp = lu(a, budget)?;
        let n = a.rows();
        let nrhs = b.cols();
        let mut p_unitless = decomp.p;
        p_unitless.set_unit(SiUnit::DIMENSIONLESS);
        let pb = matmul(&p_unitless, b, budget)?;
        let mut x = Tensor::full_with_budget(n, nrhs, 0.0, out_unit, budget)?;

        for col in 0..nrhs {
            let mut y = vec![0.0; n];
            for i in 0..n {
                let mut sum = pb.get(i, col)?;
                for j in 0..i {
                    sum -= decomp.l.get(i, j)? * y.get(j).copied().unwrap_or(0.0);
                }
                if let Some(slot) = y.get_mut(i) {
                    *slot = sum;
                }
            }
            let mut x_col = vec![0.0; n];
            for i in (0..n).rev() {
                let mut sum = y.get(i).copied().unwrap_or(0.0);
                for j in (i + 1)..n {
                    sum -= decomp.u.get(i, j)? * x_col.get(j).copied().unwrap_or(0.0);
                }
                let u_ii = decomp.u.get(i, i)?;
                if u_ii.abs() < 1e-15 {
                    return Err(LinalgError::SingularMatrix {
                        index: i,
                        pivot_val: u_ii,
                    });
                }
                if let Some(slot) = x_col.get_mut(i) {
                    *slot = sum / u_ii;
                }
                x.set(i, col, x_col.get(i).copied().unwrap_or(0.0))?;
            }
        }
        Ok(x)
    } else {
        let (q, r) = qr(a, budget)?;
        let qt = q.transpose()?;
        let qtb = matmul(&qt, b, budget)?;
        let n = a.cols();
        let nrhs = b.cols();
        let mut x = Tensor::full_with_budget(n, nrhs, 0.0, out_unit, budget)?;
        for col in 0..nrhs {
            let mut x_col = vec![0.0; n];
            for i in (0..n).rev() {
                let mut sum = qtb.get(i, col)?;
                for j in (i + 1)..n {
                    sum -= r.get(i, j)? * x_col.get(j).copied().unwrap_or(0.0);
                }
                let r_ii = r.get(i, i)?;
                if r_ii.abs() < 1e-15 {
                    return Err(LinalgError::SingularMatrix {
                        index: i,
                        pivot_val: r_ii,
                    });
                }
                if let Some(slot) = x_col.get_mut(i) {
                    *slot = sum / r_ii;
                }
                x.set(i, col, x_col.get(i).copied().unwrap_or(0.0))?;
            }
        }
        Ok(x)
    }
}

pub fn qr(
    a: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<(Tensor<f64>, Tensor<f64>), LinalgError> {
    let m = a.rows();
    let n = a.cols();
    let k_max = m.min(n);
    let mut r = a.clone();
    let mut q = Tensor::eye(m, budget)?;

    for k in 0..k_max {
        let mut norm_x: f64 = 0.0;
        for i in k..m {
            let v = r.get(i, k)?;
            norm_x = norm_x.hypot(v);
        }
        if norm_x < 1e-15 {
            continue;
        }
        let r_kk = r.get(k, k)?;
        let alpha = if r_kk >= 0.0 { -norm_x } else { norm_x };
        let mut v = vec![0.0; m - k];
        if let Some(first) = v.first_mut() {
            *first = r_kk - alpha;
        }
        for i in (k + 1)..m {
            if let Some(slot) = v.get_mut(i - k) {
                *slot = r.get(i, k)?;
            }
        }
        let mut norm_v: f64 = 0.0;
        for &vi in &v {
            norm_v = norm_v.hypot(vi);
        }
        if norm_v < 1e-15 {
            continue;
        }
        for vi in &mut v {
            *vi /= norm_v;
        }

        for j in k..n {
            let mut dot = 0.0;
            for i in k..m {
                dot += v.get(i - k).copied().unwrap_or(0.0) * r.get(i, j)?;
            }
            for i in k..m {
                let updated = r.get(i, j)? - 2.0 * v.get(i - k).copied().unwrap_or(0.0) * dot;
                r.set(i, j, updated)?;
            }
        }

        for i in 0..m {
            let mut dot = 0.0;
            for l in k..m {
                dot += q.get(i, l)? * v.get(l - k).copied().unwrap_or(0.0);
            }
            for l in k..m {
                let updated = q.get(i, l)? - 2.0 * dot * v.get(l - k).copied().unwrap_or(0.0);
                q.set(i, l, updated)?;
            }
        }
    }

    Ok((q, r))
}

pub fn chol(a: &Tensor<f64>, budget: &Arc<MemoryBudget>) -> Result<Tensor<f64>, LinalgError> {
    let n = a.rows();
    if n != a.cols() {
        return Err(LinalgError::NotSquare {
            op: "chol",
            shape: a.shape(),
        });
    }
    let out_unit = a.unit().checked_sqrt()?;
    let mut l = Tensor::full_with_budget(n, n, 0.0, out_unit, budget)?;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l.get(i, k)? * l.get(j, k)?;
            }
            if i == j {
                let diag = a.get(i, i)? - sum;
                if diag <= 0.0 {
                    return Err(LinalgError::NotPositiveDefinite);
                }
                l.set(i, j, diag.sqrt())?;
            } else {
                let l_jj = l.get(j, j)?;
                l.set(i, j, (a.get(i, j)? - sum) / l_jj)?;
            }
        }
    }
    Ok(l)
}

pub fn det(a: &Tensor<f64>, budget: &Arc<MemoryBudget>) -> Result<f64, LinalgError> {
    match lu(a, budget) {
        Ok(decomp) => {
            let mut prod = decomp.piv_sign;
            for i in 0..a.rows() {
                prod *= decomp.u.get(i, i)?;
            }
            Ok(prod)
        }
        Err(LinalgError::SingularMatrix { .. }) => Ok(0.0),
        Err(other) => Err(other),
    }
}

pub fn inv(a: &Tensor<f64>, budget: &Arc<MemoryBudget>) -> Result<Tensor<f64>, LinalgError> {
    let n = a.rows();
    if n != a.cols() {
        return Err(LinalgError::NotSquare {
            op: "inv",
            shape: a.shape(),
        });
    }
    let eye = Tensor::eye(n, budget)?;
    mldivide(a, &eye, budget)
}

pub fn eig_sym(
    a: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<(Tensor<f64>, Tensor<f64>), LinalgError> {
    let n = a.rows();
    if n != a.cols() {
        return Err(LinalgError::NotSquare {
            op: "eig",
            shape: a.shape(),
        });
    }
    let mut d = a.clone();
    let mut v = Tensor::eye(n, budget)?;
    let max_sweeps = 100;

    for _sweep in 0..max_sweeps {
        let mut off_norm: f64 = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off_norm = off_norm.hypot(d.get(p, q)?);
            }
        }
        if off_norm < 1e-13 {
            return Ok((v, d));
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = d.get(p, q)?;
                if apq.abs() < 1e-15 {
                    continue;
                }
                let app = d.get(p, p)?;
                let aqq = d.get(q, q)?;
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                d.set(p, p, app - t * apq)?;
                d.set(q, q, aqq + t * apq)?;
                d.set(p, q, 0.0)?;
                d.set(q, p, 0.0)?;

                for r in 0..n {
                    if r != p && r != q {
                        let drp = d.get(r, p)?;
                        let drq = d.get(r, q)?;
                        d.set(r, p, c * drp - s * drq)?;
                        d.set(p, r, c * drp - s * drq)?;
                        d.set(r, q, s * drp + c * drq)?;
                        d.set(q, r, s * drp + c * drq)?;
                    }
                    let vrp = v.get(r, p)?;
                    let vrq = v.get(r, q)?;
                    v.set(r, p, c * vrp - s * vrq)?;
                    v.set(r, q, s * vrp + c * vrq)?;
                }
            }
        }
    }
    Ok((v, d))
}

/// General (Non-Symmetric) Eigenvalues (`eig(A)`) returning `Vec<Complex64>` via Upper Hessenberg
/// Householder reduction followed by Shifted QR / 2x2 real-Schur block extraction!
pub fn eig_general(
    a: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<Vec<Complex64>, LinalgError> {
    let n = a.rows();
    if n != a.cols() {
        return Err(LinalgError::NotSquare {
            op: "eig",
            shape: a.shape(),
        });
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![Complex64::new(a.get(0, 0)?, 0.0)]);
    }

    let mut h = a.clone();
    h.set_unit(SiUnit::DIMENSIONLESS);

    // 1. Reduce to Upper Hessenberg form via Householder similarity transforms
    for k in 0..n.saturating_sub(2) {
        let m_len = n - (k + 1);
        let mut x_norm: f64 = 0.0;
        for i in (k + 1)..n {
            x_norm = x_norm.hypot(h.get(i, k)?);
        }
        if x_norm < 1e-15 {
            continue;
        }
        let x0 = h.get(k + 1, k)?;
        let alpha = if x0 >= 0.0 { -x_norm } else { x_norm };
        let mut v = vec![0.0; m_len];
        if let Some(first) = v.first_mut() {
            *first = x0 - alpha;
        }
        for i in (k + 2)..n {
            if let Some(slot) = v.get_mut(i - (k + 1)) {
                *slot = h.get(i, k)?;
            }
        }
        let mut v_norm: f64 = 0.0;
        for &vi in &v {
            v_norm = v_norm.hypot(vi);
        }
        if v_norm < 1e-15 {
            continue;
        }
        for vi in &mut v {
            *vi /= v_norm;
        }

        // Left multiply H(k+1..n, k..n)
        for j in k..n {
            let mut dot = 0.0;
            for i in (k + 1)..n {
                dot += v.get(i - (k + 1)).copied().unwrap_or(0.0) * h.get(i, j)?;
            }
            for i in (k + 1)..n {
                let val =
                    h.get(i, j)? - 2.0 * v.get(i - (k + 1)).copied().unwrap_or(0.0) * dot;
                h.set(i, j, val)?;
            }
        }
        // Right multiply H(0..n, k+1..n)
        for i in 0..n {
            let mut dot = 0.0;
            for j in (k + 1)..n {
                dot += h.get(i, j)? * v.get(j - (k + 1)).copied().unwrap_or(0.0);
            }
            for j in (k + 1)..n {
                let val =
                    h.get(i, j)? - 2.0 * dot * v.get(j - (k + 1)).copied().unwrap_or(0.0);
                h.set(i, j, val)?;
            }
        }
    }

    // 2. Shifted QR iteration with Wilkinson shift
    let eye = Tensor::eye(n, budget)?;
    for _iter in 0..120 {
        let mut sub_norm = 0.0;
        for i in 0..(n - 1) {
            if h.get(i + 1, i)?.abs() <= 1e-13 * (h.get(i, i)?.abs() + h.get(i + 1, i + 1)?.abs())
            {
                h.set(i + 1, i, 0.0)?;
            } else {
                sub_norm += h.get(i + 1, i)?.abs();
            }
        }
        if sub_norm < 1e-12 {
            break;
        }
        let mu = h.get(n - 1, n - 1)?;
        let shift = eye.map_unary(SiUnit::DIMENSIONLESS, |x| x * mu)?;
        let shifted = h.checked_sub(&shift)?;
        let (q, r) = qr(&shifted, budget)?;
        let rq = matmul(&r, &q, budget)?;
        h = rq.checked_add(&shift)?;
    }

    // 3. Extract 1x1 real and 2x2 complex conjugate eigenvalues along the quasi-triangular diagonal
    let mut eigs = Vec::with_capacity(n);
    let mut idx = 0;
    while idx < n {
        if idx + 1 < n && h.get(idx + 1, idx)?.abs() > 1e-10 {
            let a11 = h.get(idx, idx)?;
            let a12 = h.get(idx, idx + 1)?;
            let a21 = h.get(idx + 1, idx)?;
            let a22 = h.get(idx + 1, idx + 1)?;
            let tr = a11 + a22;
            let det_2x2 = a11 * a22 - a12 * a21;
            let disc = tr * tr - 4.0 * det_2x2;
            if disc >= 0.0 {
                let s = disc.sqrt();
                eigs.push(Complex64::new(0.5 * (tr + s), 0.0));
                eigs.push(Complex64::new(0.5 * (tr - s), 0.0));
            } else {
                let s = (-disc).sqrt();
                eigs.push(Complex64::new(0.5 * tr, 0.5 * s));
                eigs.push(Complex64::new(0.5 * tr, -0.5 * s));
            }
            idx += 2;
        } else {
            eigs.push(Complex64::new(h.get(idx, idx)?, 0.0));
            idx += 1;
        }
    }
    Ok(eigs)
}

#[allow(clippy::type_complexity)]
pub fn svd(
    a: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<(Tensor<f64>, Tensor<f64>, Tensor<f64>), LinalgError> {
    let m = a.rows();
    let n = a.cols();
    let mut u_work = a.clone();
    let mut v = Tensor::eye(n, budget)?;

    for _sweep in 0..80 {
        let mut converged = true;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for r in 0..m {
                    let ui = u_work.get(r, i)?;
                    let uj = u_work.get(r, j)?;
                    alpha += ui * ui;
                    beta += uj * uj;
                    gamma += ui * uj;
                }
                if gamma.abs() <= 1e-14 * (alpha * beta).sqrt() {
                    continue;
                }
                converged = false;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for r in 0..m {
                    let ui = u_work.get(r, i)?;
                    let uj = u_work.get(r, j)?;
                    u_work.set(r, i, c * ui - s * uj)?;
                    u_work.set(r, j, s * ui + c * uj)?;
                }
                for r in 0..n {
                    let vi = v.get(r, i)?;
                    let vj = v.get(r, j)?;
                    v.set(r, i, c * vi - s * vj)?;
                    v.set(r, j, s * vi + c * vj)?;
                }
            }
        }
        if converged {
            break;
        }
    }

    let mut sing_vals = vec![0.0; n];
    for j in 0..n {
        let mut norm: f64 = 0.0;
        for r in 0..m {
            norm = norm.hypot(u_work.get(r, j)?);
        }
        if let Some(slot) = sing_vals.get_mut(j) {
            *slot = norm;
        }
        if norm > 1e-15 {
            for r in 0..m {
                u_work.set(r, j, u_work.get(r, j)? / norm)?;
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a_idx, &b_idx| {
        let sa = sing_vals.get(a_idx).copied().unwrap_or(0.0);
        let sb = sing_vals.get(b_idx).copied().unwrap_or(0.0);
        sb.total_cmp(&sa)
    });

    let mut u_sorted = Tensor::zeros(m, n, budget)?;
    let mut s_mat = Tensor::full_with_budget(n, n, 0.0, a.unit(), budget)?;
    let mut v_sorted = Tensor::zeros(n, n, budget)?;

    for (new_col, &old_col) in order.iter().enumerate() {
        s_mat.set(
            new_col,
            new_col,
            sing_vals.get(old_col).copied().unwrap_or(0.0),
        )?;
        for r in 0..m {
            u_sorted.set(r, new_col, u_work.get(r, old_col)?)?;
        }
        for r in 0..n {
            v_sorted.set(r, new_col, v.get(r, old_col)?)?;
        }
    }

    Ok((u_sorted, s_mat, v_sorted))
}

/// Moore-Penrose Pseudoinverse (`pinv(A) = V * S^+ * U^T`).
pub fn pinv(a: &Tensor<f64>, budget: &Arc<MemoryBudget>) -> Result<Tensor<f64>, LinalgError> {
    let (u, s, v) = svd(a, budget)?;
    let n = a.cols();
    let max_sv = if n > 0 { s.get(0, 0)? } else { 0.0 };
    let tol = 1e-12 * (a.rows().max(n) as f64) * max_sv;
    let inv_unit = SiUnit::DIMENSIONLESS.checked_div(a.unit())?;
    let mut s_plus = Tensor::full_with_budget(n, n, 0.0, inv_unit, budget)?;
    for i in 0..n {
        let sv = s.get(i, i)?;
        if sv > tol {
            s_plus.set(i, i, 1.0 / sv)?;
        }
    }
    let vs = matmul(&v, &s_plus, budget)?;
    let ut = u.transpose()?;
    matmul(&vs, &ut, budget)
}

/// 2-Norm Condition Number (`cond(A) = sigma_max / sigma_min`).
pub fn cond(a: &Tensor<f64>, budget: &Arc<MemoryBudget>) -> Result<f64, LinalgError> {
    let (_, s, _) = svd(a, budget)?;
    let n = a.cols();
    if n == 0 {
        return Ok(0.0);
    }
    let s_max = s.get(0, 0)?;
    let s_min = s.get(n - 1, n - 1)?;
    if s_min <= 1e-15 {
        Ok(f64::INFINITY)
    } else {
        Ok(s_max / s_min)
    }
}

/// Numerical Matrix Rank (`rank(A)`).
pub fn rank(a: &Tensor<f64>, budget: &Arc<MemoryBudget>) -> Result<usize, LinalgError> {
    let (_, s, _) = svd(a, budget)?;
    let n = a.cols();
    if n == 0 {
        return Ok(0);
    }
    let max_sv = s.get(0, 0)?;
    let tol = 1e-12 * (a.rows().max(n) as f64) * max_sv;
    let mut r = 0;
    for i in 0..n {
        if s.get(i, i)? > tol {
            r += 1;
        }
    }
    Ok(r)
}

/// Matrix Exponential (`expm(A)`) via Scaling & Squaring with Order-6 Padé Approximant!
/// Essential for exact Zero-Order Hold (`c2d`) state-space discretization $A_d = e^{A T_s}$.
pub fn expm(a: &Tensor<f64>, budget: &Arc<MemoryBudget>) -> Result<Tensor<f64>, LinalgError> {
    let n = a.rows();
    if n != a.cols() {
        return Err(LinalgError::NotSquare {
            op: "expm",
            shape: a.shape(),
        });
    }
    a.unit().verify_dimensionless("expm")?;
    let norm = a.norm_frobenius()?;
    let s = if norm > 0.5 {
        (norm / 0.5).log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let scale = 0.5_f64.powi(s as i32);
    let a_scaled = a.map_unary(SiUnit::DIMENSIONLESS, |x| x * scale)?;

    let eye = Tensor::eye(n, budget)?;
    let a2 = matmul(&a_scaled, &a_scaled, budget)?;
    let a4 = matmul(&a2, &a2, budget)?;
    let a6 = matmul(&a4, &a2, budget)?;

    // Padé(6,6) coefficients: c = [1, 1/2, 5/44, 1/66, 1/792, 1/15840, 1/665280]
    let c0 = 1.0;
    let c1 = 0.5;
    let c2 = 5.0 / 44.0;
    let c3 = 1.0 / 66.0;
    let c4 = 1.0 / 792.0;
    let c5 = 1.0 / 15_840.0;
    let c6 = 1.0 / 665_280.0;

    // Even part: V = c0*I + c2*A2 + c4*A4 + c6*A6
    let v_mat = eye
        .map_unary(SiUnit::DIMENSIONLESS, |x| x * c0)?
        .checked_add(&a2.map_unary(SiUnit::DIMENSIONLESS, |x| x * c2)?)?
        .checked_add(&a4.map_unary(SiUnit::DIMENSIONLESS, |x| x * c4)?)?
        .checked_add(&a6.map_unary(SiUnit::DIMENSIONLESS, |x| x * c6)?)?;

    // Odd inner: W = c1*I + c3*A2 + c5*A4, U = A * W
    let w_mat = eye
        .map_unary(SiUnit::DIMENSIONLESS, |x| x * c1)?
        .checked_add(&a2.map_unary(SiUnit::DIMENSIONLESS, |x| x * c3)?)?
        .checked_add(&a4.map_unary(SiUnit::DIMENSIONLESS, |x| x * c5)?)?;
    let u_mat = matmul(&a_scaled, &w_mat, budget)?;

    let num = v_mat.checked_add(&u_mat)?;
    let den = v_mat.checked_sub(&u_mat)?;
    let mut exp_mat = mldivide(&den, &num, budget)?;

    for _ in 0..s {
        exp_mat = matmul(&exp_mat, &exp_mat, budget)?;
    }
    Ok(exp_mat)
}

/// 1D Discrete Convolution (`conv(u, v)`).
#[must_use]
pub fn conv_1d(u: &[f64], v: &[f64]) -> Vec<f64> {
    if u.is_empty() || v.is_empty() {
        return Vec::new();
    }
    let out_len = u.len() + v.len() - 1;
    let mut out = vec![0.0; out_len];
    for (i, &ui) in u.iter().enumerate() {
        for (j, &vj) in v.iter().enumerate() {
            if let Some(slot) = out.get_mut(i + j) {
                *slot += ui * vj;
            }
        }
    }
    out
}

#[must_use]
pub fn fft_1d(signal: &[Complex64]) -> Vec<Complex64> {
    let n = signal.len();
    if n <= 1 {
        return signal.to_vec();
    }
    if n.is_power_of_two() {
        let mut even = Vec::with_capacity(n / 2);
        let mut odd = Vec::with_capacity(n / 2);
        for (idx, &val) in signal.iter().enumerate() {
            if idx % 2 == 0 {
                even.push(val);
            } else {
                odd.push(val);
            }
        }
        let even_fft = fft_1d(&even);
        let odd_fft = fft_1d(&odd);
        let mut out = vec![Complex64::ZERO; n];
        for k in 0..(n / 2) {
            let angle = -2.0 * PI * (k as f64) / (n as f64);
            let twiddle = Complex64::new(angle.cos(), angle.sin());
            let e = even_fft.get(k).copied().unwrap_or(Complex64::ZERO);
            let o = odd_fft.get(k).copied().unwrap_or(Complex64::ZERO);
            let t = twiddle * o;
            if let Some(slot) = out.get_mut(k) {
                *slot = e + t;
            }
            if let Some(slot) = out.get_mut(k + n / 2) {
                *slot = e - t;
            }
        }
        out
    } else {
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let mut sum = Complex64::ZERO;
            for (t, &x) in signal.iter().enumerate() {
                let angle = -2.0 * PI * ((k * t) as f64) / (n as f64);
                let w = Complex64::new(angle.cos(), angle.sin());
                sum = sum + x * w;
            }
            out.push(sum);
        }
        out
    }
}

/// Inverse 1D Discrete Fourier Transform (`ifft`).
#[must_use]
pub fn ifft_1d(spectrum: &[Complex64]) -> Vec<Complex64> {
    let n = spectrum.len();
    if n == 0 {
        return Vec::new();
    }
    let conj_in: Vec<Complex64> = spectrum.iter().map(|c| c.conj()).collect();
    let forward = fft_1d(&conj_in);
    let scale = 1.0 / (n as f64);
    forward
        .into_iter()
        .map(|c| Complex64::new(c.re * scale, -c.im * scale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expm_eig_general_and_pinv() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        // 1. expm([0, -pi; pi, 0]) = [-1, 0; 0, -1] (180-degree rotation!)
        let rot_gen = Tensor::from_column_major(
            2,
            2,
            vec![0.0, PI, -PI, 0.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        let e_rot = expm(&rot_gen, &budget)?;
        assert!((e_rot.get(0, 0)? - (-1.0)).abs() < 1e-10);
        assert!((e_rot.get(1, 1)? - (-1.0)).abs() < 1e-10);

        // 2. General eigenvalues of rotation generator [0, -2; 2, 0] are +-2i
        let gen2 = Tensor::from_column_major(
            2,
            2,
            vec![0.0, 2.0, -2.0, 0.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        let eigs = eig_general(&gen2, &budget)?;
        assert_eq!(eigs.len(), 2);
        assert!((eigs.first().copied().unwrap_or_default().im.abs() - 2.0).abs() < 1e-10);

        // 3. Pseudoinverse A * pinv(A) * A == A
        let rect = Tensor::from_column_major(
            3,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            SiUnit::DIMENSIONLESS,
            &budget,
        )?;
        let p = pinv(&rect, &budget)?;
        let ap = matmul(&rect, &p, &budget)?;
        let apa = matmul(&ap, &rect, &budget)?;
        assert!((apa.get(2, 1)? - 6.0).abs() < 1e-9);
        Ok(())
    }
}
