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

//! Advanced matrix decompositions: bidiagonalization, Hessenberg reduction, Gram-Schmidt QR.

use crate::LinalgError;
use num_core::{MemoryBudget, SiUnit};
use std::sync::Arc;
use tensor::Tensor;

// ─────────────────────────── helpers ───────────────────────────

/// Create an n×n identity matrix.
fn eye_n(n: usize, budget: &Arc<MemoryBudget>) -> Result<Tensor<f64>, LinalgError> {
    Ok(Tensor::eye(n, budget)?)
}

/// Apply a Householder reflector H = I - 2*v*v^T to a matrix column-slice in place.
/// Updates `mat` columns `[col_start..ncols]` for rows `[row_start..nrows]`.
fn householder_left(
    mat: &mut Tensor<f64>,
    v: &[f64],
    row_start: usize,
    col_start: usize,
    budget: &Arc<MemoryBudget>,
) -> Result<(), LinalgError> {
    let nrows = mat.rows();
    let ncols = mat.cols();
    let vlen = v.len();

    // Compute w = 2 * mat[row_start:, col_start:]^T * v  (column vector of length ncols-col_start)
    let mut w = vec![0.0f64; ncols.saturating_sub(col_start)];
    for c in col_start..ncols {
        let mut dot = 0.0;
        for (vi, vval) in v.iter().enumerate().take(vlen) {
            let r = row_start.saturating_add(vi);
            if r < nrows {
                dot += mat.get(r, c).unwrap_or(0.0) * vval;
            }
        }
        if let Some(slot) = w.get_mut(c.saturating_sub(col_start)) {
            *slot = 2.0 * dot;
        }
    }

    // Apply: mat[row_start:, col_start:] -= v * w^T
    for (vi, vval) in v.iter().enumerate().take(vlen) {
        let r = row_start.saturating_add(vi);
        if r >= nrows {
            break;
        }
        for c in col_start..ncols {
            let wval = w.get(c.saturating_sub(col_start)).copied().unwrap_or(0.0);
            let cur = mat.get(r, c).unwrap_or(0.0);
            let _ = mat.set(r, c, cur - vval * wval);
        }
    }
    let _ = budget; // budget used by callers
    Ok(())
}

/// Apply a Householder reflector H = I - 2*v*v^T on the right: mat = mat * H.
fn householder_right(
    mat: &mut Tensor<f64>,
    v: &[f64],
    row_start: usize,
    col_start: usize,
    budget: &Arc<MemoryBudget>,
) -> Result<(), LinalgError> {
    let nrows = mat.rows();
    let ncols = mat.cols();
    let vlen = v.len();

    // Compute w = 2 * mat[:, col_start:] * v  (column vector of length nrows)
    let mut w = vec![0.0f64; nrows.saturating_sub(row_start)];
    for (vi, vval) in v.iter().enumerate().take(vlen) {
        let c = col_start.saturating_add(vi);
        if c >= ncols {
            break;
        }
        for r in row_start..nrows {
            let cur = mat.get(r, c).unwrap_or(0.0);
            if let Some(slot) = w.get_mut(r.saturating_sub(row_start)) {
                *slot += 2.0 * cur * vval;
            }
        }
    }

    // Apply: mat[:, col_start:] -= w * v^T
    for r in row_start..nrows {
        let wval = w.get(r.saturating_sub(row_start)).copied().unwrap_or(0.0);
        for (vi, vval) in v.iter().enumerate().take(vlen) {
            let c = col_start.saturating_add(vi);
            if c >= ncols {
                break;
            }
            let cur = mat.get(r, c).unwrap_or(0.0);
            let _ = mat.set(r, c, cur - wval * vval);
        }
    }
    let _ = budget;
    Ok(())
}

/// Compute the Householder vector `v` for column `col` of `mat` starting at row `row_start`.
/// Returns (v, norm) where H = I - 2*v*v^T zeroes out elements below the diagonal.
fn compute_householder_vec_col(
    mat: &Tensor<f64>,
    row_start: usize,
    col: usize,
) -> Result<Vec<f64>, LinalgError> {
    let nrows = mat.rows();
    let mut x: Vec<f64> = (row_start..nrows)
        .map(|r| mat.get(r, col).unwrap_or(0.0))
        .collect();
    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return Ok(x);
    }
    let sign = if x.first().copied().unwrap_or(0.0) >= 0.0 { 1.0 } else { -1.0 };
    if let Some(first) = x.first_mut() {
        *first += sign * norm;
    }
    let new_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if new_norm > 1e-15 {
        for val in &mut x {
            *val /= new_norm;
        }
    }
    Ok(x)
}

/// Compute Householder vector for a row of `mat` from column `col_start` onward at given row.
fn compute_householder_vec_row(
    mat: &Tensor<f64>,
    row: usize,
    col_start: usize,
) -> Result<Vec<f64>, LinalgError> {
    let ncols = mat.cols();
    let mut x: Vec<f64> = (col_start..ncols)
        .map(|c| mat.get(row, c).unwrap_or(0.0))
        .collect();
    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return Ok(x);
    }
    let sign = if x.first().copied().unwrap_or(0.0) >= 0.0 { 1.0 } else { -1.0 };
    if let Some(first) = x.first_mut() {
        *first += sign * norm;
    }
    let new_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if new_norm > 1e-15 {
        for val in &mut x {
            *val /= new_norm;
        }
    }
    Ok(x)
}

// ─────────────────────────── bidiag ───────────────────────────

/// Upper bidiagonalization of `a` via two-sided Householder reflectors.
/// Returns `(U, B, V)` where `B` is bidiagonal, `U^T A V = B`.
#[allow(clippy::type_complexity)]
pub fn bidiag(
    a: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<(Tensor<f64>, Tensor<f64>, Tensor<f64>), LinalgError> {
    let m = a.rows();
    let n = a.cols();

    if m == 0 || n == 0 {
        let u = eye_n(m.max(1), budget)?;
        let b = a.clone();
        let v = eye_n(n.max(1), budget)?;
        return Ok((u, b, v));
    }

    let mut b = a.clone();
    let mut u = eye_n(m, budget)?;
    let mut v = eye_n(n, budget)?;

    let steps = m.min(n);
    for k in 0..steps {
        // Left Householder: zero below diagonal in column k
        let v_left = compute_householder_vec_col(&b, k, k)?;
        householder_left(&mut b, &v_left, k, k, budget)?;
        // Accumulate U: apply from the right (U = U * H_left^T = U * H_left since H is symmetric)
        apply_householder_to_u(&mut u, &v_left, k, budget)?;

        // Right Householder: zero to the right of superdiagonal in row k (if columns remain)
        if k + 2 <= n {
            let v_right = compute_householder_vec_row(&b, k, k + 1)?;
            householder_right(&mut b, &v_right, 0, k + 1, budget)?;
            // Accumulate V
            apply_householder_to_v(&mut v, &v_right, k + 1, budget)?;
        }
    }
    Ok((u, b, v))
}

fn apply_householder_to_u(
    u: &mut Tensor<f64>,
    v: &[f64],
    col_start: usize,
    budget: &Arc<MemoryBudget>,
) -> Result<(), LinalgError> {
    // u = u * H  where H = I - 2*v*v^T (applied to rows col_start..m of U columns)
    // Equivalent to: for each row of u, subtract 2*(row[col_start:] . v) * v
    let m = u.rows();
    let vlen = v.len();
    for r in 0..m {
        let mut dot = 0.0;
        for (vi, vval) in v.iter().enumerate().take(vlen) {
            let c = col_start.saturating_add(vi);
            dot += u.get(r, c).unwrap_or(0.0) * vval;
        }
        dot *= 2.0;
        for (vi, vval) in v.iter().enumerate().take(vlen) {
            let c = col_start.saturating_add(vi);
            let cur = u.get(r, c).unwrap_or(0.0);
            let _ = u.set(r, c, cur - dot * vval);
        }
    }
    let _ = budget;
    Ok(())
}

fn apply_householder_to_v(
    v_mat: &mut Tensor<f64>,
    v: &[f64],
    col_start: usize,
    budget: &Arc<MemoryBudget>,
) -> Result<(), LinalgError> {
    let n = v_mat.rows();
    let vlen = v.len();
    for r in 0..n {
        let mut dot = 0.0;
        for (vi, vval) in v.iter().enumerate().take(vlen) {
            let c = col_start.saturating_add(vi);
            dot += v_mat.get(r, c).unwrap_or(0.0) * vval;
        }
        dot *= 2.0;
        for (vi, vval) in v.iter().enumerate().take(vlen) {
            let c = col_start.saturating_add(vi);
            let cur = v_mat.get(r, c).unwrap_or(0.0);
            let _ = v_mat.set(r, c, cur - dot * vval);
        }
    }
    let _ = budget;
    Ok(())
}

// ─────────────────────────── hessenberg ───────────────────────────

/// Hessenberg reduction of square matrix `a` via Householder reflectors.
/// Returns `(H, Q)` where `Q^T A Q = H` and `Q` is unitary.
pub fn hessenberg(
    a: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<(Tensor<f64>, Tensor<f64>), LinalgError> {
    let n = a.rows();
    if n != a.cols() {
        return Err(LinalgError::NotSquare {
            op: "hessenberg",
            shape: a.shape(),
        });
    }
    if n <= 2 {
        return Ok((a.clone(), eye_n(n.max(1), budget)?));
    }

    let mut h = a.clone();
    let mut q = eye_n(n, budget)?;

    for k in 0..(n - 2) {
        // Compute Householder vector for subdiagonal of column k
        let v = compute_householder_vec_col(&h, k + 1, k)?;

        // Apply from left: H = P * H  (rows k+1..n, all cols)
        householder_left_full(&mut h, &v, k + 1, 0, budget)?;
        // Apply from right: H = H * P  (all rows, cols k+1..n)
        householder_right_full(&mut h, &v, 0, k + 1, budget)?;
        // Accumulate Q: Q = Q * P
        apply_householder_to_u(&mut q, &v, k + 1, budget)?;
    }

    Ok((h, q))
}

fn householder_left_full(
    mat: &mut Tensor<f64>,
    v: &[f64],
    row_start: usize,
    _col_start: usize,
    budget: &Arc<MemoryBudget>,
) -> Result<(), LinalgError> {
    let ncols = mat.cols();
    householder_left(mat, v, row_start, 0, budget)?;
    let _ = ncols;
    Ok(())
}

fn householder_right_full(
    mat: &mut Tensor<f64>,
    v: &[f64],
    _row_start: usize,
    col_start: usize,
    budget: &Arc<MemoryBudget>,
) -> Result<(), LinalgError> {
    householder_right(mat, v, 0, col_start, budget)
}

// ─────────────────────────── gram_schmidt ───────────────────────────

/// Classical Gram-Schmidt QR orthonormalization.
/// Returns the orthonormal Q matrix (columns of `a` projected orthonormal).
pub fn gram_schmidt(
    a: &Tensor<f64>,
    budget: &Arc<MemoryBudget>,
) -> Result<Tensor<f64>, LinalgError> {
    let m = a.rows();
    let n = a.cols();
    if m == 0 || n == 0 {
        return Tensor::zeros(m, n, budget).map_err(LinalgError::from);
    }

    let mut q_cols: Vec<Vec<f64>> = Vec::with_capacity(n);

    for j in 0..n {
        // Start with column j of a
        let mut col: Vec<f64> = (0..m).map(|r| a.get(r, j).unwrap_or(0.0)).collect();

        // Subtract projections onto already-computed Q columns
        for q_col in &q_cols {
            let dot: f64 = col.iter().zip(q_col.iter()).map(|(&x, &y)| x * y).sum();
            for (c_val, q_val) in col.iter_mut().zip(q_col.iter()) {
                *c_val -= dot * q_val;
            }
        }

        // Normalize
        let norm: f64 = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for val in &mut col {
                *val /= norm;
            }
            q_cols.push(col);
        } else {
            // Linearly dependent column — insert zero column
            q_cols.push(vec![0.0; m]);
        }
    }

    // Assemble Q from column-major order
    let mut data = vec![0.0f64; m.saturating_mul(n)];
    for (j, col) in q_cols.iter().enumerate() {
        for (i, val) in col.iter().enumerate() {
            if let Some(slot) = data.get_mut(j.saturating_mul(m).saturating_add(i)) {
                *slot = *val;
            }
        }
    }

    Tensor::from_column_major(m, n, data, SiUnit::DIMENSIONLESS, budget)
        .map_err(LinalgError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul;
    use num_core::MemoryBudget;

    fn budget() -> Arc<MemoryBudget> {
        MemoryBudget::default_budget()
    }

    #[test]
    fn test_gram_schmidt_orthonormality() -> Result<(), Box<dyn std::error::Error>> {
        let b = budget();
        let a = Tensor::from_column_major(
            3, 3,
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            SiUnit::DIMENSIONLESS, &b,
        )?;
        let q = gram_schmidt(&a, &b)?;
        // Q^T Q should be identity
        let qt = q.transpose()?;
        let qtq = matmul(&qt, &q, &b)?;
        assert!((qtq.get(0, 0)? - 1.0).abs() < 1e-10);
        assert!((qtq.get(0, 1)?).abs() < 1e-10);
        assert!((qtq.get(1, 1)? - 1.0).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_hessenberg_reduces_square() -> Result<(), Box<dyn std::error::Error>> {
        let b = budget();
        let a = Tensor::from_column_major(
            3, 3,
            vec![4.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 2.0],
            SiUnit::DIMENSIONLESS, &b,
        )?;
        let (h, q) = hessenberg(&a, &b)?;
        assert_eq!(h.shape(), [3, 3]);
        assert_eq!(q.shape(), [3, 3]);
        // The (2,0) element of H should be ~0 (subsubdiagonal zeroed)
        assert!(h.get(2, 0)?.abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn test_bidiag_shape() -> Result<(), Box<dyn std::error::Error>> {
        let b = budget();
        let a = Tensor::from_column_major(
            3, 2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            SiUnit::DIMENSIONLESS, &b,
        )?;
        let (u, bmat, v) = bidiag(&a, &b)?;
        assert_eq!(u.shape(), [3, 3]);
        assert_eq!(bmat.shape(), [3, 2]);
        assert_eq!(v.shape(), [2, 2]);
        Ok(())
    }
}
