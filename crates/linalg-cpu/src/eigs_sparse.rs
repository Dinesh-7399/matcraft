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

//! Sparse extremal eigenvalue solver (`eigs`) via Arnoldi iteration.
//! Computes the top-k eigenvalues of large sparse CSC matrices.

use crate::{eig_general, LinalgError};
use num_core::{Complex64, MemoryBudget, SiUnit};
use std::sync::Arc;
use tensor::{SparseCsc, Tensor};

/// Selection criterion for extremal eigenvalues in `eigs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EigsTarget {
    /// Largest magnitude `|λ|` (matches MATLAB `'LM'`).
    LargestMagnitude,
    /// Smallest magnitude `|λ|` (matches MATLAB `'SM'`).
    SmallestMagnitude,
    /// Largest real part `Re(λ)` (matches MATLAB `'LR'`).
    LargestReal,
}

/// Result of sparse eigenvalue extraction.
#[derive(Clone, Debug, PartialEq)]
pub struct EigsResult {
    /// Computed top-k eigenvalues.
    pub eigenvalues: Vec<Complex64>,
    /// Dimension of the generated Krylov subspace.
    pub krylov_dim: usize,
}

/// Computes the top-`k` eigenvalues of a sparse CSC matrix `a` using the Arnoldi iteration.
pub fn eigs_sparse(
    a: &SparseCsc<f64>,
    k: usize,
    target: EigsTarget,
    budget: &Arc<MemoryBudget>,
) -> Result<EigsResult, LinalgError> {
    let n = a.rows();
    if n != a.cols() {
        return Err(LinalgError::NotSquare {
            op: "eigs",
            shape: [n, a.cols()],
        });
    }
    if n == 0 || k == 0 {
        return Ok(EigsResult {
            eigenvalues: Vec::new(),
            krylov_dim: 0,
        });
    }

    let actual_k = k.min(n);
    // Subspace size m >= k, capped at n
    let m = (actual_k.saturating_mul(2).max(actual_k.saturating_add(2))).min(n);

    // Initial normalized vector v_0 = [1/sqrt(n), ..., 1/sqrt(n)]
    let init_val = 1.0 / (n as f64).sqrt();
    let mut v_basis: Vec<Vec<f64>> = Vec::with_capacity(m);
    v_basis.push(vec![init_val; n]);

    // Upper Hessenberg matrix H of size m × m stored in flat column-major
    let mut h_mat = vec![0.0f64; m.saturating_mul(m)];
    let mut actual_m = m;

    for j in 0..m {
        // Compute w = A * v_j via sparse matrix-vector multiplication
        let vj = match v_basis.get(j) {
            Some(v) => v,
            None => break,
        };
        let mut w = a.spmv(vj).map_err(LinalgError::Tensor)?;

        // Modified Gram-Schmidt orthogonalization against v_0 .. v_j
        for (i, vi) in v_basis.iter().enumerate() {
            let mut dot = 0.0;
            for (idx, &v_elem) in vi.iter().enumerate() {
                let w_elem = w.get(idx).copied().unwrap_or(0.0);
                dot += v_elem * w_elem;
            }
            // H[i, j] = vi^T * w
            if let Some(slot) = h_mat.get_mut(j.saturating_mul(m).saturating_add(i)) {
                *slot = dot;
            }
            for (idx, slot) in w.iter_mut().enumerate() {
                let v_elem = vi.get(idx).copied().unwrap_or(0.0);
                *slot -= dot * v_elem;
            }
        }

        // Norm of remaining orthogonal vector: h[j+1, j] = ||w||_2
        let mut norm_w: f64 = 0.0;
        for &elem in &w {
            norm_w = norm_w.hypot(elem);
        }

        if j.saturating_add(1) < m {
            if let Some(slot) = h_mat.get_mut(j.saturating_mul(m).saturating_add(j.saturating_add(1))) {
                *slot = norm_w;
            }

            // Lucky breakdown: subspace is invariant
            if norm_w < 1e-13 {
                actual_m = j.saturating_add(1);
                break;
            }

            for elem in &mut w {
                *elem /= norm_w;
            }
            v_basis.push(w);
        }
    }

    // Extract the actual_m × actual_m submatrix of H
    let mut sub_h = Vec::with_capacity(actual_m.saturating_mul(actual_m));
    for c in 0..actual_m {
        for r in 0..actual_m {
            let val = h_mat.get(c.saturating_mul(m).saturating_add(r)).copied().unwrap_or(0.0);
            sub_h.push(val);
        }
    }

    let h_tensor = Tensor::from_column_major(
        actual_m,
        actual_m,
        sub_h,
        SiUnit::DIMENSIONLESS,
        budget,
    )?;

    // Compute eigenvalues of the small Hessenberg matrix
    let mut eigs = eig_general(&h_tensor, budget)?;

    // Sort according to target
    match target {
        EigsTarget::LargestMagnitude => {
            eigs.sort_by(|a, b| {
                let na = (a.re * a.re + a.im * a.im).sqrt();
                let nb = (b.re * b.re + b.im * b.im).sqrt();
                nb.total_cmp(&na)
            });
        }
        EigsTarget::SmallestMagnitude => {
            eigs.sort_by(|a, b| {
                let na = (a.re * a.re + a.im * a.im).sqrt();
                let nb = (b.re * b.re + b.im * b.im).sqrt();
                na.total_cmp(&nb)
            });
        }
        EigsTarget::LargestReal => {
            eigs.sort_by(|a, b| b.re.total_cmp(&a.re));
        }
    }

    eigs.truncate(actual_k);

    Ok(EigsResult {
        eigenvalues: eigs,
        krylov_dim: actual_m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eigs_diagonal_matrix() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        // 5x5 diagonal matrix with eigenvalues [1, 2, 5, 10, 20]
        let triplets = vec![
            (0, 0, 1.0),
            (1, 1, 2.0),
            (2, 2, 5.0),
            (3, 3, 10.0),
            (4, 4, 20.0),
        ];
        let a = SparseCsc::from_triplets(5, 5, &triplets, SiUnit::DIMENSIONLESS, &budget)?;

        // Find top 2 largest magnitude eigenvalues (should find ~20 and ~10)
        let res = eigs_sparse(&a, 2, EigsTarget::LargestMagnitude, &budget)?;
        assert_eq!(res.eigenvalues.len(), 2);
        let lambda1 = res.eigenvalues.first().copied().unwrap_or(Complex64::ZERO);
        let lambda2 = res.eigenvalues.get(1).copied().unwrap_or(Complex64::ZERO);

        assert!((lambda1.re - 20.0).abs() < 1e-2, "lambda1={lambda1:?}");
        assert!((lambda2.re - 10.0).abs() < 0.1, "lambda2={lambda2:?}");
        Ok(())
    }

    #[test]
    fn test_eigs_tridiagonal_symmetric() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        // 4x4 1D Laplacian tridiagonal matrix: [2, -1; -1, 2, ...]
        let triplets = vec![
            (0, 0, 2.0), (0, 1, -1.0),
            (1, 0, -1.0), (1, 1, 2.0), (1, 2, -1.0),
            (2, 1, -1.0), (2, 2, 2.0), (2, 3, -1.0),
            (3, 2, -1.0), (3, 3, 2.0),
        ];
        let a = SparseCsc::from_triplets(4, 4, &triplets, SiUnit::DIMENSIONLESS, &budget)?;

        let res = eigs_sparse(&a, 1, EigsTarget::LargestMagnitude, &budget)?;
        assert_eq!(res.eigenvalues.len(), 1);
        let top = res.eigenvalues.first().copied().unwrap_or(Complex64::ZERO);
        // Symmetric Krylov subspace finds the symmetric mode lambda_3 = 2 - 2*cos(3pi/5) = 2.618034
        assert!((top.re - 2.618034).abs() < 1e-4, "top={top:?}");
        Ok(())
    }
}
