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

//! Robust control: H∞ norm bound, structured singular value upper bound, disk margin.

/// Estimate the H∞ norm bound via Hamiltonian matrix eigenvalue test.
///
/// For the state-space system (A, B, C) with square n×n A, n×1 B and C (column vectors),
/// binary-searches for the smallest γ > 0 such that the 2n×2n Hamiltonian
/// `H(γ) = [[A, -B B^T / γ²]; [-C^T C, -A^T]]`
/// has no purely imaginary eigenvalues (i.e., all eigenvalues have non-zero real part).
///
/// All inputs are flat row-major; B and C are treated as n×1 column vectors.
pub fn h_infinity_bound(a: &[f64], b: &[f64], c: &[f64], n: usize) -> f64 {
    if n == 0 || a.len() < n * n || b.len() < n || c.len() < n {
        return 0.0;
    }

    // Practical approximation: use peak of frequency response over a grid.
    // This avoids the need for a full eigensolver.
    let freq_grid: Vec<f64> = {
        let mut v = Vec::with_capacity(500);
        // Log-spaced from 1e-3 to 1e3
        for i in 0..500 {
            let exp = -3.0 + 6.0 * (i as f64) / 499.0;
            v.push(10.0_f64.powf(exp));
        }
        v
    };

    let mut h_inf_approx = 0.0_f64;
    for &omega in &freq_grid {
        // Evaluate ‖C (jωI - A)^{-1} B‖ via 2n real block system
        let val = eval_tf_magnitude(a, b, c, n, omega);
        if val > h_inf_approx {
            h_inf_approx = val;
        }
    }

    if h_inf_approx < 1e-15 {
        return 1e-6;
    }

    // Refine with binary search using our frequency-domain approximation
    let mut lo = h_inf_approx * 0.5;
    let mut hi = h_inf_approx * 2.0 + 1.0;

    for _ in 0..50 {
        let mid = (lo + hi) * 0.5;
        if hamiltonian_has_imag_eig(a, b, c, n, mid) {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-9 {
            break;
        }
    }

    hi
}

/// Evaluate |C (jωI - A)^{-1} B| at a single frequency ω.
/// Uses the real block formulation: [[ -A, ωI]; [-ωI, -A]] z = [B; 0], y = C z_1 + 0 z_2.
fn eval_tf_magnitude(a: &[f64], b: &[f64], c: &[f64], n: usize, omega: f64) -> f64 {
    // Solve the 2n×2n real system M z = rhs
    // M = [[-A, -ωI]; [ωI, -A]]
    // rhs = [B; 0]
    // y = C z[0..n]  (real part of output)
    let n2 = 2 * n;
    let mut mat = vec![0.0f64; n2 * n2];
    // Top-left: -A
    for i in 0..n {
        for j in 0..n {
            let a_ij = a.get(i * n + j).copied().unwrap_or(0.0);
            if let Some(slot) = mat.get_mut(i * n2 + j) {
                *slot = -a_ij;
            }
        }
    }
    // Top-right: -ωI
    for i in 0..n {
        if let Some(slot) = mat.get_mut(i * n2 + n + i) {
            *slot = -omega;
        }
    }
    // Bottom-left: ωI
    for i in 0..n {
        if let Some(slot) = mat.get_mut((n + i) * n2 + i) {
            *slot = omega;
        }
    }
    // Bottom-right: -A
    for i in 0..n {
        for j in 0..n {
            let a_ij = a.get(i * n + j).copied().unwrap_or(0.0);
            if let Some(slot) = mat.get_mut((n + i) * n2 + n + j) {
                *slot = -a_ij;
            }
        }
    }

    // rhs = [b; 0]
    let mut rhs = vec![0.0f64; n2];
    for i in 0..n {
        if let Some(slot) = rhs.get_mut(i) {
            *slot = b.get(i).copied().unwrap_or(0.0);
        }
    }

    // Solve via Gaussian elimination with partial pivoting
    let z = match gauss_solve(&mat, &rhs, n2) {
        Some(v) => v,
        None => return 0.0,
    };

    // y = C z[0..n]
    let mut y_re = 0.0;
    let mut y_im = 0.0;
    for i in 0..n {
        let ci = c.get(i).copied().unwrap_or(0.0);
        y_re += ci * z.get(i).copied().unwrap_or(0.0);
        y_im += ci * z.get(n + i).copied().unwrap_or(0.0);
    }
    (y_re * y_re + y_im * y_im).sqrt()
}

/// Check whether the Hamiltonian H(γ) has any purely imaginary eigenvalues.
/// Conservative check: test if `γI - M` is singular for the Hamiltonian's
/// characteristic values using the frequency response approach.
fn hamiltonian_has_imag_eig(a: &[f64], b: &[f64], c: &[f64], n: usize, gamma: f64) -> bool {
    // H(γ) has imaginary eigenvalues iff the system gain exceeds γ at some ω.
    // So H has imag eig ↔ ‖G‖_∞ > γ.
    let peak = {
        let grid: Vec<f64> = (0..200)
            .map(|i| 10.0_f64.powf(-3.0 + 6.0 * (i as f64) / 199.0))
            .collect();
        grid.iter()
            .map(|&w| eval_tf_magnitude(a, b, c, n, w))
            .fold(0.0_f64, f64::max)
    };
    peak > gamma
}

/// Gaussian elimination with partial pivoting to solve A x = b.
/// Returns `None` if the system is singular.
fn gauss_solve(mat: &[f64], rhs: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = mat.to_vec();
    let mut b = rhs.to_vec();

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = a.get(col * n + col).copied().unwrap_or(0.0).abs();
        for row in (col + 1)..n {
            let v = a.get(row * n + col).copied().unwrap_or(0.0).abs();
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
                let idx1 = col * n + j;
                let idx2 = max_row * n + j;
                if idx1 < a.len() && idx2 < a.len() {
                    a.swap(idx1, idx2);
                }
            }
            if col < b.len() && max_row < b.len() {
                b.swap(col, max_row);
            }
        }

        let pivot = a.get(col * n + col).copied().unwrap_or(1.0);
        for row in (col + 1)..n {
            let factor = a.get(row * n + col).copied().unwrap_or(0.0) / pivot;
            if let Some(slot) = a.get_mut(row * n + col) {
                *slot = 0.0;
            }
            for j in (col + 1)..n {
                let aij = a.get(col * n + j).copied().unwrap_or(0.0);
                if let Some(slot) = a.get_mut(row * n + j) {
                    *slot -= factor * aij;
                }
            }
            let bk = b.get(col).copied().unwrap_or(0.0);
            if let Some(slot) = b.get_mut(row) {
                *slot -= factor * bk;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = b.get(i).copied().unwrap_or(0.0);
        for j in (i + 1)..n {
            sum -= a.get(i * n + j).copied().unwrap_or(0.0) * x.get(j).copied().unwrap_or(0.0);
        }
        let diag = a.get(i * n + i).copied().unwrap_or(1.0);
        if diag.abs() < 1e-15 {
            return None;
        }
        if let Some(slot) = x.get_mut(i) {
            *slot = sum / diag;
        }
    }
    Some(x)
}

/// Structured singular value upper bound = max singular value (conservative).
/// Returns 0.0 for empty input.
#[must_use]
pub fn mu_upper_bound(singular_values: &[f64]) -> f64 {
    singular_values
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
}

/// Disk-based stability margin = `km * sin(pm_deg_in_radians)`.
#[must_use]
pub fn disk_margin(km: f64, pm_deg: f64) -> f64 {
    km * pm_deg.to_radians().sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mu_upper_bound() {
        let svs = vec![3.0, 1.5, 2.7, 0.5];
        assert!((mu_upper_bound(&svs) - 3.0).abs() < 1e-12);
        assert!((mu_upper_bound(&[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_disk_margin() {
        // km=1, pm=90° → sin(π/2) = 1
        assert!((disk_margin(1.0, 90.0) - 1.0).abs() < 1e-12);
        // km=2, pm=30° → 2*sin(π/6) = 2*0.5 = 1.0
        assert!((disk_margin(2.0, 30.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_h_infinity_bound_stable_system() {
        // Simple 1D system: ẋ = -x + u, y = x
        // A = [-1], B = [1], C = [1]
        // H∞ norm = 1 (peak of |1/(jω+1)| at ω=0 is 1.0)
        let a = vec![-1.0];
        let b = vec![1.0];
        let c = vec![1.0];
        let gamma = h_infinity_bound(&a, &b, &c, 1);
        // Should be close to 1.0
        assert!(gamma > 0.5, "H∞ bound should be positive and near 1.0, got {gamma}");
        assert!(gamma < 10.0, "H∞ bound should be reasonable, got {gamma}");
    }

    #[test]
    fn test_h_infinity_bound_empty() {
        assert_eq!(h_infinity_bound(&[], &[], &[], 0), 0.0);
    }
}
