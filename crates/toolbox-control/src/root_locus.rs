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

//! Evans Root Locus (`rlocus`) and Pole-Zero Map (`pzmap`).
//! Traces closed-loop pole migrations as feedback gain sweeps from 0 to infinity.

use crate::TransferFunction;
use linalg_cpu::{eig_general, LinalgError};
use num_core::{Complex64, MemoryBudget, SiUnit};
use std::f64::consts::PI;
use std::sync::Arc;
use tensor::Tensor;

/// Open-loop pole and zero constellation.
#[derive(Clone, Debug, PartialEq)]
pub struct PoleZeroMap {
    /// Open-loop poles (roots of denominator).
    pub poles: Vec<Complex64>,
    /// Open-loop zeros (roots of numerator).
    pub zeros: Vec<Complex64>,
}

/// A single sample point along the root locus branches.
#[derive(Clone, Debug, PartialEq)]
pub struct RootLocusBranchPoint {
    /// Feedback gain k >= 0.
    pub gain: f64,
    /// Closed-loop poles for this gain: roots of D(s) + k * N(s) = 0.
    pub poles: Vec<Complex64>,
}

/// Evans Root Locus computation result.
#[derive(Clone, Debug, PartialEq)]
pub struct RootLocusResult {
    pub open_loop_poles: Vec<Complex64>,
    pub open_loop_zeros: Vec<Complex64>,
    pub points: Vec<RootLocusBranchPoint>,
    /// Asymptote angles in radians: (2*m + 1)*pi / (n_poles - n_zeros)
    pub asymptote_angles_rad: Vec<f64>,
    /// Centroid (center of mass) of the asymptotes on the real axis: (sum(poles) - sum(zeros)) / (n_poles - n_zeros)
    pub centroid_real: f64,
}

/// Computes the roots of a polynomial `c[0]*s^n + c[1]*s^(n-1) + ... + c[n] = 0`
/// using the eigenvalues of its Frobenius companion matrix.
pub fn poly_roots(coeffs: &[f64], budget: &Arc<MemoryBudget>) -> Result<Vec<Complex64>, LinalgError> {
    // Trim leading zeros
    let mut start = 0;
    while start < coeffs.len() && coeffs.get(start).copied().unwrap_or(0.0).abs() < 1e-15 {
        start = start.saturating_add(1);
    }
    let valid_coeffs = match coeffs.get(start..) {
        Some(s) if s.len() > 1 => s,
        _ => return Ok(Vec::new()),
    };

    let degree = valid_coeffs.len().saturating_sub(1);
    if degree == 0 {
        return Ok(Vec::new());
    }

    let a0 = valid_coeffs.first().copied().unwrap_or(1.0);
    if a0.abs() < 1e-15 {
        return Ok(Vec::new());
    }

    if degree == 1 {
        let a1 = valid_coeffs.get(1).copied().unwrap_or(0.0);
        return Ok(vec![Complex64::new(-a1 / a0, 0.0)]);
    }

    // Degree >= 2: construct degree × degree companion matrix
    // Row 0 has normalized coefficients: -c_1, -c_2, ..., -c_n
    // Subdiagonal has 1s
    let mut comp = vec![0.0f64; degree.saturating_mul(degree)];
    for j in 0..degree {
        let c_j = valid_coeffs.get(j.saturating_add(1)).copied().unwrap_or(0.0) / a0;
        // comp[0, j] in column-major: j * degree + 0
        if let Some(slot) = comp.get_mut(j.saturating_mul(degree)) {
            *slot = -c_j;
        }
        if j.saturating_add(1) < degree {
            // comp[j+1, j] in column-major: j * degree + (j+1)
            if let Some(slot) = comp.get_mut(j.saturating_mul(degree).saturating_add(j.saturating_add(1))) {
                *slot = 1.0;
            }
        }
    }

    let tensor = Tensor::from_column_major(degree, degree, comp, SiUnit::DIMENSIONLESS, budget)?;
    eig_general(&tensor, budget)
}

/// Compute open-loop poles and zeros for a transfer function (`pzmap`).
pub fn pzmap(tf: &TransferFunction, budget: &Arc<MemoryBudget>) -> Result<PoleZeroMap, LinalgError> {
    let poles = poly_roots(&tf.den, budget)?;
    let zeros = poly_roots(&tf.num, budget)?;
    Ok(PoleZeroMap { poles, zeros })
}

/// Computes the Evans root locus for transfer function `tf` across the given `gains`.
pub fn rlocus(
    tf: &TransferFunction,
    gains: &[f64],
    budget: &Arc<MemoryBudget>,
) -> Result<RootLocusResult, LinalgError> {
    let pz = pzmap(tf, budget)?;
    let n_poles = pz.poles.len();
    let n_zeros = pz.zeros.len();

    // Asymptotes: for excess = n_poles - n_zeros > 0
    let excess = n_poles.saturating_sub(n_zeros);
    let mut asymptote_angles_rad = Vec::new();
    let mut centroid_real = 0.0;

    if excess > 0 {
        let sum_poles: f64 = pz.poles.iter().map(|p| p.re).sum();
        let sum_zeros: f64 = pz.zeros.iter().map(|z| z.re).sum();
        centroid_real = (sum_poles - sum_zeros) / (excess as f64);

        for m in 0..excess {
            let angle = (2.0 * (m as f64) + 1.0) * PI / (excess as f64);
            asymptote_angles_rad.push(angle);
        }
    }

    let mut points = Vec::with_capacity(gains.len());

    let den_deg = tf.den.len();
    let num_deg = tf.num.len();
    let max_deg = den_deg.max(num_deg);

    for &k in gains {
        if k < 0.0 {
            continue;
        }
        // Closed loop characteristic polynomial: D(s) + k * N(s)
        let mut char_poly = vec![0.0f64; max_deg];
        for (i, slot) in char_poly.iter_mut().enumerate() {
            // Denominator aligned to the right (lowest power at end)
            let den_idx = den_deg as isize - (max_deg - i) as isize;
            let d_val = if den_idx >= 0 && (den_idx as usize) < den_deg {
                tf.den.get(den_idx as usize).copied().unwrap_or(0.0)
            } else {
                0.0
            };

            let num_idx = num_deg as isize - (max_deg - i) as isize;
            let n_val = if num_idx >= 0 && (num_idx as usize) < num_deg {
                tf.num.get(num_idx as usize).copied().unwrap_or(0.0)
            } else {
                0.0
            };

            *slot = d_val + k * n_val;
        }

        let cl_poles = poly_roots(&char_poly, budget)?;
        points.push(RootLocusBranchPoint {
            gain: k,
            poles: cl_poles,
        });
    }

    Ok(RootLocusResult {
        open_loop_poles: pz.poles,
        open_loop_zeros: pz.zeros,
        points,
        asymptote_angles_rad,
        centroid_real,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_roots_quadratic() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        // s^2 + 5s + 6 = 0 => roots at -2, -3
        let roots = poly_roots(&[1.0, 5.0, 6.0], &budget)?;
        assert_eq!(roots.len(), 2);
        let mut re_roots: Vec<f64> = roots.iter().map(|r| r.re).collect();
        re_roots.sort_by(|a, b| a.total_cmp(b));
        let r1 = re_roots.first().copied().unwrap_or(0.0);
        let r2 = re_roots.get(1).copied().unwrap_or(0.0);
        assert!((r1 - (-3.0)).abs() < 1e-4, "r1={r1}");
        assert!((r2 - (-2.0)).abs() < 1e-4, "r2={r2}");
        Ok(())
    }

    #[test]
    fn test_pzmap_and_rlocus() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        // G(s) = (s + 2) / (s^2 + 2s + 5)
        // Zero at s = -2
        // Poles at s = -1 +- 2j
        let tf = TransferFunction::new(vec![1.0, 2.0], vec![1.0, 2.0, 5.0]);
        let pz = pzmap(&tf, &budget)?;
        assert_eq!(pz.zeros.len(), 1);
        assert_eq!(pz.poles.len(), 2);

        let z0 = pz.zeros.first().copied().unwrap_or(Complex64::ZERO);
        assert!((z0.re - (-2.0)).abs() < 1e-4);

        let gains = vec![0.0, 1.0, 10.0, 100.0];
        let locus = rlocus(&tf, &gains, &budget)?;
        assert_eq!(locus.points.len(), 4);
        assert_eq!(locus.asymptote_angles_rad.len(), 1); // excess = 2 - 1 = 1 asymptote at pi
        assert!((locus.asymptote_angles_rad.first().copied().unwrap_or(0.0) - PI).abs() < 1e-4);
        Ok(())
    }
}
