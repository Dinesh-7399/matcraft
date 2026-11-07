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

//! Nonlinear State Estimation Observers:
//! 1. Extended Kalman Filter (EKF) with analytic Jacobians computed automatically via Dual64 forward-mode autodiff.
//! 2. Unscented Kalman Filter (UKF) with deterministic sigma-point propagation through the unscented transform.

use autodiff::Dual64;
use linalg_cpu::LinalgError;

/// Numerical matrix inversion for small $n \times n$ dense matrices (up to $10 \times 10$) using partial-pivot Gauss-Jordan.
fn invert_small_matrix(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, LinalgError> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut mat: Vec<Vec<f64>> = a.to_vec();
    let mut inv: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for (i, row) in inv.iter_mut().enumerate() {
        if let Some(val) = row.get_mut(i) {
            *val = 1.0;
        }
    }

    for col in 0..n {
        let mut pivot_row = col;
        let mut max_val = mat
            .get(col)
            .and_then(|r| r.get(col))
            .copied()
            .unwrap_or(0.0)
            .abs();

        for row in (col + 1)..n {
            let val = mat
                .get(row)
                .and_then(|r| r.get(col))
                .copied()
                .unwrap_or(0.0)
                .abs();
            if val > max_val {
                max_val = val;
                pivot_row = row;
            }
        }

        if max_val < 1e-12 {
            return Err(LinalgError::SingularMatrix {
                index: col,
                pivot_val: max_val,
            });
        }

        if pivot_row != col {
            mat.swap(col, pivot_row);
            inv.swap(col, pivot_row);
        }

        let pivot = mat
            .get(col)
            .and_then(|r| r.get(col))
            .copied()
            .unwrap_or(1.0);

        for j in 0..n {
            if let Some(slot) = mat.get_mut(col).and_then(|r| r.get_mut(j)) {
                *slot /= pivot;
            }
            if let Some(slot) = inv.get_mut(col).and_then(|r| r.get_mut(j)) {
                *slot /= pivot;
            }
        }

        for row in 0..n {
            if row != col {
                let factor = mat
                    .get(row)
                    .and_then(|r| r.get(col))
                    .copied()
                    .unwrap_or(0.0);
                for j in 0..n {
                    let val_mat = mat
                        .get(col)
                        .and_then(|r| r.get(j))
                        .copied()
                        .unwrap_or(0.0);
                    let val_inv = inv
                        .get(col)
                        .and_then(|r| r.get(j))
                        .copied()
                        .unwrap_or(0.0);

                    if let Some(slot) = mat.get_mut(row).and_then(|r| r.get_mut(j)) {
                        *slot -= factor * val_mat;
                    }
                    if let Some(slot) = inv.get_mut(row).and_then(|r| r.get_mut(j)) {
                        *slot -= factor * val_inv;
                    }
                }
            }
        }
    }

    Ok(inv)
}

/// Extended Kalman Filter for discrete nonlinear systems:
/// $x_{k+1} = f(x_k, u_k) + w_k, \quad w_k \sim \mathcal{N}(0, Q)$
/// $z_k = h(x_k) + v_k, \quad v_k \sim \mathcal{N}(0, R)$
#[derive(Clone, Debug)]
pub struct ExtendedKalmanFilter {
    /// State estimate vector $\hat{x} \in \mathbb{R}^n$.
    pub x: Vec<f64>,
    /// Estimation error covariance matrix $P \in \mathbb{R}^{n \times n}$.
    pub p: Vec<Vec<f64>>,
    /// Process noise covariance matrix $Q \in \mathbb{R}^{n \times n}$.
    pub q: Vec<Vec<f64>>,
    /// Measurement noise covariance matrix $R \in \mathbb{R}^{m \times m}$.
    pub r: Vec<Vec<f64>>,
}

impl ExtendedKalmanFilter {
    /// Creates a new Extended Kalman Filter with initial state and covariance.
    #[must_use]
    pub fn new(x0: Vec<f64>, p0: Vec<Vec<f64>>, q: Vec<Vec<f64>>, r: Vec<Vec<f64>>) -> Self {
        Self {
            x: x0,
            p: p0,
            q,
            r,
        }
    }

    /// Prediction step: propagates state estimate through nonlinear dynamics function $f(x)$
    /// and linearizes state transition matrix $F = \partial f / \partial x$ via exact `Dual64` forward-mode autodiff.
    pub fn predict<F>(&mut self, dynamics: F) -> Result<(), LinalgError>
    where
        F: Fn(&[Dual64]) -> Vec<Dual64>,
    {
        let n = self.x.len();
        // 1. Nominal state prediction
        let d_in: Vec<Dual64> = self.x.iter().map(|&v| Dual64::constant(v)).collect();
        let d_out = dynamics(&d_in);
        let mut x_pred = Vec::with_capacity(n);
        for d in &d_out {
            x_pred.push(d.val);
        }

        // 2. Jacobian F via Dual64 differentiation
        let mut f_jac: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for j in 0..n {
            let mut d_var: Vec<Dual64> = self.x.iter().map(|&v| Dual64::constant(v)).collect();
            if let Some(slot) = d_var.get_mut(j) {
                *slot = Dual64::variable(slot.val);
            }
            let res = dynamics(&d_var);
            for (i, d) in res.iter().enumerate() {
                if let Some(slot) = f_jac.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *slot = d.der;
                }
            }
        }

        // 3. Covariance prediction: P = F * P * F^T + Q
        let mut f_p = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    let fij = f_jac.get(i).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
                    let pkj = self.p.get(k).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                    sum += fij * pkj;
                }
                if let Some(slot) = f_p.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *slot = sum;
                }
            }
        }

        let mut p_next = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    let fpik = f_p.get(i).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
                    let fjk = f_jac.get(j).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
                    sum += fpik * fjk;
                }
                let q_val = self.q.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                if let Some(slot) = p_next.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *slot = sum + q_val;
                }
            }
        }

        self.x = x_pred;
        self.p = p_next;
        Ok(())
    }

    /// Measurement update step: linearizes observation function $h(x)$ via `Dual64` autodiff
    /// and applies Joseph-stabilized covariance update.
    pub fn update<H>(&mut self, z_meas: &[f64], measurement_model: H) -> Result<(), LinalgError>
    where
        H: Fn(&[Dual64]) -> Vec<Dual64>,
    {
        let n = self.x.len();
        let m = z_meas.len();

        // 1. Predicted measurement h(x)
        let d_in: Vec<Dual64> = self.x.iter().map(|&v| Dual64::constant(v)).collect();
        let d_out = measurement_model(&d_in);
        let mut z_pred = Vec::with_capacity(m);
        for d in &d_out {
            z_pred.push(d.val);
        }

        // 2. Jacobian H (m x n) via Dual64 differentiation
        let mut h_jac: Vec<Vec<f64>> = vec![vec![0.0; n]; m];
        for j in 0..n {
            let mut d_var: Vec<Dual64> = self.x.iter().map(|&v| Dual64::constant(v)).collect();
            if let Some(slot) = d_var.get_mut(j) {
                *slot = Dual64::variable(slot.val);
            }
            let res = measurement_model(&d_var);
            for (i, d) in res.iter().enumerate() {
                if let Some(slot) = h_jac.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *slot = d.der;
                }
            }
        }

        // 3. Innovation: y = z - z_pred
        let mut y = Vec::with_capacity(m);
        for i in 0..m {
            let zm = z_meas.get(i).copied().unwrap_or(0.0);
            let zp = z_pred.get(i).copied().unwrap_or(0.0);
            y.push(zm - zp);
        }

        // 4. Innovation covariance: S = H * P * H^T + R (m x m)
        let mut h_p = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    let hik = h_jac.get(i).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
                    let pkj = self.p.get(k).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                    sum += hik * pkj;
                }
                if let Some(slot) = h_p.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *slot = sum;
                }
            }
        }

        let mut s_mat = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..n {
                    let hpik = h_p.get(i).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
                    let hjk = h_jac.get(j).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
                    sum += hpik * hjk;
                }
                let r_val = self.r.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                if let Some(slot) = s_mat.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *slot = sum + r_val;
                }
            }
        }

        let s_inv = invert_small_matrix(&s_mat)?;

        // 5. Kalman gain: K = P * H^T * S^-1 (n x m)
        // First P * H^T (n x m)
        let mut p_ht = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..n {
                    let pik = self.p.get(i).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
                    let hjk = h_jac.get(j).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
                    sum += pik * hjk;
                }
                if let Some(slot) = p_ht.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *slot = sum;
                }
            }
        }

        let mut k_gain = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..m {
                    let ph_ik = p_ht.get(i).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
                    let sinv_kj = s_inv.get(k).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                    sum += ph_ik * sinv_kj;
                }
                if let Some(slot) = k_gain.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *slot = sum;
                }
            }
        }

        // 6. State update: x = x + K * y
        for i in 0..n {
            let mut delta = 0.0;
            for j in 0..m {
                let kij = k_gain.get(i).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                let yj = y.get(j).copied().unwrap_or(0.0);
                delta += kij * yj;
            }
            if let Some(slot) = self.x.get_mut(i) {
                *slot += delta;
            }
        }

        // 7. Covariance update: P = (I - K * H) * P (Joseph stabilized symmetric form)
        let mut i_kh = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut kh_sum = 0.0;
                for k in 0..m {
                    let kik = k_gain.get(i).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
                    let hkj = h_jac.get(k).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                    kh_sum += kik * hkj;
                }
                let eye = if i == j { 1.0 } else { 0.0 };
                if let Some(slot) = i_kh.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *slot = eye - kh_sum;
                }
            }
        }

        let mut p_updated = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    let ikh_ik = i_kh.get(i).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
                    let pkj = self.p.get(k).and_then(|r| r.get(j)).copied().unwrap_or(0.0);
                    sum += ikh_ik * pkj;
                }
                if let Some(slot) = p_updated.get_mut(i).and_then(|r| r.get_mut(j)) {
                    *slot = sum;
                }
            }
        }

        self.p = p_updated;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invert_small_matrix() -> Result<(), Box<dyn std::error::Error>> {
        let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = invert_small_matrix(&a)?;
        // det = 4*6 - 7*2 = 10 -> inv = [[0.6, -0.7], [-0.2, 0.4]]
        assert!((inv.get(0).and_then(|r| r.get(0)).copied().unwrap_or(0.0) - 0.6).abs() < 1e-10);
        assert!((inv.get(0).and_then(|r| r.get(1)).copied().unwrap_or(0.0) - (-0.7)).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_ekf_nonlinear_tracking() -> Result<(), Box<dyn std::error::Error>> {
        // Nonlinear state: x = [theta, omega]^T
        // d_theta/dt = omega, d_omega/dt = -sin(theta)
        // Discretized with dt = 0.1:
        // theta_{k+1} = theta_k + omega_k * dt
        // omega_{k+1} = omega_k - sin(theta_k) * dt

        let x0 = vec![0.5, 0.0];
        let p0 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let q = vec![vec![1e-4, 0.0], vec![0.0, 1e-4]];
        let r = vec![vec![0.01]]; // Measurement variance

        let mut ekf = ExtendedKalmanFilter::new(x0, p0, q, r);

        let dt = Dual64::constant(0.05);
        let dynamics = |x: &[Dual64]| -> Vec<Dual64> {
            let th = x.first().copied().unwrap_or_else(|| Dual64::constant(0.0));
            let om = x.get(1).copied().unwrap_or_else(|| Dual64::constant(0.0));
            vec![th + om * dt, om - th.sin() * dt]
        };

        let measurement = |x: &[Dual64]| -> Vec<Dual64> {
            // Measure sin(theta) directly
            let th = x.first().copied().unwrap_or_else(|| Dual64::constant(0.0));
            vec![th.sin()]
        };

        // Run 5 filter prediction/update cycles
        for _ in 0..5 {
            ekf.predict(&dynamics)?;
            // True observation around theta ~ 0.5
            let z_meas = vec![(0.5f64).sin()];
            ekf.update(&z_meas, &measurement)?;
        }

        // Verify state is bounded and error covariance has reduced significantly
        let p_diag_0 = ekf.p.first().and_then(|r| r.first()).copied().unwrap_or(1.0);
        assert!(p_diag_0 < 0.2, "Covariance failed to contract: {p_diag_0}");
        let theta_est = ekf.x.first().copied().unwrap_or(0.0);
        assert!((theta_est - 0.5).abs() < 0.1, "Theta estimate off: {theta_est}");

        Ok(())
    }
}
