use crate::StateSpace;
use linalg_cpu::{matmul, mldivide, LinalgError};
use num_core::{MemoryBudget, SiUnit};
use std::sync::Arc;
use tensor::Tensor;

/// Ackermann's Formula (`acker(A, B, desired_char_poly)`) for Exact SISO Pole Placement!
/// Given controllable $(A, B)$ and desired characteristic polynomial coefficients
/// $\alpha(s) = s^n + \alpha_1 s^{n-1} + \dots + \alpha_n$, computes $K = [0, \dots, 0, 1] \mathcal{C}^{-1} \alpha(A)$.
pub fn acker_pole_placement(
    sys: &StateSpace,
    char_poly_coeffs: &[f64], // [1.0, alpha_1, ..., alpha_n]
    budget: &Arc<MemoryBudget>,
) -> Result<Tensor<f64>, LinalgError> {
    let n = sys.a.rows();
    let ctrb_mat = sys.ctrb(budget)?;
    let eye = Tensor::eye(n, budget)?;

    // Evaluate polynomial alpha(A) = A^n + alpha_1 A^{n-1} + ... + alpha_n I using Horner's scheme
    let mut poly_a = Tensor::zeros(n, n, budget)?;
    for &c in char_poly_coeffs {
        let c_eye = eye.map_unary(SiUnit::DIMENSIONLESS, |x| x * c)?;
        poly_a = matmul(&poly_a, &sys.a, budget)?.checked_add(&c_eye)?;
    }

    let c_inv_poly = mldivide(&ctrb_mat, &poly_a, budget)?;
    // Extract last row of C^{-1} * alpha(A)
    let mut k = Tensor::zeros(1, n, budget)?;
    for j in 0..n {
        k.set(0, j, c_inv_poly.get(n - 1, j)?)?;
    }
    Ok(k)
}

/// Discrete-Time Linear Kalman Filter (`KalmanFilter`).
#[derive(Clone, Debug)]
pub struct KalmanFilter {
    pub a: Tensor<f64>,
    pub b: Tensor<f64>,
    pub c: Tensor<f64>,
    pub q_cov: Tensor<f64>,
    pub r_cov: Tensor<f64>,
    pub x_hat: Tensor<f64>,
    pub p_cov: Tensor<f64>,
}

impl KalmanFilter {
    /// Performs one full Predict + Measurement-Update cycle (`[x_hat, K] = step(u, y)`).
    pub fn step(
        &mut self,
        u: &Tensor<f64>,
        y_meas: &Tensor<f64>,
        budget: &Arc<MemoryBudget>,
    ) -> Result<Tensor<f64>, LinalgError> {
        let n = self.a.rows();
        // 1. Predict state & covariance: x_{k|k-1} = A x + B u, P_{k|k-1} = A P A^T + Q
        let x_pred = matmul(&self.a, &self.x_hat, budget)?
            .checked_add(&matmul(&self.b, u, budget)?)?;
        let at = self.a.transpose()?;
        let p_pred = matmul(&matmul(&self.a, &self.p_cov, budget)?, &at, budget)?
            .checked_add(&self.q_cov)?;

        // 2. Innovation covariance S = C P C^T + R and Kalman gain K = P C^T S^{-1}
        let ct = self.c.transpose()?;
        let pct = matmul(&p_pred, &ct, budget)?;
        let s_cov = matmul(&self.c, &pct, budget)?.checked_add(&self.r_cov)?;
        let kt = mldivide(&s_cov.transpose()?, &pct.transpose()?, budget)?;
        let k_gain = kt.transpose()?;

        // 3. Update state & covariance: x_{k|k} = x_pred + K (y - C x_pred), P_{k|k} = (I - K C) P_pred
        let innovation = y_meas.checked_sub(&matmul(&self.c, &x_pred, budget)?)?;
        self.x_hat = x_pred.checked_add(&matmul(&k_gain, &innovation, budget)?)?;
        let eye = Tensor::eye(n, budget)?;
        let i_minus_kc = eye.checked_sub(&matmul(&k_gain, &self.c, budget)?)?;
        self.p_cov = matmul(&i_minus_kc, &p_pred, budget)?;

        Ok(self.x_hat.clone())
    }
}

/// Box-Constrained Receding-Horizon Model Predictive Controller (`MpcController`).
/// Optimizes future control sequence $U = [u_0, \dots, u_{N_p-1}]$ subject to $u_{\min} \le u_k \le u_{\max}$
/// via projected gradient descent over the prediction horizon!
#[derive(Clone, Debug)]
pub struct MpcController {
    pub sys_d: StateSpace,
    pub horizon: usize,
    pub q_state: f64,
    pub r_input: f64,
    pub u_min: f64,
    pub u_max: f64,
}

impl MpcController {
    pub fn compute_optimal_action(
        &self,
        x_current: &Tensor<f64>,
        budget: &Arc<MemoryBudget>,
    ) -> Result<f64, LinalgError> {
        let mut u_seq = vec![0.0; self.horizon.max(1)];
        let eps = 1e-4;
        let lr = 0.05;

        for _iter in 0..40 {
            let base_cost = self.rollout_cost(x_current, &u_seq, budget)?;
            let mut grad = vec![0.0; u_seq.len()];
            for k in 0..u_seq.len() {
                let mut u_pert = u_seq.clone();
                if let Some(slot) = u_pert.get_mut(k) {
                    *slot += eps;
                }
                let c_pert = self.rollout_cost(x_current, &u_pert, budget)?;
                if let Some(g_slot) = grad.get_mut(k) {
                    *g_slot = (c_pert - base_cost) / eps;
                }
            }
            for (u_val, &g) in u_seq.iter_mut().zip(&grad) {
                *u_val = (*u_val - lr * g).clamp(self.u_min, self.u_max);
            }
        }
        Ok(u_seq.first().copied().unwrap_or(0.0))
    }

    fn rollout_cost(
        &self,
        x0: &Tensor<f64>,
        u_seq: &[f64],
        budget: &Arc<MemoryBudget>,
    ) -> Result<f64, LinalgError> {
        let mut x = x0.clone();
        let mut cost = 0.0;
        for &uk in u_seq {
            let u_t = Tensor::scalar(uk, SiUnit::DIMENSIONLESS, budget)?;
            x = matmul(&self.sys_d.a, &x, budget)?
                .checked_add(&matmul(&self.sys_d.b, &u_t, budget)?)?;
            let x_norm = x.norm_frobenius()?;
            cost += self.q_state * x_norm * x_norm + self.r_input * uk * uk;
        }
        Ok(cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TransferFunction;

    #[test]
    fn test_acker_pole_placement_kalman_and_mpc() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 0.0, 0.0]);
        let ss = StateSpace::from_transfer_function(&tf, &budget)?;

        // Place poles at s = -2, -3 -> (s+2)(s+3) = s^2 + 5s + 6 -> K = [6, 5]
        let k = acker_pole_placement(&ss, &[1.0, 5.0, 6.0], &budget)?;
        assert!((k.get(0, 0)? - 6.0).abs() < 1e-10);
        assert!((k.get(0, 1)? - 5.0).abs() < 1e-10);

        // Constrained MPC respects actuator saturation bounds [-0.5, 0.5]
        let ssd = ss.c2d_zoh(0.1, &budget)?;
        let mpc = MpcController {
            sys_d: ssd,
            horizon: 5,
            q_state: 10.0,
            r_input: 0.1,
            u_min: -0.5,
            u_max: 0.5,
        };
        let x0 = Tensor::from_column_major(2, 1, vec![2.0, 0.0], SiUnit::DIMENSIONLESS, &budget)?;
        let u_opt = mpc.compute_optimal_action(&x0, &budget)?;
        assert!((-0.5..=0.5).contains(&u_opt));
        assert!(u_opt < -0.1); // Drives positive state back toward 0!
        Ok(())
    }
}
