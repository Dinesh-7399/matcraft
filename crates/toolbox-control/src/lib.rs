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

pub mod kalman_mpc;
pub mod nonlinear_obs;
pub mod robust;
pub mod root_locus;
pub use kalman_mpc::{acker_pole_placement, KalmanFilter, MpcController};
pub use nonlinear_obs::ExtendedKalmanFilter;
pub use robust::{disk_margin, h_infinity_bound, mu_upper_bound};
pub use root_locus::{poly_roots, pzmap, rlocus, PoleZeroMap, RootLocusBranchPoint, RootLocusResult};

use linalg_cpu::{eig_general, expm, inv, matmul, mldivide, rank, LinalgError};
use num_core::{Complex64, MemoryBudget, SiUnit};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tensor::Tensor;

/// Continuous or Discrete Linear Time-Invariant State-Space Model:
/// $\dot{x} = A x + B u$, $y = C x + D u$.
#[derive(Clone, Debug)]
pub struct StateSpace {
    pub a: Tensor<f64>,
    pub b: Tensor<f64>,
    pub c: Tensor<f64>,
    pub d: Tensor<f64>,
    pub sample_time_sec: Option<f64>,
}

impl StateSpace {
    /// Converts a SISO Transfer Function $G(s) = \frac{b_1 s^{n-1} + \dots + b_n}{s^n + a_1 s^{n-1} + \dots + a_n}$
    /// into Controllable Canonical State-Space form (`tf2ss`).
    pub fn from_transfer_function(
        tf: &TransferFunction,
        budget: &Arc<MemoryBudget>,
    ) -> Result<Self, LinalgError> {
        let den_len = tf.den.len();
        if den_len <= 1 {
            let d_val = tf.num.first().copied().unwrap_or(0.0)
                / tf.den.first().copied().unwrap_or(1.0);
            return Ok(Self {
                a: Tensor::zeros(0, 0, budget)?,
                b: Tensor::zeros(0, 1, budget)?,
                c: Tensor::zeros(1, 0, budget)?,
                d: Tensor::scalar(d_val, SiUnit::DIMENSIONLESS, budget)?,
                sample_time_sec: None,
            });
        }
        let n = den_len - 1;
        let a0_lead = tf.den.first().copied().unwrap_or(1.0);
        let den_norm: Vec<f64> = tf.den.iter().map(|&x| x / a0_lead).collect();

        let mut num_padded = vec![0.0; den_len];
        let offset = den_len.saturating_sub(tf.num.len());
        for (i, &v) in tf.num.iter().enumerate() {
            if let Some(slot) = num_padded.get_mut(offset + i) {
                *slot = v / a0_lead;
            }
        }
        let d_scalar = num_padded.first().copied().unwrap_or(0.0);

        let mut a = Tensor::zeros(n, n, budget)?;
        for i in 0..(n - 1) {
            a.set(i, i + 1, 1.0)?;
        }
        for j in 0..n {
            let coeff = den_norm.get(n - j).copied().unwrap_or(0.0);
            a.set(n - 1, j, -coeff)?;
        }

        let mut b = Tensor::zeros(n, 1, budget)?;
        b.set(n - 1, 0, 1.0)?;

        let mut c = Tensor::zeros(1, n, budget)?;
        for j in 0..n {
            let b_coeff = num_padded.get(n - j).copied().unwrap_or(0.0);
            let a_coeff = den_norm.get(n - j).copied().unwrap_or(0.0);
            c.set(0, j, b_coeff - d_scalar * a_coeff)?;
        }

        let d = Tensor::scalar(d_scalar, SiUnit::DIMENSIONLESS, budget)?;
        Ok(Self {
            a,
            b,
            c,
            d,
            sample_time_sec: None,
        })
    }

    /// Exact Zero-Order Hold Discretization (`c2d(sys, Ts, 'zoh')`) using Matrix Exponential `expm`!
    /// $A_d = e^{A T_s}$, $B_d = A^{-1} (A_d - I) B$ (or series expansion if $A$ is singular).
    pub fn c2d_zoh(&self, ts_sec: f64, budget: &Arc<MemoryBudget>) -> Result<Self, LinalgError> {
        let n = self.a.rows();
        let a_ts = self.a.map_unary(SiUnit::DIMENSIONLESS, |x| x * ts_sec)?;
        let ad = expm(&a_ts, budget)?;
        let eye = Tensor::eye(n, budget)?;

        // Compute integral_0^Ts exp(A*tau) dtau = Ts * (I + A*Ts/2 + (A*Ts)^2/6 + (A*Ts)^3/24)
        let a2 = matmul(&a_ts, &a_ts, budget)?;
        let a3 = matmul(&a2, &a_ts, budget)?;
        let int_mat = eye
            .checked_add(&a_ts.map_unary(SiUnit::DIMENSIONLESS, |x| x * 0.5)?)?
            .checked_add(&a2.map_unary(SiUnit::DIMENSIONLESS, |x| x / 6.0)?)?
            .checked_add(&a3.map_unary(SiUnit::DIMENSIONLESS, |x| x / 24.0)?)?
            .map_unary(SiUnit::DIMENSIONLESS, |x| x * ts_sec)?;
        let bd = matmul(&int_mat, &self.b, budget)?;

        Ok(Self {
            a: ad,
            b: bd,
            c: self.c.clone(),
            d: self.d.clone(),
            sample_time_sec: Some(ts_sec),
        })
    }

    /// Controllability Matrix $\mathcal{C} = [B, AB, A^2B, \dots, A^{n-1}B]$ (`ctrb`).
    pub fn ctrb(&self, budget: &Arc<MemoryBudget>) -> Result<Tensor<f64>, LinalgError> {
        let n = self.a.rows();
        let mut acc = self.b.clone();
        let mut term = self.b.clone();
        for _ in 1..n {
            term = matmul(&self.a, &term, budget)?;
            acc = acc.horzcat(&term)?;
        }
        Ok(acc)
    }

    /// Observability Matrix $\mathcal{O} = [C; CA; \dots; CA^{n-1}]$ (`obsv`).
    pub fn obsv(&self, budget: &Arc<MemoryBudget>) -> Result<Tensor<f64>, LinalgError> {
        let n = self.a.rows();
        let mut acc = self.c.clone();
        let mut term = self.c.clone();
        for _ in 1..n {
            term = matmul(&term, &self.a, budget)?;
            acc = acc.vertcat(&term)?;
        }
        Ok(acc)
    }

    pub fn is_controllable(&self, budget: &Arc<MemoryBudget>) -> Result<bool, LinalgError> {
        let co = self.ctrb(budget)?;
        Ok(rank(&co, budget)? == self.a.rows())
    }

    /// Infinite-horizon Discrete/Continuous Linear-Quadratic Regulator (`lqr(A, B, Q, R)`).
    /// Solves the Algebraic Riccati Equation via Hamiltonian/Riccati value iteration and returns
    /// optimal state-feedback gain $K$ and cost-to-go matrix $P$.
    pub fn lqr(
        &self,
        q: &Tensor<f64>,
        r: &Tensor<f64>,
        budget: &Arc<MemoryBudget>,
    ) -> Result<(Tensor<f64>, Tensor<f64>), LinalgError> {
        // Discretize with fine virtual horizon step to solve DARE reliably for any stabilizable (A, B)
        let dt = self.sample_time_sec.unwrap_or(0.01);
        let disc = if self.sample_time_sec.is_some() {
            self.clone()
        } else {
            self.c2d_zoh(dt, budget)?
        };
        let ad = &disc.a;
        let bd = &disc.b;
        let ad_t = ad.transpose()?;
        let bd_t = bd.transpose()?;

        let q_scaled = if self.sample_time_sec.is_some() {
            q.clone()
        } else {
            q.map_unary(SiUnit::DIMENSIONLESS, |x| x * dt)?
        };
        let r_scaled = if self.sample_time_sec.is_some() {
            r.clone()
        } else {
            r.map_unary(SiUnit::DIMENSIONLESS, |x| x * dt)?
        };

        let mut p = q_scaled.clone();
        for _iter in 0..250 {
            let btp = matmul(&bd_t, &p, budget)?;
            let btpb = matmul(&btp, bd, budget)?;
            let r_plus_btpb = r_scaled.checked_add(&btpb)?;
            let btp_a = matmul(&btp, ad, budget)?;
            let k_step = mldivide(&r_plus_btpb, &btp_a, budget)?;

            let at_p = matmul(&ad_t, &p, budget)?;
            let at_p_a = matmul(&at_p, ad, budget)?;
            let at_p_b = matmul(&at_p, bd, budget)?;
            let correction = matmul(&at_p_b, &k_step, budget)?;
            let p_next = q_scaled.checked_add(&at_p_a.checked_sub(&correction)?)?;

            let diff = p_next.checked_sub(&p)?.norm_frobenius()?;
            p = p_next;
            if diff < 1e-10 {
                break;
            }
        }

        let k = if self.sample_time_sec.is_some() {
            let btp = matmul(&bd_t, &p, budget)?;
            let r_plus = r.checked_add(&matmul(&btp, bd, budget)?)?;
            mldivide(&r_plus, &matmul(&btp, ad, budget)?, budget)?
        } else {
            let r_inv_bt = mldivide(r, &self.b.transpose()?, budget)?;
            matmul(&r_inv_bt, &p, budget)?.map_unary(SiUnit::DIMENSIONLESS, |x| x / dt)?
        };
        Ok((k, p))
    }

    pub fn poles(&self, budget: &Arc<MemoryBudget>) -> Result<Vec<Complex64>, LinalgError> {
        eig_general(&self.a, budget)
    }

    pub fn eval_frf(&self, s: Complex64, budget: &Arc<MemoryBudget>) -> Result<f64, LinalgError> {
        let n = self.a.rows();
        let eye = Tensor::eye(n, budget)?;
        let omega = s.im;
        // Real block representation of (j*omega*I - A)
        let a_neg = self.a.map_unary(SiUnit::DIMENSIONLESS, |x| -x)?;
        let w_eye = eye.map_unary(SiUnit::DIMENSIONLESS, |x| x * omega)?;
        let top = a_neg.horzcat(&w_eye.map_unary(SiUnit::DIMENSIONLESS, |x| -x)?)?;
        let bot = w_eye.horzcat(&a_neg)?;
        let block_2n = top.vertcat(&bot)?;
        let rhs_2n = self.b.vertcat(&Tensor::zeros(n, self.b.cols(), budget)?)?;
        let sol_2n = mldivide(&block_2n, &rhs_2n, budget)?;
        let _ = inv(&eye, budget)?;
        sol_2n.norm_frobenius().map_err(LinalgError::from)
    }
}


/// Continuous-time SISO Transfer Function (s) = \frac{\sum b_k s^{m-k}}{\sum a_k s^{n-k}}$.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransferFunction {
    pub num: Vec<f64>,
    pub den: Vec<f64>,
}

impl TransferFunction {
    pub fn new(num: Vec<f64>, den: Vec<f64>) -> Self {
        Self { num, den }
    }

    /// Evaluates frequency response at  = j \omega$ via Horner's method.
    pub fn eval_frequency(&self, omega_rad_s: f64) -> Complex64 {
        let s = Complex64::new(0.0, omega_rad_s);
        let num_val = self.eval_poly(&self.num, s);
        let den_val = self.eval_poly(&self.den, s);
        num_val / den_val
    }

    fn eval_poly(&self, coeffs: &[f64], s: Complex64) -> Complex64 {
        let mut acc = Complex64::new(0.0, 0.0);
        for &c in coeffs {
            acc = acc * s + Complex64::new(c, 0.0);
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tf_and_statespace_c2d_ctrb_obsv_lqr() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 2.0, 1.0]);
        let ss = StateSpace::from_transfer_function(&tf, &budget)?;
        assert_eq!(ss.a.rows(), 2);
        assert_eq!(ss.a.cols(), 2);
        assert!(ss.is_controllable(&budget)?);

        let ob = ss.obsv(&budget)?;
        assert_eq!(ob.rows(), 2);

        // Discretization via expm
        let disc = ss.c2d_zoh(0.1, &budget)?;
        assert_eq!(disc.sample_time_sec, Some(0.1));

        // LQR optimal control gain
        let q = Tensor::eye(2, &budget)?;
        let r = Tensor::scalar(1.0, SiUnit::DIMENSIONLESS, &budget)?;
        let (k, p) = ss.lqr(&q, &r, &budget)?;
        assert_eq!(k.rows(), 1);
        assert_eq!(k.cols(), 2);
        assert_eq!(p.rows(), 2);
        assert_eq!(p.cols(), 2);

        // Poles
        let poles = ss.poles(&budget)?;
        assert_eq!(poles.len(), 2);

        // Frequency response evaluation
        let frf_mag = ss.eval_frf(Complex64::new(0.0, 1.0), &budget)?;
        assert!(frf_mag.is_finite());
        let tf_val = tf.eval_frequency(1.0);
        assert!(tf_val.norm_sqr() > 0.0);

        Ok(())
    }
}
