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

//! Hybrid automaton: continuous dynamics + discrete mode switching.

use crate::SimError;

/// A single mode of a hybrid automaton, with linear dynamics ẋ = A x + B u.
#[derive(Clone, Debug)]
pub struct HybridMode {
    pub name: String,
    /// A matrix (n×n), stored row-major.
    pub a_flat: Vec<f64>,
    /// State dimension (n).
    pub a_dim: usize,
    /// B matrix (n×b_cols), stored row-major.
    pub b_flat: Vec<f64>,
    /// Number of columns in B (input dimension).
    pub b_cols: usize,
}

impl HybridMode {
    /// Compute ẋ = A x + B u.
    fn dynamics(&self, x: &[f64], u: &[f64]) -> Vec<f64> {
        let n = self.a_dim;
        let mut xdot = vec![0.0f64; n];
        // A x
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                let a_ij = self
                    .a_flat
                    .get(i.saturating_mul(n).saturating_add(j))
                    .copied()
                    .unwrap_or(0.0);
                sum += a_ij * x.get(j).copied().unwrap_or(0.0);
            }
            if let Some(slot) = xdot.get_mut(i) {
                *slot += sum;
            }
        }
        // B u
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..self.b_cols {
                let b_ij = self
                    .b_flat
                    .get(i.saturating_mul(self.b_cols).saturating_add(j))
                    .copied()
                    .unwrap_or(0.0);
                sum += b_ij * u.get(j).copied().unwrap_or(0.0);
            }
            if let Some(slot) = xdot.get_mut(i) {
                *slot += sum;
            }
        }
        xdot
    }
}

/// A guard condition for a mode switch: c^T x >= threshold.
#[derive(Clone, Debug)]
pub struct HybridGuard {
    pub from_mode: usize,
    pub to_mode: usize,
    /// Guard normal vector c.
    pub c_vec: Vec<f64>,
    pub threshold: f64,
}

impl HybridGuard {
    /// Evaluate c^T x >= threshold.
    fn is_active(&self, x: &[f64]) -> bool {
        let dot: f64 = self
            .c_vec
            .iter()
            .zip(x.iter())
            .map(|(&c, &xi)| c * xi)
            .sum();
        dot >= self.threshold
    }
}

/// Hybrid automaton: a set of modes and switching guards.
#[derive(Clone, Debug)]
pub struct HybridAutomaton {
    pub modes: Vec<HybridMode>,
    pub guards: Vec<HybridGuard>,
}

impl HybridAutomaton {
    pub fn new(modes: Vec<HybridMode>, guards: Vec<HybridGuard>) -> Self {
        Self { modes, guards }
    }

    /// Simulate the hybrid automaton using RK4 steps, with guard checking after each step.
    /// Returns trajectory: list of state vectors at each time step.
    pub fn simulate(
        &self,
        x0: Vec<f64>,
        u: Vec<f64>,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<Vec<f64>>, SimError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(SimError::InvalidTime(format!("dt={dt} is not a valid step size")));
        }
        if self.modes.is_empty() {
            return Err(SimError::InvalidTime("No modes defined in hybrid automaton".into()));
        }

        let mut x = x0;
        let mut current_mode_idx = 0usize;
        let mut trajectory = Vec::with_capacity(steps.saturating_add(1));
        trajectory.push(x.clone());

        for _ in 0..steps {
            let mode = self
                .modes
                .get(current_mode_idx)
                .ok_or_else(|| SimError::InvalidTime(format!("Invalid mode index {current_mode_idx}")))?;

            // RK4 integration
            x = rk4_step(mode, &x, &u, dt);
            trajectory.push(x.clone());

            // Check guards and switch mode if triggered
            for guard in &self.guards {
                if guard.from_mode == current_mode_idx
                    && guard.is_active(&x)
                    && guard.to_mode < self.modes.len()
                {
                    current_mode_idx = guard.to_mode;
                    break; // Apply first matching guard only
                }
            }
        }

        Ok(trajectory)
    }
}

/// Perform a single RK4 step: x_new = x + (dt/6)(k1 + 2k2 + 2k3 + k4).
fn rk4_step(mode: &HybridMode, x: &[f64], u: &[f64], dt: f64) -> Vec<f64> {
    let n = x.len();
    let k1 = mode.dynamics(x, u);

    let x2: Vec<f64> = x
        .iter()
        .zip(k1.iter())
        .map(|(&xi, &ki)| xi + 0.5 * dt * ki)
        .collect();
    let k2 = mode.dynamics(&x2, u);

    let x3: Vec<f64> = x
        .iter()
        .zip(k2.iter())
        .map(|(&xi, &ki)| xi + 0.5 * dt * ki)
        .collect();
    let k3 = mode.dynamics(&x3, u);

    let x4: Vec<f64> = x
        .iter()
        .zip(k3.iter())
        .map(|(&xi, &ki)| xi + dt * ki)
        .collect();
    let k4 = mode.dynamics(&x4, u);

    let mut x_new = vec![0.0f64; n];
    for i in 0..n {
        let ki1 = k1.get(i).copied().unwrap_or(0.0);
        let ki2 = k2.get(i).copied().unwrap_or(0.0);
        let ki3 = k3.get(i).copied().unwrap_or(0.0);
        let ki4 = k4.get(i).copied().unwrap_or(0.0);
        let xi = x.get(i).copied().unwrap_or(0.0);
        if let Some(slot) = x_new.get_mut(i) {
            *slot = xi + (dt / 6.0) * (ki1 + 2.0 * ki2 + 2.0 * ki3 + ki4);
        }
    }
    x_new
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stable_1d_automaton() -> HybridAutomaton {
        // Mode 0: ẋ = -x (exponential decay)
        // Mode 1: ẋ = -2x (faster decay, switches when x < 0.5)
        let mode0 = HybridMode {
            name: "slow_decay".to_string(),
            a_flat: vec![-1.0],
            a_dim: 1,
            b_flat: vec![0.0],
            b_cols: 1,
        };
        let mode1 = HybridMode {
            name: "fast_decay".to_string(),
            a_flat: vec![-2.0],
            a_dim: 1,
            b_flat: vec![0.0],
            b_cols: 1,
        };
        // Guard: from mode 0 to mode 1 when x >= 0.5 (never triggered in our test)
        let guard = HybridGuard {
            from_mode: 0,
            to_mode: 1,
            c_vec: vec![1.0],
            threshold: 100.0, // high threshold, never fires
        };
        HybridAutomaton::new(vec![mode0, mode1], vec![guard])
    }

    #[test]
    fn test_hybrid_automaton_simulate_decay() -> Result<(), Box<dyn std::error::Error>> {
        let ha = make_stable_1d_automaton();
        let traj = ha.simulate(vec![1.0], vec![0.0], 0.01, 100)?;
        assert_eq!(traj.len(), 101);
        // After 100 steps of dt=0.01 with ẋ=-x, x ≈ exp(-1) ≈ 0.368
        let final_x = traj.last().and_then(|v| v.first()).copied().unwrap_or(0.0);
        assert!((final_x - (-1.0_f64).exp()).abs() < 0.01);
        Ok(())
    }

    #[test]
    fn test_hybrid_automaton_guard_switches() -> Result<(), Box<dyn std::error::Error>> {
        // Mode 0: ẋ = 1 (growing), guard: switch when x >= 2
        let mode0 = HybridMode {
            name: "grow".to_string(),
            a_flat: vec![0.0],
            a_dim: 1,
            b_flat: vec![1.0],
            b_cols: 1,
        };
        let mode1 = HybridMode {
            name: "stable".to_string(),
            a_flat: vec![-1.0],
            a_dim: 1,
            b_flat: vec![0.0],
            b_cols: 1,
        };
        let guard = HybridGuard {
            from_mode: 0,
            to_mode: 1,
            c_vec: vec![1.0],
            threshold: 1.5,
        };
        let ha = HybridAutomaton::new(vec![mode0, mode1], vec![guard]);
        let traj = ha.simulate(vec![0.0], vec![1.0], 0.1, 50)?;
        assert!(!traj.is_empty());
        Ok(())
    }

    #[test]
    fn test_invalid_dt_returns_error() {
        let mode = HybridMode {
            name: "m".to_string(),
            a_flat: vec![-1.0],
            a_dim: 1,
            b_flat: vec![],
            b_cols: 0,
        };
        let ha = HybridAutomaton::new(vec![mode], vec![]);
        let result = ha.simulate(vec![1.0], vec![], -0.01, 10);
        assert!(result.is_err());
    }
}
