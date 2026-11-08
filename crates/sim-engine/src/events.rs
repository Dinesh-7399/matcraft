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

//! Discontinuous Zero-Crossing Detection & Illinois Event Root-Finding.
//! Provides Simulink-grade zero-crossing event localization to machine precision,
//! preventing energy drift, numerical chatter, and overstepping during hybrid state transitions.

use crate::SimError;
use serde::{Deserialize, Serialize};

/// Direction filter for zero-crossing detection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ZeroCrossingDirection {
    #[default]
    Any,
    Rising,
    Falling,
}

impl ZeroCrossingDirection {
    /// Tests whether a transition from $g_{prev}$ to $g_{curr}$ qualifies as an event.
    #[must_use]
    pub fn is_triggered(&self, g_prev: f64, g_curr: f64) -> bool {
        match self {
            Self::Any => (g_prev < -1e-8 && g_curr >= 0.0) || (g_prev > 1e-8 && g_curr <= 0.0),
            Self::Rising => g_prev < -1e-8 && g_curr >= 0.0,
            Self::Falling => g_prev > 1e-8 && g_curr <= 0.0,
        }
    }
}

/// Finds the exact root $t^* \in [a, b]$ where $g(t^*) = 0$ using the modified Illinois false-position bracket solver.
///
/// Guaranteed to converge superlinearly without bracket loss ($g(a) \cdot g(b) \le 0$).
pub fn find_zero_crossing_time<F>(
    mut a: f64,
    mut b: f64,
    mut ga: f64,
    mut gb: f64,
    tol: f64,
    max_iter: usize,
    mut g_eval: F,
) -> Result<f64, SimError>
where
    F: FnMut(f64) -> f64,
{
    if ga.abs() < 1e-12 {
        return Ok(a);
    }
    if gb.abs() < 1e-12 {
        return Ok(b);
    }

    if ga * gb > 0.0 {
        return Err(SimError::EventRootFindingFailed(format!(
            "Root not bracketed: g({a}) = {ga}, g({b}) = {gb}"
        )));
    }

    let effective_tol = tol.max(1e-12);
    let mut side = 0i32;

    for _ in 0..max_iter {
        if (b - a).abs() <= effective_tol {
            return Ok(0.5 * (a + b));
        }

        // Secant interpolation
        let denominator = gb - ga;
        if denominator.abs() < 1e-15 {
            return Ok(0.5 * (a + b));
        }

        let mut c = (a * gb - b * ga) / denominator;
        if c <= a || c >= b {
            c = 0.5 * (a + b);
        }
        let gc = g_eval(c);

        if gc.abs() < 1e-12 {
            return Ok(c);
        }

        if ga * gc > 0.0 {
            a = c;
            ga = gc;
            if side == -1 {
                gb *= 0.5; // Illinois modification to prevent sluggish stagnation
            }
            side = -1;
        } else {
            b = c;
            gb = gc;
            if side == 1 {
                ga *= 0.5; // Illinois modification
            }
            side = 1;
        }
    }

    Ok(0.5 * (a + b))
}

/// Hybrid state reset action when a zero-crossing event fires.
#[derive(Clone, Debug, PartialEq)]
pub enum EventResetAction {
    /// Inelastic/elastic velocity reflection: $v^+ = -\text{elasticity} \cdot |v^-|$
    VelocityReflection {
        velocity_idx: usize,
        elasticity: f64,
        position_clamp: Option<(usize, f64)>,
    },
    /// Clamp state variable to a boundary value: $x[\text{idx}] = \text{val}$
    ClampState { state_idx: usize, value: f64 },
}

/// A zero-crossing event descriptor.
#[derive(Clone, Debug)]
pub struct ZeroCrossingSpec {
    pub id: usize,
    pub name: String,
    pub direction: ZeroCrossingDirection,
    pub tolerance: f64,
    pub action: EventResetAction,
}

/// Hybrid event simulation result trace.
#[derive(Clone, Debug, Default)]
pub struct HybridSimTrace {
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
    pub event_times: Vec<f64>,
    pub event_ids: Vec<usize>,
}

/// Runs a hybrid simulation with continuous Runge-Kutta 4th order stepping and exact zero-crossing event localization.
pub fn simulate_with_events<F, G>(
    x0: &[f64],
    t_end: f64,
    dt: f64,
    events: &[ZeroCrossingSpec],
    dynamics: F,
    event_indicator: G,
) -> Result<HybridSimTrace, SimError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    G: Fn(f64, &[f64], usize) -> f64,
{
    if !t_end.is_finite() || !dt.is_finite() || t_end <= 0.0 || dt <= 0.0 {
        return Err(SimError::InvalidTime(format!(
            "Invalid t_end={t_end} or dt={dt}"
        )));
    }

    let n = x0.len();
    let mut t = 0.0;
    let mut x = x0.to_vec();

    let mut trace = HybridSimTrace::default();
    trace.times.push(t);
    trace.states.push(x.clone());

    let rk4_step = |t_curr: f64, x_curr: &[f64], h: f64| -> Vec<f64> {
        let k1 = dynamics(t_curr, x_curr);
        let mut x_k2 = Vec::with_capacity(n);
        for i in 0..n {
            let xi = x_curr.get(i).copied().unwrap_or(0.0);
            let k1i = k1.get(i).copied().unwrap_or(0.0);
            x_k2.push(xi + 0.5 * h * k1i);
        }
        let k2 = dynamics(t_curr + 0.5 * h, &x_k2);

        let mut x_k3 = Vec::with_capacity(n);
        for i in 0..n {
            let xi = x_curr.get(i).copied().unwrap_or(0.0);
            let k2i = k2.get(i).copied().unwrap_or(0.0);
            x_k3.push(xi + 0.5 * h * k2i);
        }
        let k3 = dynamics(t_curr + 0.5 * h, &x_k3);

        let mut x_k4 = Vec::with_capacity(n);
        for i in 0..n {
            let xi = x_curr.get(i).copied().unwrap_or(0.0);
            let k3i = k3.get(i).copied().unwrap_or(0.0);
            x_k4.push(xi + h * k3i);
        }
        let k4 = dynamics(t_curr + h, &x_k4);

        let mut x_next = Vec::with_capacity(n);
        for i in 0..n {
            let xi = x_curr.get(i).copied().unwrap_or(0.0);
            let k1i = k1.get(i).copied().unwrap_or(0.0);
            let k2i = k2.get(i).copied().unwrap_or(0.0);
            let k3i = k3.get(i).copied().unwrap_or(0.0);
            let k4i = k4.get(i).copied().unwrap_or(0.0);
            x_next.push(xi + (h / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i));
        }
        x_next
    };

    let mut step_count = 0usize;
    const MAX_SIM_STEPS: usize = 1_000_000;
    while t < t_end {
        if step_count >= MAX_SIM_STEPS {
            return Err(SimError::ConvergenceFailed(format!(
                "Simulation exceeded maximum step count of {MAX_SIM_STEPS}"
            )));
        }
        step_count = step_count.saturating_add(1);
        let h = dt.min(t_end - t);
        let x_trial = rk4_step(t, &x, h);
        let t_trial = t + h;

        // Check all event indicators
        let mut triggered_event = None;
        let mut min_event_t = t_trial;

        for ev in events {
            let g_prev = event_indicator(t, &x, ev.id);
            let g_trial = event_indicator(t_trial, &x_trial, ev.id);

            if ev.direction.is_triggered(g_prev, g_trial) {
                // Bracket solver for exact event time
                let g_eval = |t_mid: f64| -> f64 {
                    let dt_mid = (t_mid - t).max(0.0);
                    let x_mid = rk4_step(t, &x, dt_mid);
                    event_indicator(t_mid, &x_mid, ev.id)
                };

                if let Ok(t_star) = find_zero_crossing_time(
                    t,
                    t_trial,
                    g_prev,
                    g_trial,
                    ev.tolerance,
                    50,
                    g_eval,
                ) {
                    if t_star < min_event_t {
                        min_event_t = t_star;
                        triggered_event = Some(ev);
                    }
                }
            }
        }

        if let Some(ev) = triggered_event {
            // Step up to exact event time t_star
            let h_event = (min_event_t - t).max(1e-12);
            let mut x_at_event = rk4_step(t, &x, h_event);

            // Execute reset map
            match ev.action {
                EventResetAction::VelocityReflection {
                    velocity_idx,
                    elasticity,
                    position_clamp,
                } => {
                    if let Some(v) = x_at_event.get_mut(velocity_idx) {
                        *v = elasticity * (*v).abs(); // reflect velocity upwards
                    }
                    if let Some((pos_idx, clamp_val)) = position_clamp {
                        if let Some(p) = x_at_event.get_mut(pos_idx) {
                            *p = clamp_val;
                        }
                    }
                }
                EventResetAction::ClampState { state_idx, value } => {
                    if let Some(s) = x_at_event.get_mut(state_idx) {
                        *s = value;
                    }
                }
            }

            t = (min_event_t + 1e-8).min(t_end);
            x = x_at_event;
            trace.times.push(t);
            trace.states.push(x.clone());
            trace.event_times.push(t);
            trace.event_ids.push(ev.id);
        } else {
            t = t_trial;
            x = x_trial;
            trace.times.push(t);
            trace.states.push(x.clone());
        }
    }

    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_illinois_root_finding_polynomial() -> Result<(), SimError> {
        // g(t) = t^2 - 4 on [1.0, 3.0], root at t = 2.0
        let g = |t: f64| t * t - 4.0;
        let root = find_zero_crossing_time(1.0, 3.0, g(1.0), g(3.0), 1e-9, 40, g)?;
        assert!((root - 2.0).abs() < 1e-8, "Root mismatch: {root}");
        Ok(())
    }

    #[test]
    fn test_bouncing_ball_hybrid_simulation() -> Result<(), SimError> {
        // State x = [h, v]^T (height and velocity)
        // dh/dt = v
        // dv/dt = -9.81
        // Event: h <= 0 with Falling direction
        // Reset: v^+ = -0.8 * v^- (elastic collision)
        let x0 = vec![5.0, 0.0]; // Drop from 5 meters with 0 initial velocity
        let events = vec![ZeroCrossingSpec {
            id: 1,
            name: "Ground Contact".into(),
            direction: ZeroCrossingDirection::Falling,
            tolerance: 1e-8,
            action: EventResetAction::VelocityReflection {
                velocity_idx: 1,
                elasticity: 0.8,
                position_clamp: Some((0, 0.0)),
            },
        }];

        let dynamics = |_t: f64, x: &[f64]| -> Vec<f64> {
            let v = x.get(1).copied().unwrap_or(0.0);
            vec![v, -9.81]
        };

        let indicator = |_t: f64, x: &[f64], _ev_id: usize| -> f64 {
            // Event occurs when height reaches 0
            x.first().copied().unwrap_or(0.0)
        };

        let trace = simulate_with_events(&x0, 2.5, 0.02, &events, dynamics, indicator)?;

        // Theoretical first bounce time: t = sqrt(2*h0 / g) = sqrt(10 / 9.81) = 1.0096 s
        assert!(!trace.event_times.is_empty(), "No bounce events detected");
        let first_bounce = trace.event_times.first().copied().unwrap_or(0.0);
        assert!(
            (first_bounce - 1.0096).abs() < 0.02,
            "First bounce time mismatch: got {first_bounce}"
        );

        // Verify height never penetrates deeply into negative ground
        for state in &trace.states {
            let h = state.first().copied().unwrap_or(0.0);
            assert!(h >= -1e-4, "Ground penetration violation: {h}");
        }

        Ok(())
    }
}
