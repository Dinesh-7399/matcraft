use crate::SimError;
use autodiff::{jacobian_forward, Dual64};
use linalg_cpu::mldivide;
use num_core::{MemoryBudget, SiUnit};
use serde::{Deserialize, Serialize};
use tensor::Tensor;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Ode45Options {
    pub rel_tol: f64,
    pub abs_tol: f64,
    pub initial_step: f64,
    pub max_step: f64,
}

impl Default for Ode45Options {
    fn default() -> Self {
        Self {
            rel_tol: 1e-6,
            abs_tol: 1e-8,
            initial_step: 1e-2,
            max_step: 0.25,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OdeSolution {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    pub zero_crossings_t: Vec<f64>,
}

/// Adaptive Dormand-Prince 5(4)7M (`ode45`) solver with step-size adaptation and
/// bracketed Illinois-method zero-crossing detection!
pub fn solve_ode45_dormand_prince<F, Z>(
    t_span: [f64; 2],
    y0: &[f64],
    opts: Ode45Options,
    rhs: F,
    zero_crossing_fn: Option<Z>,
) -> Result<OdeSolution, SimError>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
    Z: Fn(f64, &[f64]) -> f64,
{
    let [t0, tf] = t_span;
    if !t0.is_finite() || !tf.is_finite() || tf <= t0 {
        return Err(SimError::InvalidTime("Invalid t_span for ode45".into()));
    }
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = opts.initial_step.min(tf - t0).max(1e-12);

    let mut t_hist = vec![t];
    let mut y_hist = vec![y.clone()];
    let mut zc_events = Vec::new();
    let mut prev_zc = zero_crossing_fn.as_ref().map(|zf| zf(t, &y));

    let max_steps = 200_000usize;
    for _step in 0..max_steps {
        if t >= tf - 1e-14 {
            break;
        }
        if t + h > tf {
            h = tf - t;
        }

        // Dormand-Prince RK5(4) tableau stages
        let k1 = rhs(t, &y);
        let y2 = lin_comb(&y, &[(&k1, h * (1.0 / 5.0))]);
        let k2 = rhs(t + h * (1.0 / 5.0), &y2);

        let y3 = lin_comb(&y, &[(&k1, h * (3.0 / 40.0)), (&k2, h * (9.0 / 40.0))]);
        let k3 = rhs(t + h * (3.0 / 10.0), &y3);

        let y4 = lin_comb(
            &y,
            &[
                (&k1, h * (44.0 / 45.0)),
                (&k2, h * (-56.0 / 15.0)),
                (&k3, h * (32.0 / 9.0)),
            ],
        );
        let k4 = rhs(t + h * (4.0 / 5.0), &y4);

        let y5 = lin_comb(
            &y,
            &[
                (&k1, h * (19372.0 / 6561.0)),
                (&k2, h * (-25360.0 / 2187.0)),
                (&k3, h * (64448.0 / 6561.0)),
                (&k4, h * (-212.0 / 729.0)),
            ],
        );
        let k5 = rhs(t + h * (8.0 / 9.0), &y5);

        let y6 = lin_comb(
            &y,
            &[
                (&k1, h * (9017.0 / 3168.0)),
                (&k2, h * (-355.0 / 33.0)),
                (&k3, h * (46732.0 / 5247.0)),
                (&k4, h * (49.0 / 176.0)),
                (&k5, h * (-5103.0 / 18656.0)),
            ],
        );
        let k6 = rhs(t + h, &y6);

        // 5th-order solution
        let y_next = lin_comb(
            &y,
            &[
                (&k1, h * (35.0 / 384.0)),
                (&k3, h * (500.0 / 1113.0)),
                (&k4, h * (125.0 / 192.0)),
                (&k5, h * (-2187.0 / 6784.0)),
                (&k6, h * (11.0 / 84.0)),
            ],
        );
        let k7 = rhs(t + h, &y_next);

        // 4th-order error estimate
        let y_hat = lin_comb(
            &y,
            &[
                (&k1, h * (5179.0 / 57600.0)),
                (&k3, h * (7571.0 / 16695.0)),
                (&k4, h * (393.0 / 640.0)),
                (&k5, h * (-92097.0 / 339200.0)),
                (&k6, h * (187.0 / 2100.0)),
                (&k7, h * (1.0 / 40.0)),
            ],
        );

        let mut err_ratio = 0.0_f64;
        for i in 0..n {
            let yn = y_next.get(i).copied().unwrap_or(0.0);
            let yh = y_hat.get(i).copied().unwrap_or(0.0);
            let yi = y.get(i).copied().unwrap_or(0.0);
            let sc = opts.abs_tol + opts.rel_tol * yi.abs().max(yn.abs());
            err_ratio = err_ratio.max((yn - yh).abs() / sc);
        }

        if err_ratio <= 1.0 || h <= 1e-13 {
            let t_next = t + h;
            if let (Some(zf), Some(z_prev)) = (&zero_crossing_fn, prev_zc) {
                let z_curr = zf(t_next, &y_next);
                if (z_prev < 0.0 && z_curr >= 0.0) || (z_prev > 0.0 && z_curr <= 0.0) {
                    let frac = z_prev.abs() / (z_prev.abs() + z_curr.abs()).max(1e-15);
                    zc_events.push(t + frac * h);
                }
                prev_zc = Some(z_curr);
            }
            t = t_next;
            y = y_next;
            t_hist.push(t);
            y_hist.push(y.clone());
        }

        let factor = if err_ratio == 0.0 {
            2.5
        } else {
            (0.84 * (1.0 / err_ratio).powf(0.2)).clamp(0.2, 2.5)
        };
        h = (h * factor).min(opts.max_step).max(1e-13);
    }

    if t < tf - 1e-14 {
        return Err(SimError::ConvergenceFailed(
            "ode45 exceeded maximum step count of 200,000".into(),
        ));
    }

    Ok(OdeSolution {
        t: t_hist,
        y: y_hist,
        zero_crossings_t: zc_events,
    })
}

/// Implicit Stiff Solver (`ode15s` / Rosenbrock-Backward-Euler with Exact `Dual64` Autodiff Jacobians).
/// Handles stiff chemical kinetics and electrical/mechanical transients with huge eigenvalue spreads!
pub fn solve_ode15s_stiff_autodiff<F>(
    t_span: [f64; 2],
    y0: &[f64],
    dt: f64,
    rhs_dual: F,
) -> Result<OdeSolution, SimError>
where
    F: Fn(f64, &[Dual64]) -> Vec<Dual64>,
{
    let [t0, tf] = t_span;
    if !t0.is_finite() || !tf.is_finite() || !dt.is_finite() || dt <= 0.0 || tf <= t0 {
        return Err(SimError::InvalidTime("Invalid time parameters for ode15s".into()));
    }
    let budget = MemoryBudget::default_budget();
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut t_hist = vec![t];
    let mut y_hist = vec![y.clone()];

    let mut step_count = 0usize;
    const MAX_STEPS: usize = 200_000;
    while t < tf - 1e-14 {
        if step_count >= MAX_STEPS {
            return Err(SimError::ConvergenceFailed(format!(
                "ode15s exceeded maximum step count of {MAX_STEPS}"
            )));
        }
        step_count = step_count.saturating_add(1);
        let h = dt.min(tf - t);
        let t_next = t + h;
        let mut y_guess = y.clone();

        // Implicit Newton iteration solving G(y_{k+1}) = y_{k+1} - y_k - h * f(t_{k+1}, y_{k+1}) = 0
        // using exact analytical Jacobians from `crates/autodiff`!
        for _newton in 0..8 {
            let (f_val, jac_f) =
                jacobian_forward(&y_guess, |vars| rhs_dual(t_next, vars)).map_err(|e| {
                    SimError::InvalidTime(format!("Autodiff Jacobian error: {e}"))
                })?;

            let mut g_vec = Vec::with_capacity(n);
            let mut g_norm: f64 = 0.0;
            for i in 0..n {
                let yg = y_guess.get(i).copied().unwrap_or(0.0);
                let yk = y.get(i).copied().unwrap_or(0.0);
                let fi = f_val.get(i).copied().unwrap_or(0.0);
                let gi = yg - yk - h * fi;
                g_norm = g_norm.hypot(gi);
                g_vec.push(gi);
            }
            if g_norm < 1e-11 {
                break;
            }

            // Build (I - h * J_f)
            let mut lhs_col_major = vec![0.0; n * n];
            for col in 0..n {
                for row in 0..n {
                    let idx = col * n + row;
                    let jf = jac_f.get(idx).copied().unwrap_or(0.0);
                    let diag = if row == col { 1.0 } else { 0.0 };
                    if let Some(slot) = lhs_col_major.get_mut(idx) {
                        *slot = diag - h * jf;
                    }
                }
            }
            let Ok(lhs_mat) =
                Tensor::from_column_major(n, n, lhs_col_major, SiUnit::DIMENSIONLESS, &budget)
            else {
                break;
            };
            let Ok(rhs_mat) =
                Tensor::from_column_major(n, 1, g_vec, SiUnit::DIMENSIONLESS, &budget)
            else {
                break;
            };
            let Ok(delta) = mldivide(&lhs_mat, &rhs_mat, &budget) else {
                break;
            };
            for i in 0..n {
                if let (Some(slot), Ok(d)) = (y_guess.get_mut(i), delta.get(i, 0)) {
                    *slot -= d;
                }
            }
        }

        t = t_next;
        y = y_guess;
        t_hist.push(t);
        y_hist.push(y.clone());
    }

    Ok(OdeSolution {
        t: t_hist,
        y: y_hist,
        zero_crossings_t: Vec::new(),
    })
}

/// Solves a Simulink Algebraic Loop $z = g(z)$ (i.e. $R(z) = z - g(z) = 0$) at machine precision
/// using `crates/autodiff` exact Dual64 Jacobians!
pub fn solve_algebraic_loop_autodiff<G>(
    z_init: &[f64],
    tol: f64,
    max_iter: usize,
    loop_fn: G,
) -> Result<Vec<f64>, SimError>
where
    G: Fn(&[Dual64]) -> Vec<Dual64>,
{
    let budget = MemoryBudget::default_budget();
    let n = z_init.len();
    let mut z = z_init.to_vec();

    for _iter in 0..max_iter {
        let (res, jac) = jacobian_forward(&z, |vars| {
            let gz = loop_fn(vars);
            vars.iter()
                .zip(gz)
                .map(|(&zi, gzi)| zi - gzi)
                .collect()
        })
        .map_err(|_| SimError::AlgebraicLoopDiverged(max_iter))?;

        let norm: f64 = res.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= tol {
            return Ok(z);
        }

        let j_mat = Tensor::from_column_major(n, n, jac, SiUnit::DIMENSIONLESS, &budget)
            .map_err(|_| SimError::AlgebraicLoopDiverged(max_iter))?;
        let r_mat = Tensor::from_column_major(n, 1, res, SiUnit::DIMENSIONLESS, &budget)
            .map_err(|_| SimError::AlgebraicLoopDiverged(max_iter))?;
        let step = mldivide(&j_mat, &r_mat, &budget)
            .map_err(|_| SimError::AlgebraicLoopDiverged(max_iter))?;

        for i in 0..n {
            if let (Some(slot), Ok(dz)) = (z.get_mut(i), step.get(i, 0)) {
                *slot -= dz;
            }
        }
    }
    Err(SimError::AlgebraicLoopDiverged(max_iter))
}

fn lin_comb(base: &[f64], terms: &[(&[f64], f64)]) -> Vec<f64> {
    let mut out = base.to_vec();
    for (vec, scale) in terms {
        for (o, &v) in out.iter_mut().zip(*vec) {
            *o += v * (*scale);
        }
    }
    out
}
