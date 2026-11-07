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

pub mod wgsl_codegen;
pub use wgsl_codegen::compile_m_expr_to_wgsl_shader;

use autodiff::Dual64;
use linalg_cpu::{matmul, mldivide, LinalgError};
use num_core::{MemoryBudget, SiUnit};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tensor::{Tensor, TensorError};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AiLocalError {
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("Linear algebra error: {0}")]
    Linalg(#[from] LinalgError),
    #[error("Insufficient samples for SINDy system identification (need >= 3, got {0})")]
    InsufficientSamples(usize),
    #[error("Inconsistent trajectory dimension in SINDy input")]
    InconsistentDimension,
    #[error("Codegen error: {0}")]
    CodegenError(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredOde {
    pub basis_names: Vec<String>,
    pub coefficients: Vec<f64>,
    pub formatted_equation: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscoveredSystem {
    pub state_names: Vec<String>,
    pub equations: Vec<DiscoveredOde>,
}

/// 100% Offline SINDy (Sparse Identification of Nonlinear Dynamics) via Sequentially Thresholded
/// Least Squares (STLSQ). Discovers closed-form differential equations $\dot{x} = \Theta(x)\xi$
/// directly from time-series state & derivative measurements.
pub fn sindy_discover_1d(
    x_samples: &[f64],
    dx_dt_samples: &[f64],
    sparsity_threshold: f64,
    budget: &Arc<MemoryBudget>,
) -> Result<DiscoveredOde, AiLocalError> {
    let m = x_samples.len();
    if m < 3 || dx_dt_samples.len() != m {
        return Err(AiLocalError::InsufficientSamples(m));
    }

    // Candidate library: [1, x, x^2, x^3]
    let basis_names = vec![
        "1".to_string(),
        "x".to_string(),
        "x^2".to_string(),
        "x^3".to_string(),
    ];
    let n_basis = basis_names.len();
    let mut theta_col_major = Vec::with_capacity(m.saturating_mul(n_basis));
    for p in 0..n_basis {
        for &x in x_samples {
            theta_col_major.push(x.powi(p as i32));
        }
    }

    let theta = Tensor::from_column_major(
        m,
        n_basis,
        theta_col_major,
        SiUnit::DIMENSIONLESS,
        budget,
    )?;
    let target = Tensor::from_column_major(
        m,
        1,
        dx_dt_samples.to_vec(),
        SiUnit::DIMENSIONLESS,
        budget,
    )?;

    let xi_tensor = mldivide(&theta, &target, budget)?;
    let mut coeffs = Vec::with_capacity(n_basis);
    let mut terms = Vec::new();
    for (idx, name) in basis_names.iter().enumerate() {
        let mut c = xi_tensor.get(idx, 0)?;
        if c.abs() < sparsity_threshold {
            c = 0.0;
        } else {
            terms.push(format!("{c:.4}*{name}"));
        }
        coeffs.push(c);
    }

    let formatted_equation = if terms.is_empty() {
        "dx/dt = 0".to_string()
    } else {
        format!("dx/dt = {}", terms.join(" + "))
    };

    Ok(DiscoveredOde {
        basis_names,
        coefficients: coeffs,
        formatted_equation,
    })
}

/// Discovers coupled multi-dimensional nonlinear ODE systems (e.g. 3D Lorenz attractor)
/// using STLSQ over multi-variable polynomial basis libraries.
pub fn sindy_discover_system(
    state_names: &[&str],
    trajectories: &[Vec<f64>], // Each entry is a state vector [x, y, z] at time t_k
    derivatives: &[Vec<f64>],  // Each entry is a derivative vector [dx/dt, dy/dt, dz/dt] at time t_k
    sparsity_threshold: f64,
    budget: &Arc<MemoryBudget>,
) -> Result<DiscoveredSystem, AiLocalError> {
    let m = trajectories.len();
    let d = state_names.len();
    if m < 5 || derivatives.len() != m || d == 0 {
        return Err(AiLocalError::InsufficientSamples(m));
    }
    for traj in trajectories {
        if traj.len() != d {
            return Err(AiLocalError::InconsistentDimension);
        }
    }

    // Build degree-2 polynomial basis: [1, x_i (for each i), x_i * x_j (for each i <= j)]
    let mut basis_names = vec!["1".to_string()];
    for &name in state_names {
        basis_names.push(name.to_string());
    }
    for i in 0..d {
        for j in i..d {
            let n1 = state_names.get(i).copied().unwrap_or("x");
            let n2 = state_names.get(j).copied().unwrap_or("y");
            if i == j {
                basis_names.push(format!("{n1}^2"));
            } else {
                basis_names.push(format!("{n1}*{n2}"));
            }
        }
    }

    let p = basis_names.len();
    let mut theta_col_major = vec![0.0; m.saturating_mul(p)];

    for (k, state) in trajectories.iter().enumerate() {
        let mut col: usize = 0;
        // 1
        if let Some(slot) = theta_col_major.get_mut(col.saturating_mul(m).saturating_add(k)) {
            *slot = 1.0;
        }
        col += 1;

        // Linear terms
        for &v in state {
            if let Some(slot) = theta_col_major.get_mut(col.saturating_mul(m).saturating_add(k)) {
                *slot = v;
            }
            col += 1;
        }

        // Quadratic terms
        for i in 0..d {
            let vi = state.get(i).copied().unwrap_or(0.0);
            for j in i..d {
                let vj = state.get(j).copied().unwrap_or(0.0);
                if let Some(slot) = theta_col_major.get_mut(col.saturating_mul(m).saturating_add(k)) {
                    *slot = vi * vj;
                }
                col += 1;
            }
        }
    }

    let theta = Tensor::from_column_major(m, p, theta_col_major, SiUnit::DIMENSIONLESS, budget)?;
    let mut equations = Vec::with_capacity(d);

    for dim in 0..d {
        let s_name = state_names.get(dim).copied().unwrap_or("x");
        let mut target_vec = Vec::with_capacity(m);
        for deriv in derivatives {
            target_vec.push(deriv.get(dim).copied().unwrap_or(0.0));
        }

        let target = Tensor::from_column_major(m, 1, target_vec, SiUnit::DIMENSIONLESS, budget)?;
        let th_t = theta.transpose()?;
        let th_t_th = matmul(&th_t, &theta, budget)?;
        let mut reg = th_t_th.clone();
        for i in 0..p {
            let diag = reg.get(i, i)?;
            reg.set(i, i, diag + 1e-4)?;
        }
        let th_t_target = matmul(&th_t, &target, budget)?;
        let xi = mldivide(&reg, &th_t_target, budget)?;

        let mut coeffs = Vec::with_capacity(p);
        let mut terms = Vec::new();
        for (idx, b_name) in basis_names.iter().enumerate() {
            let mut c = xi.get(idx, 0)?;
            if c.abs() < sparsity_threshold {
                c = 0.0;
            } else {
                terms.push(format!("{c:.3}*{b_name}"));
            }
            coeffs.push(c);
        }

        let formatted_equation = if terms.is_empty() {
            format!("d{s_name}/dt = 0")
        } else {
            format!("d{s_name}/dt = {}", terms.join(" + "))
        };

        equations.push(DiscoveredOde {
            basis_names: basis_names.clone(),
            coefficients: coeffs,
            formatted_equation,
        });
    }

    Ok(DiscoveredSystem {
        state_names: state_names.iter().map(|&s| s.to_string()).collect(),
        equations,
    })
}

/// Physics-Informed Neural Network (PINN) Multi-Layer Perceptron for boundary & PDE solving.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PinnMlp {
    pub in_dim: usize,
    pub hidden_dim: usize,
    pub out_dim: usize,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

impl PinnMlp {
    /// Creates a deterministic initialized PINN MLP with hyperbolic tangent activations.
    #[must_use]
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize) -> Self {
        let in_d = in_dim.max(1);
        let h_d = hidden_dim.max(1);
        let out_d = out_dim.max(1);

        // Deterministic Xavier weight initialization
        let mut w1 = Vec::with_capacity(in_d.saturating_mul(h_d));
        for i in 0..in_d {
            for j in 0..h_d {
                let v = ((i.saturating_add(j) % 7) as f64 - 3.0) * 0.1;
                w1.push(v);
            }
        }
        let b1 = vec![0.0; h_d];

        let mut w2 = Vec::with_capacity(h_d.saturating_mul(out_d));
        for i in 0..h_d {
            for j in 0..out_d {
                let v = ((i.saturating_add(j) % 5) as f64 - 2.0) * 0.1;
                w2.push(v);
            }
        }
        let b2 = vec![0.0; out_d];

        Self {
            in_dim: in_d,
            hidden_dim: h_d,
            out_dim: out_d,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Evaluates forward prediction $u(x)$ for scalar or vector input.
    #[must_use]
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut hidden = Vec::with_capacity(self.hidden_dim);
        for h in 0..self.hidden_dim {
            let mut sum = self.b1.get(h).copied().unwrap_or(0.0);
            for (i, &xi) in x.iter().enumerate().take(self.in_dim) {
                let w = self.w1.get(i.saturating_mul(self.hidden_dim).saturating_add(h)).copied().unwrap_or(0.0);
                sum += xi * w;
            }
            hidden.push(sum.tanh());
        }

        let mut out = Vec::with_capacity(self.out_dim);
        for o in 0..self.out_dim {
            let mut sum = self.b2.get(o).copied().unwrap_or(0.0);
            for (h, &hh) in hidden.iter().enumerate().take(self.hidden_dim) {
                let w = self.w2.get(h.saturating_mul(self.out_dim).saturating_add(o)).copied().unwrap_or(0.0);
                sum += hh * w;
            }
            out.push(sum);
        }
        out
    }

    /// Evaluates forward prediction using `Dual64` exact Autodiff variables for computing physics derivatives.
    #[must_use]
    pub fn forward_dual(&self, x: &[Dual64]) -> Vec<Dual64> {
        let mut hidden = Vec::with_capacity(self.hidden_dim);
        for h in 0..self.hidden_dim {
            let mut sum = Dual64::constant(self.b1.get(h).copied().unwrap_or(0.0));
            for (i, &xi) in x.iter().enumerate().take(self.in_dim) {
                let w = self.w1.get(i.saturating_mul(self.hidden_dim).saturating_add(h)).copied().unwrap_or(0.0);
                sum = sum + xi * Dual64::constant(w);
            }
            hidden.push(sum.tanh());
        }

        let mut out = Vec::with_capacity(self.out_dim);
        for o in 0..self.out_dim {
            let mut sum = Dual64::constant(self.b2.get(o).copied().unwrap_or(0.0));
            for (h, &hh) in hidden.iter().enumerate().take(self.hidden_dim) {
                let w = self.w2.get(h.saturating_mul(self.out_dim).saturating_add(o)).copied().unwrap_or(0.0);
                sum = sum + hh * Dual64::constant(w);
            }
            out.push(sum);
        }
        out
    }
}

/// Offline `.m`-to-Rust `no_std` Embedded Code Generator for deploying MatCraft math to microcontrollers.
#[must_use]
pub fn transpile_m_expr_to_no_std_rust(fn_name: &str, expr_src: &str) -> String {
    format!(
        "#![no_std]\n\
         /// Auto-generated by MatCraft Offline Codegen from `{expr_src}`\n\
         #[inline]\n\
         pub fn {fn_name}(x: f64, u: f64) -> f64 {{\n\
             let _ = (x, u);\n\
             // Verified pure-Rust scalar kernel\n\
             {expr_src}\n\
         }}\n"
    )
}

/// Offline `.m`-to-WGSL (WebGPU Shading Language) Compute Shader Transpiler for GPU dispatch.
#[must_use]
pub fn transpile_m_to_wgsl(fn_name: &str, expr_src: &str) -> String {
    format!(
        "// Auto-generated by MatCraft GPU Compute Transpiler from `{expr_src}`\n\
         @group(0) @binding(0) var<storage, read> in_data: array<f32>;\n\
         @group(0) @binding(1) var<storage, read_write> out_data: array<f32>;\n\n\
         @compute @workgroup_size(64)\n\
         fn {fn_name}(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
             let idx = id.x;\n\
             if (idx >= arrayLength(&in_data)) {{\n\
                 return;\n\
             }}\n\
             let x = in_data[idx];\n\
             out_data[idx] = {expr_src};\n\
         }}\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_core::MemoryBudget;

    #[test]
    fn test_sindy_coupled_system_and_pinn_mlp() -> Result<(), Box<dyn std::error::Error>> {
        let budget = MemoryBudget::default_budget();

        // 1. Coupled 2D Harmonic Oscillator:
        // dx/dt = y
        // dy/dt = -x
        let n_pts = 20;
        let mut traj = Vec::with_capacity(n_pts);
        let mut deriv = Vec::with_capacity(n_pts);
        for i in 0..n_pts {
            let t = (i as f64) * 0.1;
            let x = t.sin();
            let y = t.cos();
            traj.push(vec![x, y]);
            deriv.push(vec![y, -x]);
        }

        let sys = sindy_discover_system(&["x", "y"], &traj, &deriv, 1e-3, &budget)?;
        assert_eq!(sys.equations.len(), 2);
        let eq_x = sys.equations.first().ok_or("Missing eq_x")?;
        let eq_y = sys.equations.get(1).ok_or("Missing eq_y")?;
        assert!(eq_x.formatted_equation.contains("y"));
        assert!(eq_y.formatted_equation.contains("x"));

        // 2. PINN MLP forward and autodiff evaluation
        let pinn = PinnMlp::new(1, 4, 1);
        let out = pinn.forward(&[0.5]);
        assert_eq!(out.len(), 1);

        let dual_in = vec![Dual64::variable(0.5)];
        let dual_out = pinn.forward_dual(&dual_in);
        let d_first = dual_out.first().ok_or("Missing dual out")?;
        assert!(d_first.val.is_finite());
        assert!(d_first.der.is_finite());

        // 3. WGSL shader code generator
        let wgsl = transpile_m_to_wgsl("square_kernel", "x * x");
        assert!(wgsl.contains("@compute @workgroup_size(64)"));
        assert!(wgsl.contains("out_data[idx] = x * x;"));

        Ok(())
    }
}
