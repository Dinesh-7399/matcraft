use crate::SimError;
use linalg_cpu::mldivide;
use num_core::{MemoryBudget, SiUnit};
use serde::{Deserialize, Serialize};
use tensor::Tensor;

/// Acausal Physical Component (Simscape / Modelica Equivalent).
/// Nodes are 0-based integers where Node `0` is the reference Ground / Mechanical Frame!
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AcausalComponent {
    /// Resistor ($V_p - V_n = R \cdot I$) or Mechanical Damper ($F = b \cdot \Delta v$)
    Resistor {
        node_p: usize,
        node_n: usize,
        resistance_ohm: f64,
    },
    /// Capacitor ($I = C \frac{d(V_p - V_n)}{dt}$) or Mechanical Mass ($F = m \frac{dv}{dt}$)
    Capacitor {
        node_p: usize,
        node_n: usize,
        capacitance_f: f64,
        initial_voltage: f64,
    },
    /// Inductor ($V_p - V_n = L \frac{dI}{dt}$) or Mechanical Spring
    Inductor {
        node_p: usize,
        node_n: usize,
        inductance_h: f64,
        initial_current: f64,
    },
    /// Independent DC Current / Force Source ($I$ flowing from `node_n` into `node_p`)
    CurrentSource {
        node_p: usize,
        node_n: usize,
        current_a: f64,
    },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AcausalNetwork {
    pub num_non_ground_nodes: usize,
    pub components: Vec<AcausalComponent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcausalTrace {
    pub times_sec: Vec<f64>,
    pub node_voltages: Vec<Vec<f64>>,
}

impl AcausalNetwork {
    #[must_use]
    pub const fn new(num_non_ground_nodes: usize) -> Self {
        Self {
            num_non_ground_nodes,
            components: Vec::new(),
        }
    }

    pub fn add_component(&mut self, comp: AcausalComponent) {
        self.components.push(comp);
    }

    /// Solves the Acausal Conservation DAE ($\sum I_{\text{node}} = 0$) using Backward-Euler
    /// companion models (Modified Nodal Analysis / MNA) for unconditional A-stability!
    pub fn simulate_mna(
        &self,
        duration_sec: f64,
        dt_sec: f64,
    ) -> Result<AcausalTrace, SimError> {
        let n = self.num_non_ground_nodes;
        if n == 0 || dt_sec <= 0.0 || duration_sec <= 0.0 {
            return Err(SimError::InvalidTime("Invalid acausal MNA parameters".into()));
        }
        let budget = MemoryBudget::default_budget();
        let steps = ((duration_sec / dt_sec).round() as usize).max(1);

        let mut v_nodes = vec![0.0; n];
        for comp in &self.components {
            if let AcausalComponent::Capacitor {
                node_p,
                node_n,
                initial_voltage,
                ..
            } = *comp
            {
                if node_p > 0 && node_n == 0 {
                    if let Some(slot) = v_nodes.get_mut(node_p - 1) {
                        *slot = initial_voltage;
                    }
                }
            }
        }

        let mut inductor_currents = vec![0.0; self.components.len()];
        for (idx, comp) in self.components.iter().enumerate() {
            if let AcausalComponent::Inductor {
                initial_current, ..
            } = *comp
            {
                if let Some(slot) = inductor_currents.get_mut(idx) {
                    *slot = initial_current;
                }
            }
        }

        let mut times_sec = Vec::with_capacity(steps + 1);
        let mut node_voltages = Vec::with_capacity(steps + 1);
        times_sec.push(0.0);
        node_voltages.push(v_nodes.clone());

        for step in 1..=steps {
            let mut g_mat = vec![0.0; n * n];
            let mut rhs_i = vec![0.0; n];

            for (comp_idx, comp) in self.components.iter().enumerate() {
                match *comp {
                    AcausalComponent::Resistor {
                        node_p,
                        node_n,
                        resistance_ohm,
                    } => {
                        let g = 1.0 / resistance_ohm.max(1e-12);
                        stamp_conductance(&mut g_mat, n, node_p, node_n, g);
                    }
                    AcausalComponent::Capacitor {
                        node_p,
                        node_n,
                        capacitance_f,
                        ..
                    } => {
                        let g_eq = capacitance_f / dt_sec;
                        let vp_prev = read_node(&v_nodes, node_p);
                        let vn_prev = read_node(&v_nodes, node_n);
                        let i_eq = g_eq * (vp_prev - vn_prev);
                        stamp_conductance(&mut g_mat, n, node_p, node_n, g_eq);
                        stamp_current(&mut rhs_i, node_p, node_n, i_eq);
                    }
                    AcausalComponent::Inductor {
                        node_p,
                        node_n,
                        inductance_h,
                        ..
                    } => {
                        let g_eq = dt_sec / inductance_h.max(1e-12);
                        let i_prev = inductor_currents.get(comp_idx).copied().unwrap_or(0.0);
                        stamp_conductance(&mut g_mat, n, node_p, node_n, g_eq);
                        stamp_current(&mut rhs_i, node_n, node_p, i_prev);
                    }
                    AcausalComponent::CurrentSource {
                        node_p,
                        node_n,
                        current_a,
                    } => {
                        stamp_current(&mut rhs_i, node_p, node_n, current_a);
                    }
                }
            }

            let g_tensor = Tensor::from_column_major(n, n, g_mat, SiUnit::DIMENSIONLESS, &budget)
                .map_err(|_| SimError::SingularAcausalNetwork)?;
            let i_tensor = Tensor::from_column_major(n, 1, rhs_i, SiUnit::DIMENSIONLESS, &budget)
                .map_err(|_| SimError::SingularAcausalNetwork)?;
            let v_sol = mldivide(&g_tensor, &i_tensor, &budget)
                .map_err(|_| SimError::SingularAcausalNetwork)?;

            for i in 0..n {
                if let (Some(slot), Ok(val)) = (v_nodes.get_mut(i), v_sol.get(i, 0)) {
                    *slot = val;
                }
            }

            // Update inductor branch currents
            for (comp_idx, comp) in self.components.iter().enumerate() {
                if let AcausalComponent::Inductor {
                    node_p,
                    node_n,
                    inductance_h,
                    ..
                } = *comp
                {
                    let vp = read_node(&v_nodes, node_p);
                    let vn = read_node(&v_nodes, node_n);
                    if let Some(slot) = inductor_currents.get_mut(comp_idx) {
                        *slot += (dt_sec / inductance_h.max(1e-12)) * (vp - vn);
                    }
                }
            }

            times_sec.push((step as f64) * dt_sec);
            node_voltages.push(v_nodes.clone());
        }

        Ok(AcausalTrace {
            times_sec,
            node_voltages,
        })
    }
}

fn read_node(v_nodes: &[f64], node: usize) -> f64 {
    if node == 0 {
        0.0
    } else {
        v_nodes.get(node - 1).copied().unwrap_or(0.0)
    }
}

fn stamp_conductance(g_mat: &mut [f64], n: usize, np: usize, nn: usize, g: f64) {
    if np > 0 {
        let i = np - 1;
        if let Some(s) = g_mat.get_mut(i * n + i) {
            *s += g;
        }
    }
    if nn > 0 {
        let j = nn - 1;
        if let Some(s) = g_mat.get_mut(j * n + j) {
            *s += g;
        }
    }
    if np > 0 && nn > 0 {
        let i = np - 1;
        let j = nn - 1;
        if let Some(s) = g_mat.get_mut(j * n + i) {
            *s -= g;
        }
        if let Some(s) = g_mat.get_mut(i * n + j) {
            *s -= g;
        }
    }
}

fn stamp_current(rhs: &mut [f64], np: usize, nn: usize, current_into_np: f64) {
    if np > 0 {
        if let Some(s) = rhs.get_mut(np - 1) {
            *s += current_into_np;
        }
    }
    if nn > 0 {
        if let Some(s) = rhs.get_mut(nn - 1) {
            *s -= current_into_np;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use autodiff::Dual64;
    use crate::solvers::{
        solve_algebraic_loop_autodiff, solve_ode15s_stiff_autodiff, solve_ode45_dormand_prince,
        Ode45Options,
    };

    #[test]
    fn test_ode45_zero_crossing_stiff_ode15s_and_acausal_rlc(
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 1. ode45 with zero-crossing: free-falling ball y'' = -9.81 from y(0) = 4.905 m hits y=0 at t = 1.0 s!
        let sol = solve_ode45_dormand_prince(
            [0.0, 1.5],
            &[4.905, 0.0],
            Ode45Options::default(),
            |_t, y| {
                let v = y.get(1).copied().unwrap_or(0.0);
                vec![v, -9.81]
            },
            Some(|_t: f64, y: &[f64]| y.first().copied().unwrap_or(0.0)),
        )?;
        let impact_t = sol.zero_crossings_t.first().copied().unwrap_or(0.0);
        assert!((impact_t - 1.0).abs() < 1e-3);

        // 2. Stiff ode15s with Dual64 Autodiff Jacobian: dy/dt = -1000 * y, y(0) = 1.0
        let stiff = solve_ode15s_stiff_autodiff([0.0, 0.1], &[1.0], 0.01, |_t, y| {
            let y0 = y.first().copied().unwrap_or_default();
            vec![Dual64::constant(-1000.0) * y0]
        })?;
        let final_stiff = stiff
            .y
            .last()
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(1.0);
        assert!(final_stiff.abs() < 1e-4);

        // 3. Algebraic loop solver: z = cos(z) -> Dottie number ~0.7390851332
        let alg = solve_algebraic_loop_autodiff(&[0.5], 1e-12, 20, |z| {
            let z0 = z.first().copied().unwrap_or_default();
            vec![z0.cos()]
        })?;
        assert!((alg.first().copied().unwrap_or(0.0) - 0.739_085_133_215).abs() < 1e-9);

        // 4. Acausal Parallel RLC circuit excited by 1A step current source -> steady state V = 0 across inductor!
        let mut net = AcausalNetwork::new(1);
        net.add_component(AcausalComponent::CurrentSource {
            node_p: 1,
            node_n: 0,
            current_a: 1.0,
        });
        net.add_component(AcausalComponent::Resistor {
            node_p: 1,
            node_n: 0,
            resistance_ohm: 10.0,
        });
        net.add_component(AcausalComponent::Capacitor {
            node_p: 1,
            node_n: 0,
            capacitance_f: 0.01,
            initial_voltage: 0.0,
        });
        net.add_component(AcausalComponent::Inductor {
            node_p: 1,
            node_n: 0,
            inductance_h: 0.1,
            initial_current: 0.0,
        });
        let trace = net.simulate_mna(2.0, 0.005)?;
        let v_end = trace
            .node_voltages
            .last()
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(99.0);
        assert!(v_end.abs() < 1e-2);
        Ok(())
    }
}
