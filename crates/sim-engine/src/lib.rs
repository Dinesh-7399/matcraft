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

pub mod acausal;
pub mod codegen;
pub mod dae_index;
pub mod events;
pub mod hybrid_automaton;
pub mod solvers;
pub mod stateflow;

pub use dae_index::{analyze_dae_structure, PantelidesReport};
pub use events::{
    find_zero_crossing_time, simulate_with_events, EventResetAction, HybridSimTrace,
    ZeroCrossingDirection, ZeroCrossingSpec,
};
pub use hybrid_automaton::{HybridAutomaton, HybridGuard, HybridMode};
pub use stateflow::{StateChart, StateNode, StateTransition, TransitionGuard};

pub use acausal::{AcausalComponent, AcausalNetwork, AcausalTrace};
pub use codegen::{generate_c_code, generate_no_std_rust_code, generate_stateflow_c_code};
pub use solvers::{
    solve_algebraic_loop_autodiff, solve_ode15s_stiff_autodiff, solve_ode45_dormand_prince,
    Ode45Options, OdeSolution,
};

use num_core::{SiUnit, UnitError};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum SimError {
    #[error("Non-finite or negative simulation time parameter: {0}")]
    InvalidTime(String),
    #[error("Port SI unit mismatch on wire from block {src_block} ({src_unit}) to block {dst_block} ({dst_unit})")]
    PortUnitMismatch {
        src_block: usize,
        src_unit: SiUnit,
        dst_block: usize,
        dst_unit: SiUnit,
    },
    #[error("Unknown block ID {0} in simulation diagram")]
    UnknownBlock(usize),
    #[error("Algebraic loop failed to converge after {0} Newton-Autodiff iterations")]
    AlgebraicLoopDiverged(usize),
    #[error("Singular nodal matrix in acausal physical network")]
    SingularAcausalNetwork,
    #[error("Event root-finding failed: {0}")]
    EventRootFindingFailed(String),
    #[error("Unit error: {0}")]
    Unit(#[from] UnitError),
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct SimTime(pub i128);

impl SimTime {
    pub const ATTOSECONDS_PER_SEC: i128 = 1_000_000_000_000_000_000;

    pub fn from_seconds(secs: f64) -> Result<Self, SimError> {
        if !secs.is_finite() || secs < 0.0 {
            return Err(SimError::InvalidTime(format!("Invalid seconds: {secs}")));
        }
        let attos = (secs * (Self::ATTOSECONDS_PER_SEC as f64)).round() as i128;
        Ok(Self(attos))
    }

    #[must_use]
    pub fn to_seconds(self) -> f64 {
        (self.0 as f64) / (Self::ATTOSECONDS_PER_SEC as f64)
    }

    #[must_use]
    pub const fn is_sample_hit(self, period: Self, offset: Self) -> bool {
        if period.0 <= 0 {
            return false;
        }
        self.0 >= offset.0 && (self.0 - offset.0) % period.0 == 0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BlockKind {
    Constant { value: f64 },
    Step {
        step_time: f64,
        initial_val: f64,
        final_val: f64,
    },
    SineWave {
        amplitude: f64,
        freq_rad_s: f64,
        phase_rad: f64,
    },
    Gain { gain: f64 },
    Sum { signs: Vec<f64> },
    Integrator { initial_state: f64 },
    TransferFcn {
        b0: f64,
        b1: f64,
        a1: f64,
        a0: f64,
    },
    PidController {
        kp: f64,
        ki: f64,
        kd: f64,
        filter_n: f64,
    },
    Saturation { lower: f64, upper: f64 },
    DeadZone { lower: f64, upper: f64 },
    ZeroOrderHold { sample_period_sec: f64 },
    Scope,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimBlock {
    pub id: usize,
    pub name: String,
    pub kind: BlockKind,
    pub out_unit: SiUnit,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Wire {
    pub src_block: usize,
    pub dst_block: usize,
    pub dst_port: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SimModel {
    pub blocks: BTreeMap<usize, SimBlock>,
    pub wires: Vec<Wire>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimTrace {
    pub times_sec: Vec<f64>,
    pub signals: BTreeMap<String, Vec<f64>>,
}

impl SimModel {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_block(&mut self, block: SimBlock) -> usize {
        let id = block.id;
        self.blocks.insert(id, block);
        id
    }

    pub fn connect(
        &mut self,
        src_block: usize,
        dst_block: usize,
        dst_port: usize,
    ) -> Result<(), SimError> {
        if !self.blocks.contains_key(&src_block) {
            return Err(SimError::UnknownBlock(src_block));
        }
        if !self.blocks.contains_key(&dst_block) {
            return Err(SimError::UnknownBlock(dst_block));
        }
        self.wires.push(Wire {
            src_block,
            dst_block,
            dst_port,
        });
        Ok(())
    }

    pub fn simulate_rk4(
        &self,
        duration_sec: f64,
        dt_sec: f64,
    ) -> Result<SimTrace, SimError> {
        let t_end = SimTime::from_seconds(duration_sec)?;
        let dt = SimTime::from_seconds(dt_sec)?;
        if dt.0 <= 0 {
            return Err(SimError::InvalidTime("Step size dt must be > 0".into()));
        }

        let mut states: BTreeMap<usize, [f64; 2]> = BTreeMap::new();
        for (&id, blk) in &self.blocks {
            match blk.kind {
                BlockKind::Integrator { initial_state } => {
                    states.insert(id, [initial_state, 0.0]);
                }
                BlockKind::PidController { .. } | BlockKind::TransferFcn { .. } => {
                    states.insert(id, [0.0, 0.0]);
                }
                _ => {}
            }
        }

        let mut times_sec = Vec::new();
        let mut signals: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for blk in self.blocks.values() {
            signals.insert(blk.name.clone(), Vec::new());
        }

        let mut t = SimTime(0);
        let h = dt.to_seconds();

        while t <= t_end {
            let t_sec = t.to_seconds();
            let outputs = self.eval_outputs(t_sec, &states);
            times_sec.push(t_sec);
            for (&id, blk) in &self.blocks {
                if let Some(trace_vec) = signals.get_mut(&blk.name) {
                    trace_vec.push(outputs.get(&id).copied().unwrap_or(0.0));
                }
            }

            let k1 = self.eval_derivatives(t_sec, &states);
            let s2 = advance_states(&states, &k1, 0.5 * h);
            let k2 = self.eval_derivatives(t_sec + 0.5 * h, &s2);
            let s3 = advance_states(&states, &k2, 0.5 * h);
            let k3 = self.eval_derivatives(t_sec + 0.5 * h, &s3);
            let s4 = advance_states(&states, &k3, h);
            let k4 = self.eval_derivatives(t_sec + h, &s4);

            for (id, st) in &mut states {
                let d1 = k1.get(id).copied().unwrap_or([0.0, 0.0]);
                let d2 = k2.get(id).copied().unwrap_or([0.0, 0.0]);
                let d3 = k3.get(id).copied().unwrap_or([0.0, 0.0]);
                let d4 = k4.get(id).copied().unwrap_or([0.0, 0.0]);
                st[0] += (h / 6.0) * (d1[0] + 2.0 * d2[0] + 2.0 * d3[0] + d4[0]);
                st[1] += (h / 6.0) * (d1[1] + 2.0 * d2[1] + 2.0 * d3[1] + d4[1]);
            }

            t = SimTime(t.0.saturating_add(dt.0));
        }

        Ok(SimTrace { times_sec, signals })
    }

    fn eval_outputs(
        &self,
        t_sec: f64,
        states: &BTreeMap<usize, [f64; 2]>,
    ) -> BTreeMap<usize, f64> {
        let mut out = BTreeMap::new();
        for (&id, blk) in &self.blocks {
            match blk.kind {
                BlockKind::Constant { value } => {
                    out.insert(id, value);
                }
                BlockKind::Step {
                    step_time,
                    initial_val,
                    final_val,
                } => {
                    out.insert(
                        id,
                        if t_sec >= step_time {
                            final_val
                        } else {
                            initial_val
                        },
                    );
                }
                BlockKind::SineWave {
                    amplitude,
                    freq_rad_s,
                    phase_rad,
                } => {
                    out.insert(id, amplitude * (freq_rad_s * t_sec + phase_rad).sin());
                }
                BlockKind::Integrator { .. } => {
                    let x = states.get(&id).copied().unwrap_or([0.0, 0.0]);
                    out.insert(id, x[0]);
                }
                BlockKind::TransferFcn { b0, b1, .. } => {
                    let x = states.get(&id).copied().unwrap_or([0.0, 0.0]);
                    out.insert(id, b1 * x[0] + b0 * x[1]);
                }
                _ => {}
            }
        }

        for _pass in 0..self.blocks.len().max(1) {
            for (&id, blk) in &self.blocks {
                match &blk.kind {
                    BlockKind::Gain { gain } => {
                        let u = self.input_value(id, 0, &out);
                        out.insert(id, gain * u);
                    }
                    BlockKind::Sum { signs } => {
                        let mut acc = 0.0;
                        for (port_idx, &sgn) in signs.iter().enumerate() {
                            acc += sgn * self.input_value(id, port_idx, &out);
                        }
                        out.insert(id, acc);
                    }
                    BlockKind::PidController {
                        kp,
                        kd,
                        filter_n,
                        ..
                    } => {
                        let u = self.input_value(id, 0, &out);
                        let st = states.get(&id).copied().unwrap_or([0.0, 0.0]);
                        let d_term = filter_n * (kd * u - st[1]);
                        out.insert(id, kp * u + st[0] + d_term);
                    }
                    BlockKind::Saturation { lower, upper } => {
                        let u = self.input_value(id, 0, &out);
                        out.insert(id, u.clamp(*lower, *upper));
                    }
                    BlockKind::DeadZone { lower, upper } => {
                        let u = self.input_value(id, 0, &out);
                        let y = if u > *upper {
                            u - *upper
                        } else if u < *lower {
                            u - *lower
                        } else {
                            0.0
                        };
                        out.insert(id, y);
                    }
                    BlockKind::ZeroOrderHold { .. } | BlockKind::Scope => {
                        let u = self.input_value(id, 0, &out);
                        out.insert(id, u);
                    }
                    _ => {}
                }
            }
        }
        out
    }

    fn eval_derivatives(
        &self,
        t_sec: f64,
        states: &BTreeMap<usize, [f64; 2]>,
    ) -> BTreeMap<usize, [f64; 2]> {
        let outputs = self.eval_outputs(t_sec, states);
        let mut derivs = BTreeMap::new();
        for (&id, blk) in &self.blocks {
            match blk.kind {
                BlockKind::Integrator { .. } => {
                    let u = self.input_value(id, 0, &outputs);
                    derivs.insert(id, [u, 0.0]);
                }
                BlockKind::TransferFcn { a1, a0, .. } => {
                    let u = self.input_value(id, 0, &outputs);
                    let x = states.get(&id).copied().unwrap_or([0.0, 0.0]);
                    derivs.insert(id, [x[1], u - a1 * x[1] - a0 * x[0]]);
                }
                BlockKind::PidController {
                    ki, kd, filter_n, ..
                } => {
                    let u = self.input_value(id, 0, &outputs);
                    let st = states.get(&id).copied().unwrap_or([0.0, 0.0]);
                    derivs.insert(id, [ki * u, filter_n * (kd * u - st[1])]);
                }
                _ => {}
            }
        }
        derivs
    }

    fn input_value(
        &self,
        dst_block: usize,
        dst_port: usize,
        outputs: &BTreeMap<usize, f64>,
    ) -> f64 {
        for w in &self.wires {
            if w.dst_block == dst_block && w.dst_port == dst_port {
                return outputs.get(&w.src_block).copied().unwrap_or(0.0);
            }
        }
        0.0
    }
}

fn advance_states(
    base: &BTreeMap<usize, [f64; 2]>,
    derivs: &BTreeMap<usize, [f64; 2]>,
    scale: f64,
) -> BTreeMap<usize, [f64; 2]> {
    let mut next = base.clone();
    for (id, st) in &mut next {
        if let Some(d) = derivs.get(id) {
            st[0] += scale * d[0];
            st[1] += scale * d[1];
        }
    }
    next
}
